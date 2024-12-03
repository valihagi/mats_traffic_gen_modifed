import os
import pickle
import shutil
import time
from copy import deepcopy

import numpy as np
import optree
import torch
from metadrive import ScenarioEnv
from metadrive.policy.idm_policy import IDMPolicy
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from metadrive.scenario import ScenarioDescription
from tensorflow import Tensor

from trafficgen.trafficgen.init.model.tg_init import initializer
from trafficgen.trafficgen.traffic_generator.traffic_generator import TrafficGen
from trafficgen.trafficgen.traffic_generator.utils.data_utils import process_data_to_internal_format, process_agent
from trafficgen.trafficgen.traffic_generator.utils.vis_utils import draw, draw_seq
from trafficgen.trafficgen.utils.config import load_config_init
from trafficgen.trafficgen.utils.get_md_data import metadrive_scenario_to_init_data
from trafficgen.trafficgen.utils.utils import WaymoAgent, process_map, rotate
from ued.buffer import PrioritizedLevelReplayBuffer


class ScenarioEditor:

    def __init__(
            self,
            init_model_path: str,
            trajectory_model_path: str,
            device: str = "cpu"
    ):
        cfg = load_config_init("local")
        self.traffic_gen = TrafficGen(cfg)
        self._device = device

    def edit_scenario(self, scenario: ScenarioDescription) -> ScenarioDescription:
        # Remove, modify or add track
        new_scenario = ScenarioDescription(scenario)
        edit_type = np.random.choice(["remove", "modify", "add"])
        edit_type = "add"
        if edit_type == "remove":
            new_scenario = self._remove_track(new_scenario)
        else:
            new_scenario = self._add_track(new_scenario)
        if "-" in scenario["id"]:
            parent_id = scenario["id"].split("-")[1]
        else:
            parent_id = scenario["id"]
        new_scenario["id"] = parent_id + f"-{time.time_ns()}"
        new_scenario[new_scenario.METADATA]["scenario_id"] = new_scenario["id"]
        ScenarioDescription.update_summaries(new_scenario)
        return new_scenario

    def _remove_track(self, scenario: ScenarioDescription) -> ScenarioDescription:
        tracks = set(scenario[scenario.TRACKS].keys()) - {scenario[scenario.METADATA]["sdc_id"]}
        track_to_remove = np.random.choice(list(tracks))
        _ = scenario[scenario.TRACKS].pop(track_to_remove)
        return scenario

    def _add_track(self, scenario: ScenarioDescription) -> ScenarioDescription:
        data = metadrive_scenario_to_init_data(scenario)
        batch = process_data_to_internal_format(data, max_agent_num=len(scenario[ScenarioDescription.TRACKS]))
        batch = optree.tree_map(lambda x: np.expand_dims(x, axis=0), batch)[0]

        for key in batch.keys():
            if isinstance(batch[key], np.ndarray):
                batch[key] = torch.Tensor(batch[key])
            if isinstance(batch[key], torch.DoubleTensor):
                batch[key] = batch[key].float()
            if isinstance(batch[key], torch.Tensor) and self._device == 'cuda':
                batch[key] = batch[key].cuda()
            if 'mask' in key:
                batch[key] = batch[key].to(bool)

        # Call the initialization model
        # The output is a dict with these keys: center, rest, bound, agent
        num_processed_agents = batch["agent_mask"].sum()
        with (torch.no_grad()):
            model_output = self.traffic_gen.init_model.inference(
                data=batch,
                context_num=num_processed_agents,
                pred_num=num_processed_agents + 1
            )
            agent, agent_mask = WaymoAgent.from_list_to_array(model_output['agent'])
            for key in batch.keys():
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].cpu().numpy()

            gt_agent = batch['other']['gt_agent'][0][:, batch['other']['gt_agent_mask'][0, 0] == 1]
            perm = self._find_permuation(agent[agent_mask][:-1], gt_agent[0])
            gt_agent = gt_agent[:, perm]
            output = {}
            output['context_num'] = num_processed_agents
            output['all_agent'] = agent
            output['agent_mask'] = agent_mask
            output['lane'] = batch['other']['lane'][0]
            output['unsampled_lane'] = batch['other']['unsampled_lane'][0]
            output['traf'] = optree.tree_map(lambda x: x[0], batch['other']['traf'])
            output['gt_agent'] = gt_agent  # batch['other']['gt_agent'][0]
            output['gt_agent_mask'] = batch['other']['gt_agent_mask'][0]
            output['center_info'] = batch.get("center_info", {})
            trajectories = self.traffic_gen.inference_control(
                data=output,
                length=scenario[ScenarioDescription.LENGTH],
                context_num=num_processed_agents
            )

        center = batch['center'][0]
        rest = batch['rest'][0]
        bound = batch['bound'][0]

        # visualize generated traffic snapshots
        if True:
            os.makedirs("visualizations/scenarios", exist_ok=True)
            output_path = os.path.join("visualizations/scenarios", f'scenario_{scenario["id"]}.png')
            draw(center, model_output['agent'][:-1], other=rest, edge=bound, save=True, path=output_path)
            output_path = os.path.join("visualizations", f'scenario_{scenario["id"]}_edited.png')
            #draw(center, model_output['agent'], other=rest, edge=bound, save=True, path=output_path)

            ind = list(range(0, scenario[ScenarioDescription.LENGTH], 10))
            agent = trajectories[ind]

            agent_0 = agent[0]
            agent0_list = []
            agent_num = agent_0.shape[0]
            for a in range(agent_num):
                agent0_list.append(WaymoAgent(agent_0[[a]]))

            print("Number of agents:", agent_num)
            print("number of processed agents:", num_processed_agents)
            print("Number of scenario agents:", len(scenario[ScenarioDescription.TRACKS]))

            cent, cent_mask, bound, bound_mask, _, _, rest, _ = process_map(
                output['lane'][np.newaxis], [output['traf'][0]],
                center_num=1000,
                edge_num=500,
                offest=0,
                lane_range=60
            )
            img_path = os.path.join("visualizations/scenarios", f'traj_{scenario["id"]}.png')
            draw_seq(
                cent[0], agent0_list, agent[..., :2], edge=bound[0], other=rest[0], path=img_path, save=True
            )

        next_agent_id = max([int(track) for track in scenario[ScenarioDescription.TRACKS]]) + 1
        agent_traj = trajectories[:, -1]
        sdc_idx = scenario[ScenarioDescription.METADATA]["sdc_id"]
        ego_heading = scenario[ScenarioDescription.TRACKS][sdc_idx]["state"]["heading"][0]
        ego_position = scenario[ScenarioDescription.TRACKS][sdc_idx]["state"]["position"][0]
        agent_traj[:, :2] = rotate(agent_traj[:, 0], agent_traj[:, 1], ego_heading)
        agent_traj[:, 2:4] = rotate(agent_traj[:, 2], agent_traj[:, 3], ego_heading)
        agent_traj[:, 4]  += ego_heading
        agent_traj[:, :2] += ego_position[:2]

        track = self._to_track(str(next_agent_id), agent_traj)
        scenario[ScenarioDescription.TRACKS][str(next_agent_id)] = track
        return scenario

    def _to_track(self, id: str, trajectory: Tensor) -> dict:
        return {
            "type": "VEHICLE",
            "metadata": {
                "dataset": "waymo",
                "object_id": id,
                "track_length": trajectory.shape[0],
                "type": "VEHICLE"
            },
            "state": {
                "heading": trajectory[:, 4],
                "position": trajectory[:, :2],
                "velocity": trajectory[:, 2:4],
                "length": trajectory[:, 5],
                "width": trajectory[:, 6],
                "height": trajectory[:, 7],
                "valid": np.ones(trajectory.shape[0], dtype=bool)
            }
        }

    def _find_permuation(self, agents, gt_agents):
        permutation = []
        for i in range(agents.shape[0]):
            idx = np.argmin((np.abs(agents[i] - gt_agents).sum(-1)))
            permutation.append(idx)
        return permutation


if __name__ == '__main__':
    dataset_folder = "datasets/metadrive"

    editor = ScenarioEditor(
        init_model_path="trafficgen/trafficgen/traffic_generator/ckpt/init.ckpt",
        trajectory_model_path="path/to/trajectory/model"
    )
    plr = PrioritizedLevelReplayBuffer(
        directory="datasets/plr_buffer",
        max_size=1000,
        replay_rate=0.1,
        p=0.6,
        temperature=0.4,
        update_sampler=False,
        load_existing=True
    )

    #with open("datasets/metadrive/0.pkl", "rb") as f:
    #    scenario = pickle.load(f)
    #plr.add(scenario, score=0)




    #for i in range(20):
    #    scenario = editor.edit_scenario(scenario)
    #    plr.add(scenario, score=0)
        #env.num_scenarios = len(plr)
        #env.engine.data_manager.num_scenarios = len(plr)
        #env.config["num_scenarios"] = len(plr)

    env = ScenarioEnv(dict(
        use_render=True,
        agent_policy=ReplayEgoCarPolicy,
        data_directory="datasets/plr_buffer",
        start_scenario_index=0,
        num_scenarios=len(plr),
        sequential_seed=False,
        force_reuse_object_name=True,
        horizon=50,
        no_light=True,
        reactive_traffic=True,
        vehicle_config=dict(
            lidar=dict(num_lasers=30, distance=50, num_others=3),
            side_detector=dict(num_lasers=30),
            lane_line_detector=dict(num_lasers=12)),
    ))

    for i in range(20):
        env.reset(i)
        done = False
        while not done:
            obs, reward, done, trun, info = env.step([0, 0])
            env.render(
                screen_record=True,
                screen_size=(700,700)
            )
        #env.top_down_renderer.generate_gif(f"visualizations/scenario_{i}.gif")
