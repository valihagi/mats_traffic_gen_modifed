from datetime import datetime
import enum
import math
import pickle
import random

import bezier
import carla
import gymnasium
from helpers import calc_euclidian_distance, calculate_near_miss_ttc, calculate_risk_coefficient, calculate_ttc, compute_bounding_box_corners, plot_stuff, plot_trajectory_vs_network
import numpy as np
import optree
import tensorflow as tf
import torch
from mats_gym.envs.scenario_env_wrapper import BaseScenarioEnvWrapper
from pettingzoo.utils.env import AgentID, ObsType, ActionType
from scenic.domains.driving.roads import Network
from scenic.simulators.carla.utils.utils import scenicToCarlaLocation
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from cat.advgen.adv_generator import get_polyline_yaw, Intersect, get_polyline_vel
from cat.advgen.adv_utils import process_data
from cat.advgen.modeling.vectornet import VectorNet
from scipy.spatial import cKDTree
import shapely.geometry
from scenic.core.vectors import Vector


class RoadGraphTypes(enum.Enum):
    UNKNOWN = 0
    LANE_FREEWAY = 1
    LANE_SURFACE_STREET = 2
    LANE_BIKE_LANE = 3
    ROAD_LINE_BROKEN_SINGLE_WHITE = 6
    ROAD_LINE_SOLID_SINGLE_WHITE = 7
    ROAD_LINE_SOLID_DOUBLE_WHITE = 8
    ROAD_LINE_BROKEN_SINGLE_YELLOW = 9
    ROAD_LINE_BROKEN_DOUBLE_YELLOW = 10
    ROAD_LINE_SOLID_SINGLE_YELLOW = 11
    ROAD_LINE_SOLID_DOUBLE_YELLOW = 12
    ROAD_LINE_PASSING_DOUBLE_YELLOW = 13
    ROAD_EDGE_BOUNDARY = 15
    ROAD_EDGE_MEDIAN = 16
    STOP_SIGN = 17
    CROSSWALK = 18
    SPEED_BUMP = 19

class GLOBALS(enum.Enum):
    DEFAULT_FOR_TTC = 100000
    MAX_TIME_FOR_TTC = 20
    TTC_TIMESTEP = 0.001
    TTC_THRESHOLD = 2


class AgentTypes(enum.Enum):
    VEHICLE = 1
    PEDESTRIAN = 2
    CYCLIST = 3
    OTHERS = 4

    @staticmethod
    def from_carla_type(actor: carla.Actor) -> "AgentTypes":
        if actor.type_id.startswith("vehicle"):
            if actor.attributes["base_type"] == "bicycle":
                return AgentTypes.CYCLIST
            return AgentTypes.VEHICLE
        elif actor.type_id.startswith("walker"):
            return AgentTypes.PEDESTRIAN
        else:
            return AgentTypes.OTHERS


class TrafficLightStates(enum.Enum):
    LANE_STATE_UNKNOWN = 0
    LANE_STATE_ARROW_STOP = 1
    LANE_STATE_ARROW_CAUTION = 2
    LANE_STATE_ARROW_GO = 3
    LANE_STATE_STOP = 4
    LANE_STATE_CAUTION = 5
    LANE_STATE_GO = 6
    LANE_STATE_FLASHING_STOP = 7
    LANE_STATE_FLASHING_CAUTION = 8


def visualize_traj(x, y, yaw, width, length, color, skip=5):
    world = CarlaDataProvider.get_world()
    map = CarlaDataProvider.get_map()
    color = carla.Color(*color)
    ts = np.arange(0, 0.1 * len(x), 0.1)
    for t in range(0, len(x), skip):
        x_t, y_t, yaw_t = x[t], y[t], yaw[t]
        wp = map.get_waypoint(carla.Location(x=x_t.item(), y=y_t.item()))
        loc = carla.Location(x=x_t.item(), y=y_t.item(), z=wp.transform.location.z)
        bbox = carla.BoundingBox(
            loc,
            carla.Vector3D(x=length / 2, y=width / 2, z=0.05)
        )
        front = map.get_waypoint(carla.Location(
            x=x_t.item() + 0.5 * length * np.cos(yaw_t.item()),
            y=y_t.item() - 0.5 * length * np.sin(yaw_t.item())
        ))

        pitch = np.arcsin((front.transform.location.z - loc.z) / (length / 2))
        pitch = np.rad2deg(pitch)
        world.debug.draw_point(loc, size=0.1, color=color, life_time=0)
        world.debug.draw_box(bbox, carla.Rotation(yaw=yaw_t.item(), pitch=pitch),
                             thickness=0.1, color=color,
                             life_time=0)
        time = ts[t]
        loc += carla.Location(z=0.2)
        world.debug.draw_string(loc, f"{time:.1f}s", draw_shadow=True, color=color,
                                life_time=1000000)


def get_full_trajectory(id, features, with_yaw=False, future=None):
    trajs = []
    for time in ["past", "current", "future"]:
        x = features[f"state/{time}/x"][id]
        y = features[f"state/{time}/y"][id]
        traj = [x, y]
        if with_yaw:
            traj.append(features[f"state/{time}/bbox_yaw"][id])
        trajs.append(np.stack(traj, axis=1))
    if future is not None:
        trajs[-1] = future
    return np.concatenate(trajs, axis=0)

def setup_collision_detector(sensor_bp_library, ego_vehicle):
    collision_sensor = None
    if ego_vehicle:
        coll_sensor = sensor_bp_library.find('sensor.other.collision')
        collision_sensor = ego_vehicle.get_world().spawn_actor(coll_sensor, carla.Transform(), attach_to=ego_vehicle)
    return collision_sensor


class AdversarialTrainingWrapper(BaseScenarioEnvWrapper):

    def __init__(
            self,
            env,
            args,
            model_path: str,
            ego_agent: str = None,
            adv_agents: str | list[str] = None,
            max_samples: int = 20000,
            resolution: float = 0.5,
            sample_frequency: int = 2
    ):
        super().__init__(env)
        self._ego_agent = ego_agent or self.agents[0]
        adv_agents = adv_agents or self.agents[1:]
        if isinstance(adv_agents, str):
            adv_agents = [adv_agents]
        self._adv_agent = adv_agents[0]
        self._topology: list[tuple[carla.Waypoint, carla.Waypoint]] = None
        self._map = None
        self._network = None
        self._max_samples = max_samples
        self._resolution = resolution
        self._sample_frequency = sample_frequency
        self._trajectories = {}
        self._traffic_light_states = {}
        self._lane_ids = []
        self._args = args
        self._model = VectorNet(args).to("cpu")
        self._model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.kpis = {"ttc": [],
                     "risk_coefficient": [],
                     "adv_acc": [],
                     "drac": [],
                     "mttc": [],
                     "near_miss_ttc": [],
                     "euclidean_distance": []}
        self.parameters = {}

    def reset(
            self, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[AgentID, ObsType], dict[AgentID, dict]]:
        options = options or {}

        if options.get("save", False):
            self._save()

        obs, info = self.env.reset(seed, options)
        world: carla.World = CarlaDataProvider.get_world()
        map: carla.Map = CarlaDataProvider.get_map()

        # collission sensor
        self.coll = False
        self._ego_actor = self.actors[self._ego_agent]
        self.coll_sensor = setup_collision_detector(self.client.get_world().get_blueprint_library(), self._ego_actor)
        def function_handler(event):
            self.coll = True
        self.coll_sensor.listen(function_handler)

        if self._map is None or map.name != self._map.name:
            self._map = map
            self._topology = map.get_topology()
            town = map.name.split("/")[-1]
            self._network = Network.fromFile(f"scenarios/maps/{town}.xodr", useCache=True)
            self._lane_ids = []
            self.roadgraph = self._get_roadgraph("ego_vehicle")
            self.filter_road_graph_for_odd_checking(self._get_roadgraph_features(self._max_samples))

        if options.get("load", False):
            data = self._load()
            self._trajectories = data["trajectories"]
            self._traffic_light_states = data["traffic_light_states"]
            self._traffic_light_locations = {
                k: carla.Location(x=tl[0], y=tl[1], z=tl[2])
                for k, tl
                in data["traffic_light_locations"].items()
            }
            self._lane_ids = data["lane_ids"]
            self.roadgraph = data["roadgraph"]

        """if options.get("random", False):
            adv_traj = self._generate_random_adversarial_route(80)
            info[self._adv_agent] = {}
            info[self._adv_agent]["adv_trajectory"] = adv_traj

        if options.get("parametrized", False):
            adv_traj = self._generate_parametrized_adversarial_route(80)
            info[self._adv_agent] = {}
            info[self._adv_agent]["adv_trajectory"] = adv_traj"""

            

        if options.get("adversarial", False):
            # make option for random trajectory
            self._update_actor_ids()
            features, adv_traj = self._generate_adversarial_route()
            if self._adv_agent not in info:
                info[self._adv_agent] = {}
            info[self._adv_agent]["adv_trajectory"] = adv_traj
            #info[self._adv_agent]["all_trajectories"] = adv_traj
            #info[self._adv_agent]["is_critical"] = found_intersection

        self._traffic_light_locations = {
            traffic_light.id: traffic_light.get_location()
            for traffic_light
            in world.get_actors().filter("traffic.traffic_light")
        }
        # initialize track records
        self._trajectories = {
            actor_id: [self._get_agent_state(self.actors[actor_id])]
            for actor_id
            in self.agents
        }
        self._traffic_light_states = {
            traffic_light.id: [self._get_traffic_light_states(traffic_light)]
            for traffic_light
            in world.get_actors().filter("traffic.traffic_light")
        }

        info["kpis"] = self.kpis
        self.reset_kpis()

        return obs, info
    
    def get_kpis(self):
        return self.kpis
    
    def reset_kpis(self):
        self.kpis = {"ttc": [],
                     "risk_coefficient": [],
                     "adv_acc": [],
                     "drac": [],
                     "mttc": [],
                     "near_miss_ttc": [],
                     "euclidean_distance": []}
    
    def get_ttc_as_dict(self):
        return {"ttc": self.kpis["ttc"]}

    def _update_actor_ids(self):
        for tl in CarlaDataProvider.get_world().get_actors().filter("traffic.traffic_light"):
            loc = tl.get_location()
            prev_tl_id = min(self._traffic_light_locations,
                             key=lambda tl_id: loc.distance(self._traffic_light_locations[tl_id]))
            self._traffic_light_locations[tl.id] = self._traffic_light_locations.pop(prev_tl_id)
            self._traffic_light_states[tl.id] = self._traffic_light_states.pop(prev_tl_id)

    def _load(self):
        with open("trajectories.pkl", "rb") as file:
            data = pickle.load(file)
        return data

    def _save(self):
        with open("trajectories.pkl", "wb") as file:
            data = {
                "trajectories": self._trajectories,
                "traffic_light_states": self._traffic_light_states,
                "traffic_light_locations": {k: [tl.x, tl.y, tl.z] for k, tl in
                                            self._traffic_light_locations.items()},
                "lane_ids": self._lane_ids,
                "roadgraph": self.roadgraph
            }
            pickle.dump(data, file)

    def step(
            self, actions: dict[AgentID, ActionType]
    ) -> tuple[
        dict[AgentID, ObsType],
        dict[AgentID, float],
        dict[AgentID, bool],
        dict[AgentID, bool],
        dict[AgentID, dict],
    ]:
        obs, rewards, terminated, truncated, info = self.env.step(actions)
        world = CarlaDataProvider.get_world()

        if info["__common__"]["current_frame"] % self._sample_frequency == 0:
            for track in self._trajectories:
                actor = self.actors[track]
                self._trajectories[track].append(self._get_agent_state(actor))
            for id in self._traffic_light_states:
                traffic_light = world.get_actor(id)
                self._traffic_light_states[id].append(self._get_traffic_light_states(traffic_light))

        ego = self.actors[self._ego_agent]
        adv = self.actors[self._adv_agent]

        # calculate KPIs and add them to self.kpis
        self.calculate_kpis(ego, adv)

        info["kpis"] = self.kpis

        spectator = world.get_spectator()
        ego_loc = ego.get_location()
        spectator.set_transform(carla.Transform(
            carla.Location(ego_loc.x, ego_loc.y, ego_loc.z + 50),
            carla.Rotation(pitch=-90)
        ))
        return obs, rewards, terminated, truncated, info

    def observe(self, agent: str) -> dict:
        obs = super().observe(agent)
        obs.update(self._get_roadgraph(agent))
        return obs

    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        space = self.env.observation_space(agent)
        N = self._max_samples
        roadgraph_space = gymnasium.spaces.Dict({
            "dir": gymnasium.spaces.Box(low=-1, high=1, shape=(N, 3), dtype=np.float32),
            "id": gymnasium.spaces.Box(low=0, high=N, shape=(N, 1), dtype=np.int64),
            "type": gymnasium.spaces.Box(low=0, high=30, shape=(N, 1), dtype=np.int64),
            "valid": gymnasium.spaces.Box(low=0, high=1, shape=(N, 1), dtype=np.int64),
            "xyz": gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(N, 3), dtype=np.float32)
        })
        return gymnasium.spaces.Dict({
            **space,
            "roadgraph": roadgraph_space
        })

    def _get_roadgraph_features(self, num_samples: int) -> dict:
        samples = {}
        ego_loc = self.actors[self._ego_agent].get_location()
        points = self.roadgraph["xyz"]
        pos = np.array([ego_loc.x, ego_loc.y, ego_loc.z])
        dists = np.linalg.norm(points - pos, axis=1)
        idxs = np.argsort(dists)[:num_samples]
        idxs.sort()
        for k, feats in self.roadgraph.items():
            samples[f"roadgraph_samples/{k}"] = feats[idxs]

        # colors = [
        #     (5, 0, 0),
        #     (0, 5, 0),
        #     (0, 0, 5),
        #     (5, 5, 0),
        #     (0, 5, 5),
        #     (5, 0, 5),
        #     (5, 5, 5),
        #     (0, 0, 0),
        # ]
        # world = CarlaDataProvider.get_world()
        # ids = samples["roadgraph_samples/id"]
        # start = 0
        # for i in range(1, len(ids)):
        #     if ids[i] != ids[start]:
        #         points = samples["roadgraph_samples/xyz"][start:i]
        #         start = i
        #         color = carla.Color(*random.choice(colors))
        #         for s, e in zip(points[:-1], points[1:]):
        #             s, e = s.tolist(), e.tolist()
        #
        #             world.debug.draw_line(
        #                 carla.Location(x=s[0], y=s[1], z=s[2]),
        #                 carla.Location(x=e[0], y=e[1], z=e[2]),
        #                 thickness=0.1,
        #                 color=color,
        #                 life_time=0
        #             )
        #
        #
        # world.tick()
        # settings = world.get_settings()
        # settings.synchronous_mode = False
        # world.apply_settings(settings)
        return samples

    def _sample_trajectories(self, features):
        for time in ["past", "current", "future"]:
            x, y = features[f"state/{time}/x"], features[f"state/{time}/y"]
            vx, vy = features[f"state/{time}/velocity_x"], features[f"state/{time}/velocity_y"]

            features[f"state/{time}/x"] = y
            features[f"state/{time}/y"] = x
            features[f"state/{time}/velocity_x"] = vy
            features[f"state/{time}/velocity_y"] = vx

            yaw = np.deg2rad(features[f"state/{time}/bbox_yaw"])
            ur_quad = np.bitwise_and(yaw >= 0, yaw < np.pi / 2)
            lr_quad = np.bitwise_and(yaw >= np.pi / 2, yaw < np.pi)
            ll_quad = np.bitwise_and(yaw >= -np.pi, yaw < -np.pi / 2)
            ul_quad = np.bitwise_and(yaw >= -np.pi / 2, yaw < 0)

            yaw[ur_quad] = np.pi / 2 - yaw[ur_quad]
            yaw[lr_quad] = np.pi / 2 - yaw[lr_quad]
            yaw[ll_quad] = np.pi / 2 - yaw[ll_quad] - 2 * np.pi
            yaw[ul_quad] = np.pi / 2 - yaw[ul_quad]

            features[f"state/{time}/bbox_yaw"] = yaw
            features[f"state/{time}/vel_yaw"] = -np.deg2rad(features[f"state/{time}/vel_yaw"])

        for time in ["past", "current"]:
            x, y = features[f"traffic_light_state/{time}/x"], features[
                f"traffic_light_state/{time}/y"]
            features[f"traffic_light_state/{time}/x"] = y
            features[f"traffic_light_state/{time}/y"] = x

        features["roadgraph_samples/xyz"] = features["roadgraph_samples/xyz"][:, [1, 0, 2]]
        features["roadgraph_samples/dir"] = features["roadgraph_samples/dir"][:, [1, 0, 2]]

        def to_tensor(x: np.ndarray):
            if x.dtype == np.float64:
                x = x.astype(np.float32)
            return tf.convert_to_tensor(x)

        data = optree.tree_map(to_tensor, features)
        batch_data = process_data(data, self._args)
        with torch.no_grad():
            pred_trajectory, pred_score, _ = self._model(batch_data[0], 'cpu')
        return pred_trajectory, pred_score


    def _score_trajectories(self, pred_trajectory, pred_score, features):
        return pred_score

    def _generate_adversarial_route(self):
        features, ego_route = self._get_features()

        pred_trajectory, pred_score = self._sample_trajectories(features)
        #remove trajs that are not on roadgraph
        scores = self._score_trajectories(pred_trajectory, pred_score, features)
        adv_traj_id, adv_traj = self._select_colliding_trajectory(features, pred_score, pred_trajectory)

        # world.apply_settings(settings)
        # world.tick()

        return features, adv_traj

    def _select_colliding_trajectory(self, features, pred_score, pred_trajectory):
        trajs_OV = pred_trajectory[self.agents.index(self._adv_agent)]
        probs_OV = pred_score[self.agents.index(self._adv_agent)]
        probs_OV[6:] = probs_OV[6]
        probs_OV = np.exp(probs_OV)
        probs_OV = probs_OV / np.sum(probs_OV)
        res = np.zeros(pred_trajectory.shape[1])
        min_dist = np.full(pred_trajectory.shape[1], fill_value=1000000)
        ego = self.actors[self._ego_agent]
        adversary = (set(self.agents) - {self._ego_agent}).pop()
        adversary = self.actors[adversary]
        adv_width, adv_length = adversary.bounding_box.extent.y * 2, adversary.bounding_box.extent.x * 2
        width, length = ego.bounding_box.extent.y * 2, ego.bounding_box.extent.x * 2
        trajs_AV = np.concatenate([
            features["state/future/x"][self.agents.index(self._ego_agent)].reshape(-1, 1),
            features["state/future/y"][self.agents.index(self._ego_agent)].reshape(-1, 1)
        ], axis=1)
        trajs_AV = np.expand_dims(trajs_AV, axis=0)
        for j, prob_OV in enumerate(probs_OV):
            P4 = 1
            P1 = prob_OV
            traj_OV = trajs_OV[j][::5]
            #---------------------
            traj_OV_plot = trajs_OV[j]
            full_adv_traj = get_full_trajectory(self.agents.index(self._adv_agent), features, future=traj_OV_plot)[:,::-1]
            full_adv_traj = np.concatenate([
                full_adv_traj,
                np.rad2deg(get_polyline_yaw(full_adv_traj)).reshape(-1, 1)
            ], axis=1)
            # CHeck if on roadgraph here:
            if not self.check_on_roadgraph(full_adv_traj, j, traj_OV):
                P4 = 0 #can be used to only allow trajs that are on the roadgraph
            
                """visualize_traj(
                    full_adv_traj[4:-4, 0],
                    full_adv_traj[4:-4, 1],
                    full_adv_traj[4:-4, 2],
                    1.4,
                    2.9,
                    (5, 5, 5)
                )"""
                res[j] += 0
                min_dist[j] = 10000000
                continue
            #------------------------
            yaw_OV = get_polyline_yaw(trajs_OV[j])[::5].reshape(-1, 1)
            width_OV = adv_width
            length_OV = adv_length
            cos_theta = np.cos(yaw_OV)
            sin_theta = np.sin(yaw_OV)
            bbox_OV = np.concatenate([
                traj_OV,
                yaw_OV,
                traj_OV[:, 0].reshape(-1,
                                      1) + 0.5 * length_OV * cos_theta + 0.5 * width_OV * sin_theta,
                traj_OV[:, 1].reshape(-1,
                                      1) + 0.5 * length_OV * sin_theta - 0.5 * width_OV * cos_theta,
                traj_OV[:, 0].reshape(-1,
                                      1) + 0.5 * length_OV * cos_theta - 0.5 * width_OV * sin_theta,
                traj_OV[:, 1].reshape(-1,
                                      1) + 0.5 * length_OV * sin_theta + 0.5 * width_OV * cos_theta,
                traj_OV[:, 0].reshape(-1,
                                      1) - 0.5 * length_OV * cos_theta - 0.5 * width_OV * sin_theta,
                traj_OV[:, 1].reshape(-1,
                                      1) - 0.5 * length_OV * sin_theta + 0.5 * width_OV * cos_theta,
                traj_OV[:, 0].reshape(-1,
                                      1) - 0.5 * length_OV * cos_theta + 0.5 * width_OV * sin_theta,
                traj_OV[:, 1].reshape(-1,
                                      1) - 0.5 * length_OV * sin_theta - 0.5 * width_OV * cos_theta
            ], axis=1)

            probs_AV = [1.]
            for i, prob_AV in enumerate(probs_AV):
                P2 = prob_AV
                traj_AV = trajs_AV[i][::5]
                yaw_AV = get_polyline_yaw(trajs_AV[i])[::5].reshape(-1, 1)
                width_AV = width
                length_AV = length
                cos_theta = np.cos(yaw_AV)
                sin_theta = np.sin(yaw_AV)

                bbox_AV = np.concatenate((traj_AV, yaw_AV, \
                                          traj_AV[:, 0].reshape(-1,
                                                                1) + 0.5 * length_AV * cos_theta + 0.5 * width_AV * sin_theta, \
                                          traj_AV[:, 1].reshape(-1,
                                                                1) + 0.5 * length_AV * sin_theta - 0.5 * width_AV * cos_theta, \
                                          traj_AV[:, 0].reshape(-1,
                                                                1) + 0.5 * length_AV * cos_theta - 0.5 * width_AV * sin_theta, \
                                          traj_AV[:, 1].reshape(-1,
                                                                1) + 0.5 * length_AV * sin_theta + 0.5 * width_AV * cos_theta, \
                                          traj_AV[:, 0].reshape(-1,
                                                                1) - 0.5 * length_AV * cos_theta - 0.5 * width_AV * sin_theta, \
                                          traj_AV[:, 1].reshape(-1,
                                                                1) - 0.5 * length_AV * sin_theta + 0.5 * width_AV * cos_theta, \
                                          traj_AV[:, 0].reshape(-1,
                                                                1) - 0.5 * length_AV * cos_theta + 0.5 * width_AV * sin_theta, \
                                          traj_AV[:, 1].reshape(-1,
                                                                1) - 0.5 * length_AV * sin_theta - 0.5 * width_AV * cos_theta),
                                         axis=1)

                P3 = 0
                uncertainty = 1.
                alpha = 0.99
                '''
                B-A  F-E
                | |  | |
                C-D  G-H
                '''
                for (Cx1, Cy1, yaw1, xA, yA, xB, yB, xC, yC, xD, yD), (
                        Cx2, Cy2, yaw2, xE, yE, xF, yF, xG, yG, xH, yH) in zip(bbox_AV, bbox_OV):
                    uncertainty *= alpha
                    ego_adv_dist = np.linalg.norm([Cx1 - Cx2, Cy1 - Cy2])
                    if ego_adv_dist < min_dist[j]:
                        min_dist[j] = ego_adv_dist
                    if ego_adv_dist >= np.linalg.norm(
                            [0.5 * length_AV, 0.5 * width_AV]) + np.linalg.norm(
                        [0.5 * length_OV, 0.5 * width_OV]):
                        pass
                    elif Intersect([xA, yA, xB, yB], [xE, yE, xF, yF]) or Intersect(
                            [xA, yA, xB, yB],
                            [xF, yF, xG, yG]) or \
                            Intersect([xA, yA, xB, yB], [xG, yG, xH, yH]) or Intersect(
                        [xA, yA, xB, yB],
                        [xH, yH, xE, yE]) or \
                            Intersect([xB, yB, xC, yC], [xE, yE, xF, yF]) or Intersect(
                        [xB, yB, xC, yC],
                        [xF, yF, xG, yG]) or \
                            Intersect([xB, yB, xC, yC], [xG, yG, xH, yH]) or Intersect(
                        [xB, yB, xC, yC],
                        [xH, yH, xE, yE]) or \
                            Intersect([xC, yC, xD, yD], [xE, yE, xF, yF]) or Intersect(
                        [xC, yC, xD, yD],
                        [xF, yF, xG, yG]) or \
                            Intersect([xC, yC, xD, yD], [xG, yG, xH, yH]) or Intersect(
                        [xC, yC, xD, yD],
                        [xH, yH, xE, yE]) or \
                            Intersect([xD, yD, xA, yA], [xE, yE, xF, yF]) or Intersect(
                        [xD, yD, xA, yA],
                        [xF, yF, xG, yG]) or \
                            Intersect([xD, yD, xA, yA], [xG, yG, xH, yH]) or Intersect(
                        [xD, yD, xA, yA],
                        [xH, yH, xE, yE]):
                        P3 = uncertainty
                        break

                res[j] += P1 * P2 * P3 * P4
                print(f"P1: {P1}, P2: {P2}, P3: {P3}, P4: {P4}")
        print("propabilities:---------------------------------")
        print(res)
        if np.any(res):
            adv_traj_id = np.argmax(res)
        else:
            adv_traj_id = np.argmin(min_dist)

        adv_path = trajs_OV[adv_traj_id]
        adv_yaw = get_polyline_yaw(adv_path).reshape(-1, 1)
        adv_vel = np.linalg.norm(get_polyline_vel(adv_path), axis=1).reshape(-1, 1)

        ego_traj = get_full_trajectory(self.agents.index(self._ego_agent), features)[:, ::-1]
        adv_traj_original = get_full_trajectory(self.agents.index(self._adv_agent), features)[:, ::-1]
        full_adv_traj = get_full_trajectory(self.agents.index(self._adv_agent), features, future=adv_path)[:,
                        ::-1]
        ego_traj = np.concatenate([
            ego_traj,
            np.rad2deg(get_polyline_yaw(ego_traj)).reshape(-1, 1)
        ], axis=1)
        full_adv_traj = np.concatenate([
            full_adv_traj,
            np.rad2deg(get_polyline_yaw(full_adv_traj)).reshape(-1, 1)
        ], axis=1)
        trajectory = [(x, -y, -z) for x, y, z in full_adv_traj]
        plot_trajectory_vs_network(trajectory, self._network, None, 0, None, "final_chosen_traj")
        adv_traj_original = np.concatenate([
            adv_traj_original,
            np.rad2deg(get_polyline_yaw(adv_traj_original)).reshape(-1, 1)
        ], axis=1)

        ego_width, ego_length = ego.bounding_box.extent.y * 2, ego.bounding_box.extent.x * 2
        adv_width, adv_length = adversary.bounding_box.extent.y * 2, adversary.bounding_box.extent.x * 2
        visualize_traj(
            ego_traj[4:-4, 0],
            ego_traj[4:-4, 1],
            ego_traj[4:-4, 2],
            ego_width,
            ego_length,
            (0, 5, 0)
        )
        visualize_traj(
            adv_traj_original[4:-4, 0],
            adv_traj_original[4:-4, 1],
            adv_traj_original[4:-4, 2],
            adv_width,
            adv_length,
            (0, 0, 5)
        )
        visualize_traj(
            full_adv_traj[4:-4, 0],
            full_adv_traj[4:-4, 1],
            full_adv_traj[4:-4, 2],
            adv_width,
            adv_length,
            (5, 0, 0)
        )

        world = CarlaDataProvider.get_world()
        spectator = world.get_spectator()
        ego_loc = ego.get_location()
        spectator.set_transform(carla.Transform(
            carla.Location(ego_loc.x, ego_loc.y, ego_loc.z + 80),
            carla.Rotation(pitch=-90)
        ))
        settings = world.get_settings()
        settings.synchronous_mode = False
        adv_traj = np.concatenate([adv_path[:, ::-1], adv_vel, adv_yaw], axis=1)
        return adv_traj_id, adv_traj

    def filter_road_graph_for_odd_checking(self, roadgraph_data):
        xyz = roadgraph_data["roadgraph_samples/xyz"]
        direction = roadgraph_data["roadgraph_samples/dir"]
        rtype = roadgraph_data["roadgraph_samples/type"].squeeze()
        valid = roadgraph_data["roadgraph_samples/valid"]
        rid = roadgraph_data["roadgraph_samples/id"]

        # Masks for filtering by type
        mask_surface = (rtype == 2)
        mask_edges = np.isin(rtype, [15, 16])
        mask_markings = np.isin(rtype, [7, 8, 9, 10, 11, 12, 13])

        # Create the three dictionaries
        road_surface = {
            "xyz": xyz[mask_surface],
            "dir": direction[mask_surface],
            "type": rtype[mask_surface],
            "valid": valid[mask_surface],
            "id": rid[mask_surface]
        }

        road_edges = {
            "xyz": xyz[mask_edges],
            "dir": direction[mask_edges],
            "type": rtype[mask_edges],
            "valid": valid[mask_edges],
            "id": rid[mask_edges]
        }

        road_markings = {
            "xyz": xyz[mask_markings],
            "dir": direction[mask_markings],
            "type": rtype[mask_markings],
            "valid": valid[mask_markings],
            "id": rid[mask_markings]
        }

        self.road_surface = road_surface
        self.road_edges = road_edges
        self.road_markings = road_markings

    def _get_features(self):
        roadgraph_features = self._get_roadgraph_features(self._max_samples)
        self.filter_road_graph_for_odd_checking(roadgraph_features)
        state_features, ego_route = self._get_state_features("ego_vehicle")
        dynamic_map_features = self._get_dynamic_map_features()

        ids = roadgraph_features["roadgraph_samples/id"]
        for new_id, old_id in enumerate(sorted(np.unique(ids))):
            roadgraph_features["roadgraph_samples/id"][ids == old_id] = new_id
            dynamic_map_features["traffic_light_state/past/id"][
                dynamic_map_features["traffic_light_state/past/id"] == old_id] = new_id
            dynamic_map_features["traffic_light_state/current/id"][
                dynamic_map_features["traffic_light_state/current/id"] == old_id] = new_id

        features = {}
        features.update(roadgraph_features)
        features.update(state_features)
        features.update(dynamic_map_features)
        features["scenario/id"] = np.array(["template"])
        features["state/objects_of_interest"] = features['state/tracks_to_predict'].copy()
        return features, ego_route
    
    def check_on_roadgraph_old(self, trajectory, idx):
        # Define solid line types
        solid_line_types = {7, 8, 9, 10, 11, 12}

        road_surface = self.road_surface
        road_edges = self.road_edges
        road_markings = self.road_markings

        surface_xyz = road_surface["xyz"][:, :2]
        surface_dir = road_surface["dir"]
        edges_xyz = road_edges["xyz"][:, :2]
        markings_xyz = road_markings["xyz"][:, :2]

        # Build KD-Trees for fast nearest-neighbor searches
        tree_surface = cKDTree(surface_xyz) if surface_xyz is not None else None
        tree_edges = cKDTree(edges_xyz) if edges_xyz is not None else None
        tree_markings = cKDTree(markings_xyz) if markings_xyz is not None else None

        for x, y, yaw in trajectory:
            # --- 1. Check closest road surface (yaw alignment) ---
            if tree_surface:
                #TODO change here to not only take the closest point but check at least 5 or so to care for intersections!!
                surface_dist, idx_surface = tree_surface.query([x, y], k=14)
                yaw_diff = []
                for dist, idx in zip(surface_dist, idx_surface):
                    closest_surface_yaw = np.degrees(np.arctan2(road_surface["dir"][idx][1], road_surface["dir"][idx][0])) 
                    yaw_diff.append(abs(yaw - closest_surface_yaw))

                if all(x > 45 for x in yaw_diff):  # Allow max 45-degree deviation
                    print(f"Yaw misalignment at ({x}, {y})")
                    #plot_stuff(surface_xyz, surface_dir, edges_xyz, markings_xyz, trajectory, idx, "yaw")
                    #self.plot_stuff_traj(trajectory, idx1)
                    return False  

            # --- 2. Check distance to closest road edge ---
            if tree_edges:
                edge_dist, _ = tree_edges.query([x, y], k=1)

                if edge_dist < 0.5:  # Too close to the edge
                    print(f"Too close to road edge at ({x}, {y}), dist={edge_dist:.2f}")
                    #plot_stuff(surface_xyz, surface_dir, edges_xyz, markings_xyz, trajectory, idx, "edge")
                    #self.plot_stuff_traj(trajectory, idx1)
                    return False  
                if surface_dist[0] > edge_dist + .2:
                    print(f"Closer to road edge than to road surface -> out of lane.")
                    #plot_stuff(surface_xyz, surface_dir, edges_xyz, markings_xyz, trajectory, idx, "edge")
                    #self.plot_stuff_traj(trajectory, idx1)
                    return False

            # --- 3. Check if crossing solid road markings ---
            if tree_markings:
                marking_dist, idx_marking = tree_markings.query([x, y], k=1)
                marking_type = road_markings["type"][idx_marking]

                if marking_type in solid_line_types and marking_dist < 0.5:
                    print(f"Crossed solid line at ({x}, {y}), dist={marking_dist:.2f}")
                    #plot_stuff(surface_xyz, surface_dir, edges_xyz, markings_xyz, trajectory, idx, "line")
                    #self.plot_stuff_traj(trajectory, idx1)
                    return False  
        #plot_stuff(surface_xyz, surface_dir, edges_xyz, markings_xyz, trajectory, idx, "passed")
        #self.plot_stuff_traj(trajectory, idx1)
        return True  # If no violations occur
    
    def correct_yaw_by_distance_step3(self, points_array, min_distance=0.5):
        """
        Adjusts yaw of points based on distance from the point 3 steps before.
        If the distance is less than `min_distance`, yaw is copied from the last valid point.

        Parameters:
            points (list of (x, y, yaw)): Input list of points.
            min_distance (float): Minimum required distance between points to keep original yaw.

        Returns:
            list of (x, y, yaw): Updated list with corrected yaw values.
        """
        if points_array.shape[0] < 4:
            return points_array.copy()

        corrected = points_array.copy()
        reference_point = corrected[0]

        for j in range(3, len(corrected), 3):
            dist = np.hypot(corrected[j][0] - reference_point[0],
                            corrected[j][1] - reference_point[1])
            if dist >= min_distance:
                new_yaw = corrected[j][2]
                corrected[:j, 2] = new_yaw  # update yaw for all earlier points
                break  # done once first valid point is found

        return corrected
    

    def is_trajectory_within_speed_limit(self, trajectory, speed_limit_mps, timestep=0.05):
        """
        Check if the trajectory ever exceeds the speed limit.

        Args:
            trajectory (np.ndarray): An array of shape (N, 3), where each row is [x, y, yaw].
            speed_limit_mps (float): Speed limit in meters per second.
            timestep (float): Time between each trajectory point in seconds (default: 0.05s).

        Returns:
            bool: True if the trajectory is within the speed limit at all times, False otherwise.
        """
        # Calculate distances between consecutive points
        trajectory = trajectory[2:-2]
        trajectory = np.asarray(trajectory)
        deltas = np.diff(trajectory[:, :2], axis=0)  # Only use x, y
        distances = np.linalg.norm(deltas, axis=1)   # Euclidean distance between consecutive points

        # Compute speed at each step
        speeds = distances / timestep

        # Check if any speed exceeds the speed limit
        return np.all(speeds <= speed_limit_mps)
    
    
    def check_on_roadgraph(self, trajectory_original, idx, traj_ov=None, debug=False):
        """
        Checks if the given trajectory is valid based on the Scenic Network object and plots the results.

        Args:
            trajectory (list of tuples): List of (x, y, yaw) points.
            network (scenic.core.network.Network): Scenic network object.

        Returns:
            bool: True if trajectory is valid, False otherwise.
        """
        network = self._network
        trajectory_yaw_corrected = self.correct_yaw_by_distance_step3(trajectory_original)
        trajectory = [(x, -y, -z) for x, y, z in trajectory_yaw_corrected]
        if traj_ov is not None:
            adv_vel = np.linalg.norm(get_polyline_vel(traj_ov), axis=1).reshape(-1, 1)
            if max(adv_vel) > 55:
                print(f"Speed limit exceeded")
                if True:    
                    print(f"Trajectory INVALID:  violations found.")
                    plot_trajectory_vs_network(trajectory, network, None, idx, None, "speed_limit/invalid_trajectory_debug")
                return False
        
        """if not self.is_trajectory_within_speed_limit(trajectory, 17):
            print(f"Speed limit exceeded")
            if True:    
                print(f"Trajectory INVALID:  violations found.")
                plot_trajectory_vs_network(trajectory, network, None, idx, None, "speed_limit/invalid_trajectory_debug")
            return False"""
        
        invalid_points = []
        invalid_reasons = []

        for x1, y1, yaw in trajectory:
            center_point = shapely.geometry.Point(x1, y1)

            # 1. Check if point is within the overall drivable area
            if not network.drivableRegion.containsPoint(center_point):
                print(f"Point ({x1}, {y1}) is OUTSIDE the drivable area.")
                invalid_points.append((x1, y1))
                invalid_reasons.append("Drivable Area")
                if not debug:
                    break
                continue 

            corners = compute_bounding_box_corners(x1, y1, .8, 2.2, yaw)
            for x, y in corners:
                point = shapely.geometry.Point(x, y)

                candiate_lanes = []
                yaw_differences = []

                # 2. Check which lanes contain this point
                for lane in network.lanes:
                    if lane.containsPoint(point):
                        candiate_lanes.append(lane)

                if len(candiate_lanes) < 1:
                    continue

                for lane in candiate_lanes:
                    # Get nearest point on lane centerline
                    nearest_pt = lane.centerline.lineString.interpolate(
                        lane.centerline.lineString.project(point)
                    )

                    # Compute expected lane direction
                    lane_yaw_rad = lane.orientation.value(Vector(nearest_pt.x, nearest_pt.y)) + 1.57

                    lane_yaw = np.degrees(lane_yaw_rad) 

                    # Compute yaw difference and store
                    yaw_diff = abs(self.wrap_to_180(yaw - lane_yaw))
                    yaw_differences.append(yaw_diff)

                if all(diff > 80 for diff in yaw_differences):
                    print(f"Yaw misalignment at ({x}, {y})")
                    invalid_points.append((x, y))
                    invalid_reasons.append("Yaw Misalignment from one of the corner points of bounding box")
                    if not debug:
                        break
                    continue  # Move to next point


        if invalid_points:
            if True:    
                print(f"Trajectory INVALID: {len(invalid_points)} violations found.")
                plot_trajectory_vs_network(trajectory, network, invalid_points, idx, invalid_reasons, "invalid/invalid_trajectory_debug")
            return False  
        else:
            if True:   
                print("Trajectory is VALID!")
                plot_trajectory_vs_network(trajectory, network, invalid_points, idx, invalid_reasons, "valid/valid_trajectory_debug")
            return True 
        
    def score_lane_adherence(self, trajectory_original, weight=1):
        """
        Checks if the given trajectory is valid based on the Scenic Network object and plots the results.

        Args:
            trajectory (list of tuples): List of (x, y, yaw) points.
            network (scenic.core.network.Network): Scenic network object.

        Returns:
            bool: True if trajectory is valid, False otherwise.
        """
        errors = []
        trajectory = [(x, -y, -z) for x, y, z in trajectory_original]
        network = self._network

        for x, y, yaw in trajectory:
            current_point = shapely.geometry.Point(x, y)

            candidate_lanes = []
            yaw_differences = []

            # 2. Check which lanes contain this point
            for lane in network.lanes:
                if lane.containsPoint(current_point):
                    candidate_lanes.append(lane)
                    nearest_pt = lane.centerline.lineString.interpolate(
                        lane.centerline.lineString.project(current_point)
                    )

                    # Compute expected lane direction
                    lane_yaw_rad = lane.orientation.value(Vector(nearest_pt.x, nearest_pt.y)) + 1.57

                    lane_yaw = np.degrees(lane_yaw_rad) 

                    # Compute yaw difference and store
                    yaw_diff = abs(self.wrap_to_180(yaw - lane_yaw))
                    yaw_differences.append(yaw_diff)
            
            # Select the lane with the closest heading to the vehicle's heading
            best_lane_idx = min(range(len(yaw_differences)), key=lambda i: abs(yaw_differences[i]))
            best_lane = candidate_lanes[best_lane_idx]
            
            # Get the nearest point on the best lane's centerline
            nearest_pt = best_lane.centerline.lineString.interpolate(
                    best_lane.centerline.lineString.project(current_point)
                )
            lateral_error = current_point.distance(nearest_pt)
            
            # Combine errors in a weighted sum
            error = weight * lateral_error
            errors.append(error)
        
        # If no points produced an error, return a default high value or 0.
        if not errors:
            return float('inf')
        
        score = np.mean(errors)
        return score

    # Helper function to normalize yaw differences
    def wrap_to_180(self, angle):
        """Normalize an angle to the range [-180, 180] degrees."""
        return (angle + 180) % 360 - 180

    def _get_state_features(self, ego_agent: str) -> tuple[dict, np.ndarray]:
        state_features = {
            "state/id": np.full([128, ], -1, dtype=np.int64),
            "state/type": np.full([128, ], 0, dtype=np.int64),
            "state/is_sdc": np.full([128, ], 0, dtype=np.int64),
            'state/tracks_to_predict': np.full([128, ], 0, dtype=np.int64),
            'state/current/bbox_yaw': np.full([128, 1], -1, dtype=np.float32),
            'state/current/height': np.full([128, 1], -1, dtype=np.float32),
            'state/current/length': np.full([128, 1], -1, dtype=np.float32),
            'state/current/valid': np.full([128, 1], 0, dtype=np.int64),
            'state/current/vel_yaw': np.full([128, 1], -1, dtype=np.float32),
            'state/current/velocity_x': np.full([128, 1], -1, dtype=np.float32),
            'state/current/velocity_y': np.full([128, 1], -1, dtype=np.float32),
            'state/current/width': np.full([128, 1], -1, dtype=np.float32),
            'state/current/x': np.full([128, 1], -1, dtype=np.float32),
            'state/current/y': np.full([128, 1], -1, dtype=np.float32),
            'state/current/z': np.full([128, 1], -1, dtype=np.float32),
            'state/past/bbox_yaw': np.full([128, 10], -1, dtype=np.float32),
            'state/past/height': np.full([128, 10], -1, dtype=np.float32),
            'state/past/length': np.full([128, 10], -1, dtype=np.float32),
            'state/past/valid': np.full([128, 10], 0, dtype=np.int64),
            'state/past/vel_yaw': np.full([128, 10], -1, dtype=np.float32),
            'state/past/velocity_x': np.full([128, 10], -1, dtype=np.float32),
            'state/past/velocity_y': np.full([128, 10], -1, dtype=np.float32),
            'state/past/width': np.full([128, 10], -1, dtype=np.float32),
            'state/past/x': np.full([128, 10], -1, dtype=np.float32),
            'state/past/y': np.full([128, 10], -1, dtype=np.float32),
            'state/past/z': np.full([128, 10], -1, dtype=np.float32),
            'state/future/bbox_yaw': np.full([128, 160], -1, dtype=np.float32),
            'state/future/height': np.full([128, 160], -1, dtype=np.float32),
            'state/future/length': np.full([128, 160], -1, dtype=np.float32),
            'state/future/valid': np.full([128, 160], 0, dtype=np.int64),
            'state/future/vel_yaw': np.full([128, 160], -1, dtype=np.float32),
            'state/future/velocity_x': np.full([128, 160], -1, dtype=np.float32),
            'state/future/velocity_y': np.full([128, 160], -1, dtype=np.float32),
            'state/future/width': np.full([128, 160], -1, dtype=np.float32),
            'state/future/x': np.full([128, 160], -1, dtype=np.float32),
            'state/future/y': np.full([128, 160], -1, dtype=np.float32),
            'state/future/z': np.full([128, 160], -1, dtype=np.float32)
        }

        for i, actor_id in enumerate(self.agents):
            actor = self.actors[actor_id]
            state_features["state/id"][i] = actor.id
            state_features["state/type"][i] = AgentTypes.from_carla_type(actor).value
            state_features["state/is_sdc"][i] = 1 if actor_id == self._ego_agent else 0
            state_features["state/tracks_to_predict"][i] = i < len(self._trajectories)

            for t in range(min(len(self._trajectories[actor_id]), 160)):
                offset = 0
                if t < 10:
                    time = "past"
                elif t == 10:
                    offset = 10
                    time = "current"
                else:
                    offset = 11
                    time = "future"

                state = self._trajectories[actor_id][t]
                for key in state:
                    state_features[f"state/{time}/{key}"][i, t - offset] = state[key]

        is_idc = state_features["state/is_sdc"] == 1
        ego_route = np.concatenate([
            state_features["state/future/x"][is_idc],
            state_features["state/future/y"][is_idc],
        ])
        return state_features, ego_route

    def _get_dynamic_map_features(self) -> dict:
        traffic_light_features = {
            'traffic_light_state/current/state': np.full([1, 16], -1, dtype=np.int64),
            'traffic_light_state/current/valid': np.full([1, 16], 0, dtype=np.int64),
            'traffic_light_state/current/id': np.full([1, 16], -1, dtype=np.int64),
            'traffic_light_state/current/x': np.full([1, 16], -1, dtype=np.float32),
            'traffic_light_state/current/y': np.full([1, 16], -1, dtype=np.float32),
            'traffic_light_state/current/z': np.full([1, 16], -1, dtype=np.float32),
            'traffic_light_state/past/state': np.full([10, 16], -1, dtype=np.int64),
            'traffic_light_state/past/valid': np.full([10, 16], 0, dtype=np.int64),
            'traffic_light_state/past/x': np.full([10, 16], -1, dtype=np.float32),
            'traffic_light_state/past/y': np.full([10, 16], -1, dtype=np.float32),
            'traffic_light_state/past/z': np.full([10, 16], -1, dtype=np.float32),
            'traffic_light_state/past/id': np.full([10, 16], -1, dtype=np.int64),
        }

        ego = self.actors[self._ego_agent]

        def min_distance_to_ego(tl_id):
            loc = ego.get_location()
            return loc.distance(self._traffic_light_locations[tl_id])

        traffic_lights = sorted(self._traffic_light_states, key=min_distance_to_ego)[:16]

        lane_ids = []

        for tl in traffic_lights:
            state = self._traffic_light_states[tl][0]
            for lane_state in state:
                lane_ids.append(lane_state["id"])
                
        print(lane_ids)

        lane_states = [[] for _ in range(len(lane_ids))]
        for tl in traffic_lights:
            for state in self._traffic_light_states[tl]:
                for lane_state in state:
                    idx = lane_ids.index(lane_state["id"])
                    lane_states[idx].append(lane_state)

        lane_states = lane_states[:16]

        for i, lane_state_t in enumerate(lane_states):
            for t in range(10):
                state = lane_state_t[t]
                for key in state:
                    traffic_light_features[f"traffic_light_state/past/{key}"][t, i] = state[key]
            state = lane_state_t[10]
            for key in state:
                traffic_light_features[f"traffic_light_state/current/{key}"][0, i] = state[key]
        return traffic_light_features

    def _get_roadgraph(self, agent: str) -> dict:
        dir, id, type, valid, xyz = [], [], [], [], []
        num_features = 0
        lanes = self._get_centerlines()
        self._lane_ids = list(sorted(lanes.keys()))
        elements = [
            *[lanes[id] for id in self._lane_ids],
            *self._get_road_edges(),
            *self._get_road_markings(),
            *self._get_crosswalks()
        ]

        for line in elements:
            assert line["xyz"].shape[0] == line["dir"].shape[0]
            xyz.append(line["xyz"])
            type.append(line["type"].reshape(-1, 1))
            valid.append(line["valid"].reshape(-1, 1))
            dir.append(line["dir"])
            length = line["type"].shape[0]
            id.append(np.full([length, 1], num_features, dtype=np.int64))
            num_features += 1

        return {
            "dir": np.concatenate(dir, axis=0),
            "id": np.concatenate(id, axis=0),
            "type": np.concatenate(type, axis=0),
            "valid": np.concatenate(valid, axis=0),
            "xyz": np.concatenate(xyz, axis=0)
        }

    def _get_crosswalks(self) -> list[dict[str, np.ndarray]]:
        crosswalks = self._map.get_crosswalks()
        crosswalk_encodings = []
        idx = 0
        while idx < len(crosswalks) - 4:
            start = idx
            idx += 1
            while crosswalks[idx] != crosswalks[start]:
                idx += 1
            end = idx
            idx += 1
            crosswalk = crosswalks[
                        start:end + 1]  # crosswalks are 5 points, first point is repeated at the end
            xyz = np.array([
                [loc.x, loc.y, loc.z + 0.1]
                for loc in crosswalk
            ], dtype=np.float32)
            dir = (xyz[1:] - xyz[:-1]) / np.linalg.norm(xyz[1:] - xyz[:-1], axis=1, keepdims=True)
            dir = np.concatenate([dir, dir[-1:]], axis=0)
            type = np.full((xyz.shape[0],), RoadGraphTypes.CROSSWALK.value, dtype=np.int64)
            valid = np.ones_like(type, dtype=np.int64)
            crosswalk_encodings.append({
                "dir": dir,
                "type": type,
                "valid": valid,
                "xyz": xyz  # last point is repeated
            })

        return crosswalk_encodings

    def _get_agent_state(self, agent: carla.Actor) -> dict:
        bbox: carla.BoundingBox = agent.bounding_box.extent
        length, width, height = bbox.x * 2, bbox.y * 2, bbox.z * 2
        loc = agent.get_location()
        vel = agent.get_velocity()
        rotation = agent.get_transform().rotation
        state = {
            "length": length,
            "width": width,
            "height": height,
            "x": loc.x,
            "y": loc.y,
            "z": loc.z,
            "velocity_x": vel.x,
            "velocity_y": vel.y,
            "bbox_yaw": rotation.yaw,
            "vel_yaw": np.arctan2(vel.y, vel.x),
            "valid": 1
        }
        return state

    def _get_traffic_light_states(self, traffic_light: carla.TrafficLight) -> list[
        dict[str, np.ndarray]]:
        tl_state = traffic_light.get_state()
        if tl_state == carla.TrafficLightState.Red:
            tl_state = TrafficLightStates.LANE_STATE_STOP
        elif tl_state == carla.TrafficLightState.Yellow:
            tl_state = TrafficLightStates.LANE_STATE_CAUTION
        elif tl_state == carla.TrafficLightState.Green:
            tl_state = TrafficLightStates.LANE_STATE_GO
        else:
            tl_state = TrafficLightStates.LANE_STATE_UNKNOWN

        states = []
        stop_points: list[carla.Waypoint] = traffic_light.get_stop_waypoints()
        for stop_wp in stop_points:
            stop_loc = stop_wp.transform.location
            lane_id = self._lane_ids.index((stop_wp.road_id, stop_wp.lane_id))
            states.append({
                "state": tl_state.value,
                "valid": 1,
                "id": lane_id,
                "x": stop_loc.x,
                "y": stop_loc.y,
                "z": stop_loc.z
            })
        return states

    def _get_centerlines(self) -> dict[tuple, dict[str, np.ndarray]]:
        centerlines = {}
        for start, _ in self._topology:
            types, xyz = [], []
            prev = None
            dist = self._resolution
            previous_wps = start.previous(dist)
            while len(previous_wps) > 0 and dist > 0:
                previous_wps = start.previous(dist)
                prev = min(previous_wps,
                           key=lambda wp: wp.transform.location.distance(start.transform.location))
                dist -= 0.5

            wps = start.next_until_lane_end(self._resolution)
            if prev is not None:
                wps = [prev, *wps]
            for wp in wps:
                if wp.lane_type == carla.LaneType.Driving:
                    type = RoadGraphTypes.LANE_SURFACE_STREET
                elif wp.lane_type == carla.LaneType.Biking:
                    type = RoadGraphTypes.LANE_BIKE_LANE
                else:
                    type = RoadGraphTypes.UNKNOWN
                location = wp.transform.location
                xyz.append([location.x, location.y, location.z])
                types.append(type.value)
            xyz = np.array(xyz, dtype=np.float32)
            types = np.array(types, dtype=np.int64)
            valid = np.ones_like(types, dtype=np.int64)
            dir = (xyz[1:] - xyz[:-1]) / (
                    np.linalg.norm(xyz[1:] - xyz[:-1], axis=1, keepdims=True) + 1e-6)
            carla.Rotation()
            dir = np.concatenate([dir, dir[-1:]], axis=0)
            id = (start.road_id, start.lane_id)
            centerlines[id] = {
                "dir": dir,
                "type": types,
                "valid": valid,
                "xyz": xyz
            }
        return centerlines

    def _get_road_markings(self) -> list[dict[str, np.ndarray]]:
        def get_type(type, color) -> RoadGraphTypes:
            rg_type = RoadGraphTypes.UNKNOWN
            if type == carla.LaneMarkingType.Broken:
                if color == carla.LaneMarkingColor.White:
                    rg_type = RoadGraphTypes.ROAD_LINE_BROKEN_SINGLE_WHITE
                elif color == carla.LaneMarkingColor.Yellow:
                    rg_type = RoadGraphTypes.ROAD_LINE_BROKEN_SINGLE_YELLOW
            elif type == carla.LaneMarkingType.Solid:
                if color == carla.LaneMarkingColor.White:
                    rg_type = RoadGraphTypes.ROAD_LINE_SOLID_SINGLE_WHITE
                elif color == carla.LaneMarkingColor.Yellow:
                    rg_type = RoadGraphTypes.ROAD_LINE_SOLID_SINGLE_YELLOW
            elif type == carla.LaneMarkingType.BrokenBroken:
                if color == carla.LaneMarkingColor.Yellow:
                    rg_type = RoadGraphTypes.ROAD_LINE_BROKEN_DOUBLE_YELLOW
            elif type == carla.LaneMarkingType.SolidSolid:
                if color == carla.LaneMarkingColor.White:
                    rg_type = RoadGraphTypes.ROAD_LINE_SOLID_DOUBLE_WHITE
                elif color == carla.LaneMarkingColor.Yellow:
                    rg_type = RoadGraphTypes.ROAD_LINE_SOLID_DOUBLE_YELLOW
            return rg_type

        def extract_line(line: list[carla.Waypoint]):
            return np.stack([
                [wp.transform.location.x, wp.transform.location.y, wp.transform.location.z]
                for wp in line
            ])

        def parallel_line(line: list[carla.Waypoint], left: bool):
            widths = np.array([wp.lane_width for wp in line])
            line = extract_line(line)
            dir = (line[1:] - line[:-1]) / np.linalg.norm(line[1:] - line[:-1], axis=1,
                                                          keepdims=True)
            dir = np.concatenate([dir, dir[-1:]], axis=0)
            angle = np.arctan2(dir[:, 1], dir[:, 0])
            offset = widths * (1 if left else -1) * 0.5
            shift = np.stack([
                offset * np.sin(angle),
                offset * -np.cos(angle),
                np.zeros_like(angle)
            ], axis=1)
            return line + shift, dir

        markings = []
        world = CarlaDataProvider.get_world()
        lanes = []
        # for road in self._network.roads:
        #     lanes.extend(road.lanes)
        # for intersection in self._network.intersections:
        #     for maneuver in filter(lambda m: m.type == ManeuverType.STRAIGHT, intersection.maneuvers):
        #         if maneuver.connectingLane is not None:
        #             lanes.append(maneuver.connectingLane)
        # lanes = [
        #    [self._map.get_waypoint(scenicToCarlaLocation(p, world=world)) for p in lane]
        #    for lane in lanes
        # ]

        for start, _ in self._topology:
            lanes.append([
                wp for wp in start.next_until_lane_end(self._resolution)
                if not wp.is_intersection
            ])

        for lane in lanes:
            centerline = lane
            if len(centerline) < 2:
                continue

            left = [
                wp for wp in centerline
                if wp.left_lane_marking.type != carla.LaneMarkingType.NONE
            ]

            right = [
                wp for wp in centerline
                if wp.right_lane_marking.type != carla.LaneMarkingType.NONE
            ]

            if len(left) > 1:
                xyz, dir = parallel_line(left, True)
                types = np.array([
                    get_type(wp.left_lane_marking.type, wp.left_lane_marking.color).value
                    for wp in left
                ], dtype=np.int64)
                markings.append({
                    "dir": dir,
                    "type": types,
                    "valid": np.ones_like(types, dtype=np.int64),
                    "xyz": xyz
                })

            if len(right) > 1:
                xyz, dir = parallel_line(right, False)
                types = np.array([
                    get_type(wp.right_lane_marking.type, wp.right_lane_marking.color).value
                    for wp in right
                ], dtype=np.int64)
                markings.append({
                    "dir": dir,
                    "type": types,
                    "valid": np.ones_like(types, dtype=np.int64),
                    "xyz": xyz
                })
        return markings

    def _get_road_edges(self) -> list[dict[str, np.ndarray]]:
        world = CarlaDataProvider.get_world()
        lines = []
        for road in self._network.roads:
            edges = [
                (road.leftEdge, RoadGraphTypes.ROAD_EDGE_BOUNDARY),
                (road.rightEdge, RoadGraphTypes.ROAD_EDGE_BOUNDARY),
                (road.centerline, RoadGraphTypes.ROAD_EDGE_MEDIAN)
            ]
            for edge, type in edges:
                line = []
                for point in edge.pointsSeparatedBy(self._resolution):
                    loc = scenicToCarlaLocation(point, world=world)
                    if len(line) > 0 and abs(line[-1][2] - loc.z) > 0.5:
                        loc.z = line[-1][2]
                    line.append([loc.x, loc.y, loc.z])
                if len(line) < 2:
                    continue
                xyz = np.array(line, dtype=np.float32)
                type = np.full([xyz.shape[0], 1], type.value, dtype=np.int64)
                valid = np.ones_like(type, dtype=np.int64)
                dir = (xyz[1:] - xyz[:-1]) / np.linalg.norm(xyz[1:] - xyz[:-1], axis=1,
                                                            keepdims=True)
                dir = np.concatenate([dir, dir[-1:]], axis=0)
                lines.append({
                    "dir": dir,
                    "type": type,
                    "valid": valid,
                    "xyz": xyz
                })
        return lines

    def _visualize(self, xyz, type):
        world = CarlaDataProvider.get_world()
        for i, line in enumerate(xyz):
            for j, (s, e) in enumerate(zip(line[:-1].tolist(), line[1:].tolist())):
                s = carla.Location(x=s[0], y=s[1], z=s[2])
                e = carla.Location(x=e[0], y=e[1], z=e[2])
                color = {
                    RoadGraphTypes.LANE_SURFACE_STREET.value: carla.Color(5, 0, 0),
                    RoadGraphTypes.LANE_BIKE_LANE.value: carla.Color(0, 5, 0),
                    RoadGraphTypes.ROAD_EDGE_BOUNDARY.value: carla.Color(0, 0, 5),
                    RoadGraphTypes.ROAD_EDGE_MEDIAN.value: carla.Color(5, 5, 0),
                    RoadGraphTypes.ROAD_LINE_BROKEN_SINGLE_WHITE.value: carla.Color(5, 5, 5),
                    RoadGraphTypes.ROAD_LINE_SOLID_SINGLE_WHITE.value: carla.Color(5, 5, 5),
                    RoadGraphTypes.ROAD_LINE_SOLID_DOUBLE_WHITE.value: carla.Color(5, 5, 5),
                    RoadGraphTypes.ROAD_LINE_BROKEN_SINGLE_YELLOW.value: carla.Color(5, 5, 0),
                    RoadGraphTypes.ROAD_LINE_BROKEN_DOUBLE_YELLOW.value: carla.Color(5, 5, 0),
                    RoadGraphTypes.ROAD_LINE_SOLID_SINGLE_YELLOW.value: carla.Color(5, 5, 0),
                    RoadGraphTypes.ROAD_LINE_SOLID_DOUBLE_YELLOW.value: carla.Color(5, 5, 0),
                    RoadGraphTypes.UNKNOWN.value: carla.Color(255, 0, 0),
                    RoadGraphTypes.CROSSWALK.value: carla.Color(0, 5, 5)

                }[type[i][j].item()]
                world.debug.draw_line(s, e, thickness=0.1, color=color, life_time=0)

        world.tick()
        settings: carla.WorldSettings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)

    def calculate_kpis(self, ego, adv):
        """try:
            ego.bounding_box_extent.x
        except:
            return"""
        ttc, drac, mttc = calculate_ttc(ego, adv)
        near_miss_ttc, _, _ = calculate_near_miss_ttc(ego, adv)
        risk_coefficient = calculate_risk_coefficient(ego, adv)
        self.kpis["ttc"].append(ttc[0])
        self.kpis["drac"].append(drac[0])
        self.kpis["mttc"].append(mttc[0])
        self.kpis["risk_coefficient"].append(risk_coefficient)
        acc = adv.get_acceleration()
        self.kpis["adv_acc"].append(math.sqrt(acc.x**2 + acc.y**2 + acc.z**2))
        self.kpis["near_miss_ttc"].append(near_miss_ttc[0])
        self.kpis["euclidean_distance"].append(calc_euclidian_distance(ego, adv))
        #self.kpis["enhanced_ttc"].append(self.calculate_enhanced_ttc(ego, adv))
        #TODO calc other KPIs also here
        #self.kpis["adv_yaw"].append(adv.get_transform().rotation.yaw)
        

    
    def get_aabb(self, vehicle):
        # Get pose info from CARLA vehicle
        transform = vehicle.get_transform()
        location = transform.location
        yaw = np.deg2rad(transform.rotation.yaw)  # Convert degrees to radians
        
        # Assume bounding box dimensions are available from vehicle
        bbox = vehicle.bounding_box
        width = bbox.extent.y * 2  # extent.y is half-width
        length = bbox.extent.x * 2  # extent.x is half-length
        
        # Get rotated AABB in world frame
        corners = np.array([
            [-width / 2, -length / 2],
            [ width / 2, -length / 2],
            [ width / 2,  length / 2],
            [-width / 2,  length / 2]
        ])
        
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s], [s, c]])
        rotated = (R @ corners.T).T
        rotated += np.array([location.x, location.y])
        
        min_x, min_y = rotated.min(axis=0)
        max_x, max_y = rotated.max(axis=0)
        return min_x, max_x, min_y, max_y
    
    def get_velocity_xy(self, vehicle):
        velocity = vehicle.get_velocity()
        return np.array([velocity.x, velocity.y])
    

    def time_to_collision_aabb(self, vehicle1, vehicle2):
        min_x1, max_x1, min_y1, max_y1 = self.get_aabb(vehicle1)
        min_x2, max_x2, min_y2, max_y2 = self.get_aabb(vehicle2)
        
        # Relative velocity
        v1_vel = self.get_velocity_xy(vehicle1)
        v2_vel = self.get_velocity_xy(vehicle2)
        rel_vel = v2_vel - v1_vel
        
        # Relative position of box2 to box1 AABB
        center1 = np.array([(min_x1 + max_x1) / 2, (min_y1 + max_y1) / 2])
        center2 = np.array([(min_x2 + max_x2) / 2, (min_y2 + max_y2) / 2])
        rel_pos = center2 - center1
        
        # Combined half extents
        half_w = ((max_x1 - min_x1) + (max_x2 - min_x2)) / 2
        half_l = ((max_y1 - min_y1) + (max_y2 - min_y2)) / 2

        def axis_ttc(r_pos, r_vel, half_size):
            if r_vel == 0:
                if abs(r_pos) > half_size:
                    return np.inf, -np.inf
                else:
                    return -np.inf, np.inf
            t1 = (-(half_size) - r_pos) / r_vel
            t2 = (half_size - r_pos) / r_vel
            return min(t1, t2), max(t1, t2)

        t_entry_x, t_exit_x = axis_ttc(rel_pos[0], rel_vel[0], half_w)
        t_entry_y, t_exit_y = axis_ttc(rel_pos[1], rel_vel[1], half_l)
        
        t_entry = max(t_entry_x, t_entry_y)
        t_exit = min(t_exit_x, t_exit_y)

        if t_entry < t_exit and t_exit > 0:
            return max(t_entry, 0)
        else:
            return GLOBALS.DEFAULT_FOR_TTC


    def calculate_ttc(self, ego_vehicle, target_vehicle):
        """
        Calculate the Time to Collision (TTC) considering the heading and speed of the ego vehicle
        and a target vehicle.

        Args:
            ego_vehicle (carla.Actor): The ego vehicle actor.
            target_vehicle (carla.Actor): The target vehicle actor.

        Returns:
            float: The TTC value in timesteps, or None if no collision is predicted.
        """
        # get bounding boxes:
        width, length = ego_vehicle.bounding_box_extent.y * 2, ego_vehicle.bounding_box_extent.x * 2
        adv_width, adv_length = target_vehicle.bounding_box_extent.y * 2, target_vehicle.bounding_box_extent.x * 2

        ego_location = ego_vehicle.get_transform().location
        target_location = target_vehicle.get_transform().location

        # Get the velocities of both vehicles
        ego_velocity = ego_vehicle.get_velocity()
        target_velocity = target_vehicle.get_velocity()

        time = 0.0
    
        half_w1, half_l1 = width / 2, length / 2
        half_w2, half_l2 = adv_width / 2, adv_length / 2

        ego_speed = math.sqrt(ego_velocity.x**2 + ego_velocity.y**2 + ego_velocity.z**2)
        target_speed = math.sqrt(target_velocity.x**2 + target_velocity.y**2 + target_velocity.z**2)

        if ego_speed == 0 and target_speed == 0:
            return GLOBALS.DEFAULT_FOR_TTC
        
        while time <= GLOBALS.MAX_TIME_FOR_TTC:
            # Update positions
            curr_pos1 = np.array(ego_location) + np.array(ego_velocity) * time
            curr_pos2 = np.array(target_location) + np.array(target_velocity) * time
            
            # Check if bounding boxes overlap
            if (abs(curr_pos1[0] - curr_pos2[0]) <= (half_w1 + half_w2)) and \
            (abs(curr_pos1[1] - curr_pos2[1]) <= (half_l1 + half_l2)):
                return time  # Collision detected
            
            time += GLOBALS.TTC_TIMESTEP  # Increment time step
        
        return GLOBALS.DEFAULT_FOR_TTC  # No collision within max_time
    
    def calculate_ttc_oldschool(self, ego_vehicle, target_vehicle):
 
        """
 
        Calculate the Time to Collision (TTC) considering the heading and speed of the ego vehicle
 
        and a target vehicle.
 

 
        Args:
 
            ego_vehicle (carla.Actor): The ego vehicle actor.
 
            target_vehicle (carla.Actor): The target vehicle actor.
 

 
        Returns:
 
            float: The TTC value in timesteps, or None if no collision is predicted.
 
        """
 
        # Get the current locations of the vehicles
 
        ego_location = ego_vehicle.get_transform().location
 
        target_location = target_vehicle.get_transform().location
 

 
        # Get the velocities of both vehicles
 
        ego_velocity = ego_vehicle.get_velocity()
 
        target_velocity = target_vehicle.get_velocity()
 

 
        # Convert velocities to speeds
 
        ego_speed = math.sqrt(ego_velocity.x**2 + ego_velocity.y**2 + ego_velocity.z**2)
 
        target_speed = math.sqrt(target_velocity.x**2 + target_velocity.y**2 + target_velocity.z**2)
 

 
        # If either vehicle is stationary, no collision can occur
 
        if ego_speed == 0 and target_speed == 0:
 
            return None
 

 
        for i in range (100):
 
            ego_projected = ego_location + ego_velocity * i
 
            target_projected = target_location + target_velocity * i
 
            distance =  ego_projected - target_projected
 
            if distance.x < 2 and distance.y < 2:
                return i
 
        return None
    
    def calculate_time_headway(self, ego_vehicle, target_vehicle):
        """
        Calculate the Time Headway considering the heading and speed of the ego vehicle
        and the position of the target vehicle.

        Args:
            ego_vehicle (carla.Actor): The ego vehicle actor.
            target_vehicle (carla.Actor): The target vehicle actor.

        Returns:
            float: The TH value in timesteps, or None if no collision is predicted.
        """
        # get bounding boxes:
        width, length = ego_vehicle.bounding_box_extent.y * 2, ego_vehicle.bounding_box_extent.x * 2
        adv_width, adv_length = target_vehicle.bounding_box_extent.y * 2, target_vehicle.bounding_box_extent.x * 2

        ego_location = ego_vehicle.get_transform().location
        target_location = target_vehicle.get_transform().location

        # Get the velocities of both vehicles
        ego_velocity = ego_vehicle.get_velocity()

        time = 0.0
    
        half_w1, half_l1 = width / 2, length / 2
        half_w2, half_l2 = adv_width / 2, adv_length / 2

        ego_speed = math.sqrt(ego_velocity.x**2 + ego_velocity.y**2 + ego_velocity.z**2)

        if ego_speed == 0:
            return GLOBALS.DEFAULT_FOR_TTC
        
        while time <= GLOBALS.MAX_TIME_FOR_TTC:
            # Update positions
            curr_pos1 = np.array(ego_location) + np.array(ego_velocity) * time
            
            # Check if bounding boxes overlap
            if (abs(curr_pos1[0] - target_location[0]) <= (half_w1 + half_w2)) and \
            (abs(curr_pos1[1] - target_location[1]) <= (half_l1 + half_l2)):
                return time  # Collision detected
            
            time += GLOBALS.TTC_TIMESTEP  # Increment time step
        
        return GLOBALS.DEFAULT_FOR_TTC  # No collision within max_time
    
    
    def calculate_enhanced_ttc(self, ego_vehicle, target_vehicle):
        """
        Function to calculate the minimal time it takes for two actors given their current speed, heading and acc
        to get to a collision state.

        Args:
            ego_vehicle (carla.Actor): The ego vehicle actor.
            target_vehicle (carla.Actor): The target vehicle actor.

        Returns:
            float: The TTC value in timesteps, or None if no collision is predicted.
        """
        # get bounding boxes:
        width, length = ego_vehicle.bounding_box_extent.y * 2, ego_vehicle.bounding_box_extent.x * 2
        adv_width, adv_length = target_vehicle.bounding_box_extent.y * 2, target_vehicle.bounding_box_extent.x * 2

        ego_location = ego_vehicle.get_transform().location
        target_location = target_vehicle.get_transform().location

        # Get the velocities of both vehicles
        ego_velocity = ego_vehicle.get_velocity()
        target_velocity = target_vehicle.get_velocity()

        time = 0.0
    
        half_w1, half_l1 = width / 2, length / 2
        half_w2, half_l2 = adv_width / 2, adv_length / 2

        ego_speed = math.sqrt(ego_velocity.x**2 + ego_velocity.y**2 + ego_velocity.z**2)
        target_speed = math.sqrt(target_velocity.x**2 + target_velocity.y**2 + target_velocity.z**2)

        ego_accel = ego_vehicle.get_acceleration()
        target_accel = target_vehicle.get_acceleration()

        if ego_speed == 0 and target_speed == 0:
            return GLOBALS.DEFAULT_FOR_TTC
        
        while time <= GLOBALS.MAX_TIME_FOR_TTC:
            # Update velocities
            curr_vel1 = np.array(ego_velocity) + np.array(ego_accel) * time
            curr_vel2 = np.array(target_velocity) + np.array(target_accel) * time
            
            # Update positions using kinematic equation
            curr_pos1 = np.array(ego_location) + np.array(ego_velocity) * time + 0.5 * np.array(ego_accel) * time**2
            curr_pos2 = np.array(target_location) + np.array(target_velocity) * time + 0.5 * np.array(target_accel) * time**2
            
            # Check if bounding boxes overlap
            if (abs(curr_pos1[0] - curr_pos2[0]) <= (half_w1 + half_w2)) and \
            (abs(curr_pos1[1] - curr_pos2[1]) <= (half_l1 + half_l2)):
                return time  # Collision detected
            
            time += GLOBALS.TTC_TIMESTEP  # Increment time step
        
        return GLOBALS.DEFAULT_FOR_TTC  # No collision within max_time
    
    def calculate_ttc_near_collision(self, ego_vehicle, target_vehicle, near_col_threshold=6):
        """
        Function to calculate the minimal time it takes for two actors given their current speed and heading
        to get to a near collision state.

        Args:
            ego_vehicle (carla.Actor): The ego vehicle actor.
            target_vehicle (carla.Actor): The target vehicle actor.
            near_col_threshold (int): Threshold of how far away the actors can be such that it still counts as near_collision
            

        Returns:
            float: The TTC value in timesteps, or None if no collision is predicted.
        """
        # get bounding boxes:
        width, length = ego_vehicle.bounding_box_extent.y * 2, ego_vehicle.bounding_box_extent.x * 2
        adv_width, adv_length = target_vehicle.bounding_box_extent.y * 2, target_vehicle.bounding_box_extent.x * 2

        ego_location = ego_vehicle.get_transform().location
        target_location = target_vehicle.get_transform().location

        # Get the velocities of both vehicles
        ego_velocity = ego_vehicle.get_velocity()
        target_velocity = target_vehicle.get_velocity()

        time = 0.0

        ego_speed = math.sqrt(ego_velocity.x**2 + ego_velocity.y**2 + ego_velocity.z**2)
        target_speed = math.sqrt(target_velocity.x**2 + target_velocity.y**2 + target_velocity.z**2)

        if ego_speed == 0 and target_speed == 0:
            return GLOBALS.DEFAULT_FOR_TTC
    
        while time <= GLOBALS.MAX_TIME_FOR_TTC:
            # Update positions using velocity
            curr_pos1 = np.array(ego_location) + np.array(ego_velocity) * time
            curr_pos2 = np.array(target_location) + np.array(target_velocity) * time
            
            # Compute Euclidean distance between centers
            distance = np.linalg.norm(curr_pos1 - curr_pos2)
            
            # Check if distance is below the threshold
            if distance <= GLOBALS.TTC_THRESHOLD:
                return time  # Near collision detected
            
            time += GLOBALS.TTC_TIMESTEP  # Increment time step
        
        return GLOBALS.DEFAULT_FOR_TTC   # No near collision within max_time
    

    def _generate_random_adversarial_route(self, num_waypoints):
        random.seed(datetime.now().timestamp())
        vehicle = self.actors["adversary"]

        x_start, y_start = vehicle.get_location().x, vehicle.get_location().y
        end_x, end_y = create_random_end_point(vehicle)

        # need two random control points to create a cubic bezier curve
        x_ctrl_1, y_ctrl_1 = create_random_control_point(vehicle)
        x_ctrl_2, y_ctrl_2 = create_random_control_point(vehicle)

        nodes = np.asfortranarray([
        [x_start, x_ctrl_1, x_ctrl_2, end_x],
        [y_start, y_ctrl_1, y_ctrl_2, end_y],
        ])
        
        curve = bezier.Curve.from_nodes(nodes)
        
        # calculate waypoints on the curve
        trajectory = []
        traj = []
        for i in np.linspace(0, 1, num_waypoints):
            speed = min(6.7, (i + 1) ** 15)
            point = curve.evaluate(i)
            trajectory.append([point[0][0], point[1][0], speed])
            traj.append([point[0][0], point[1][0]])

        ego_width, ego_length = vehicle.bounding_box.extent.y * 2, vehicle.bounding_box.extent.x * 2
        ego_traj = np.concatenate([
            traj,
            np.rad2deg(get_polyline_yaw(traj)).reshape(-1, 1)
        ], axis=1)

        #visualize tracks
        visualize_traj(
            ego_traj[4:-4, 0],
            ego_traj[4:-4, 1],
            ego_traj[4:-4, 2],
            ego_width,
            ego_length,
            (0, 5, 0)
        )

        return trajectory
    
    def _generate_parametrized_adversarial_route(self, num_waypoints):
        vehicle = self.actors["adversary"]

        x_start, y_start = vehicle.get_location().x, vehicle.get_location().y

        transform = vehicle.get_transform()
        x, y = transform.location.x, transform.location.y
        distance = self.parameters["distance_to_target"] # 50 to 55
        
        angle_offset = self.parameters["angle_to_target"] #random.uniform(-math.pi / 2, math.pi / 2)
        random_angle = math.radians(transform.rotation.yaw) + angle_offset

        end_x = x + distance * math.cos(random_angle)
        end_y = y + distance * math.sin(random_angle)


        # need two random control points to create a cubic bezier curve
        distance = self.parameters["distance_to_control"] # 20 to 30
        
        angle_offset = self.parameters["angle_to_control"] #random.uniform(-math.pi / 2, math.pi / 2)
        random_angle = math.radians(transform.rotation.yaw) + angle_offset
        x_ctrl_1 = x + distance * math.cos(random_angle)
        y_ctrl_1 = y + distance * math.sin(random_angle)

        nodes = np.asfortranarray([
        [x_start, x_ctrl_1, end_x],
        [y_start, y_ctrl_1, end_y],
        ])
        
        curve = bezier.Curve.from_nodes(nodes)
        
        # calculate waypoints on the curve
        trajectory = []
        traj = []
        for i in np.linspace(0, 1, num_waypoints):
            speed = min(6.7, (i + 1) ** 15)
            point = curve.evaluate(i)
            trajectory.append([point[0][0], point[1][0], speed])
            traj.append([point[0][0], point[1][0]])

        ego_width, ego_length = vehicle.bounding_box.extent.y * 2, vehicle.bounding_box.extent.x * 2
        ego_traj = np.concatenate([
            traj,
            np.rad2deg(get_polyline_yaw(traj)).reshape(-1, 1)
        ], axis=1)

        #visualize tracks
        visualize_traj(
            ego_traj[4:-4, 0],
            ego_traj[4:-4, 1],
            ego_traj[4:-4, 2],
            ego_width,
            ego_length,
            (0, 5, 0)
        )

        return trajectory
    
    
    def create_random_control_point(vehicle):
        half_width = 35
        length = 45
        transform = vehicle.get_transform()
        x, y = transform.location.x, transform.location.y
        heading = math.radians(transform.rotation.yaw)
        #create borders of where random points are allowed to be
        bottom_left_x = x - half_width * math.sin(heading)
        bottom_left_y = y + half_width * math.cos(heading)

        bottom_right_x = x + half_width * math.sin(heading)
        bottom_right_y = y - half_width * math.cos(heading)

        top_left_x = bottom_left_x + length * math.cos(heading)
        top_left_y = bottom_left_y + length * math.sin(heading)

        top_right_x = bottom_right_x + length * math.cos(heading)
        top_right_y = bottom_right_y + length * math.sin(heading)

        # Calculate min and max values
        x_min = min(bottom_left_x, bottom_right_x, top_left_x, top_right_x)
        x_max = max(bottom_left_x, bottom_right_x, top_left_x, top_right_x)
        y_min = min(bottom_left_y, bottom_right_y, top_left_y, top_right_y)
        y_max = max(bottom_left_y, bottom_right_y, top_left_y, top_right_y)

        random_x = random.uniform(x_min, x_max)
        random_y = random.uniform(y_min, y_max)
        return random_x, random_y

        
    def create_random_end_point(vehicle):
        transform = vehicle.get_transform()
        x, y = transform.location.x, transform.location.y
        distance = random.uniform(50, 55)
        
        angle_offset = random.uniform(-math.pi / 2, math.pi / 2)
        random_angle = math.radians(transform.rotation.yaw) + angle_offset

        new_x = x + distance * math.cos(random_angle)
        new_y = y + distance * math.sin(random_angle)

        return new_x, new_y


    def _generate_adversarial_route_iterative(self):
        features, ego_route = self._get_features_iterative()

        pred_trajectory, pred_score = self._sample_trajectories(features)
        scores = self._score_trajectories(pred_trajectory, pred_score, features)
        adv_traj_id, adv_traj, ego_traj = self._select_colliding_trajectory_iterative(features, pred_score, pred_trajectory)

        return features, adv_traj, ego_traj


    def _get_features_iterative(self):
        roadgraph_features = self._get_roadgraph_features(self._max_samples)
        self.filter_road_graph_for_odd_checking(roadgraph_features)
        state_features, ego_route = self._get_state_features_iterative("ego")
        dynamic_map_features = self._get_dynamic_map_features()

        ids = roadgraph_features["roadgraph_samples/id"]
        for new_id, old_id in enumerate(sorted(np.unique(ids))):
            roadgraph_features["roadgraph_samples/id"][ids == old_id] = new_id
            dynamic_map_features["traffic_light_state/past/id"][
                dynamic_map_features["traffic_light_state/past/id"] == old_id] = new_id
            dynamic_map_features["traffic_light_state/current/id"][
                dynamic_map_features["traffic_light_state/current/id"] == old_id] = new_id

        features = {}
        features.update(roadgraph_features)
        features.update(state_features)
        features.update(dynamic_map_features)
        features["scenario/id"] = np.array(["template"])
        features["state/objects_of_interest"] = features['state/tracks_to_predict'].copy()
        return features, ego_route

    def _get_state_features_iterative(self, ego_agent: str) -> tuple[dict, np.ndarray]:
        state_features = {
            "state/id": np.full([128, ], -1, dtype=np.int64),
            "state/type": np.full([128, ], 0, dtype=np.int64),
            "state/is_sdc": np.full([128, ], 0, dtype=np.int64),
            'state/tracks_to_predict': np.full([128, ], 0, dtype=np.int64),
            'state/current/bbox_yaw': np.full([128, 1], -1, dtype=np.float32),
            'state/current/height': np.full([128, 1], -1, dtype=np.float32),
            'state/current/length': np.full([128, 1], -1, dtype=np.float32),
            'state/current/valid': np.full([128, 1], 0, dtype=np.int64),
            'state/current/vel_yaw': np.full([128, 1], -1, dtype=np.float32),
            'state/current/velocity_x': np.full([128, 1], -1, dtype=np.float32),
            'state/current/velocity_y': np.full([128, 1], -1, dtype=np.float32),
            'state/current/width': np.full([128, 1], -1, dtype=np.float32),
            'state/current/x': np.full([128, 1], -1, dtype=np.float32),
            'state/current/y': np.full([128, 1], -1, dtype=np.float32),
            'state/current/z': np.full([128, 1], -1, dtype=np.float32),
            'state/past/bbox_yaw': np.full([128, 10], -1, dtype=np.float32),
            'state/past/height': np.full([128, 10], -1, dtype=np.float32),
            'state/past/length': np.full([128, 10], -1, dtype=np.float32),
            'state/past/valid': np.full([128, 10], 0, dtype=np.int64),
            'state/past/vel_yaw': np.full([128, 10], -1, dtype=np.float32),
            'state/past/velocity_x': np.full([128, 10], -1, dtype=np.float32),
            'state/past/velocity_y': np.full([128, 10], -1, dtype=np.float32),
            'state/past/width': np.full([128, 10], -1, dtype=np.float32),
            'state/past/x': np.full([128, 10], -1, dtype=np.float32),
            'state/past/y': np.full([128, 10], -1, dtype=np.float32),
            'state/past/z': np.full([128, 10], -1, dtype=np.float32),
            'state/future/bbox_yaw': np.full([128, 80], -1, dtype=np.float32),
            'state/future/height': np.full([128, 80], -1, dtype=np.float32),
            'state/future/length': np.full([128, 80], -1, dtype=np.float32),
            'state/future/valid': np.full([128, 80], 0, dtype=np.int64),
            'state/future/vel_yaw': np.full([128, 80], -1, dtype=np.float32),
            'state/future/velocity_x': np.full([128, 80], -1, dtype=np.float32),
            'state/future/velocity_y': np.full([128, 80], -1, dtype=np.float32),
            'state/future/width': np.full([128, 80], -1, dtype=np.float32),
            'state/future/x': np.full([128, 80], -1, dtype=np.float32),
            'state/future/y': np.full([128, 80], -1, dtype=np.float32),
            'state/future/z': np.full([128, 80], -1, dtype=np.float32)
        }

        for i, actor_id in enumerate(self.agents):
            actor = self.actors[actor_id]
            state_features["state/id"][i] = actor.id
            state_features["state/type"][i] = AgentTypes.from_carla_type(actor).value
            state_features["state/is_sdc"][i] = 1 if actor_id == self._ego_agent else 0
            state_features["state/tracks_to_predict"][i] = i < len(self._trajectories)

            for t in range(min(len(self._trajectories[actor_id]), 91)):
                min_length = min(len(self._trajectories[actor_id]), 91) - 1
                offset = 0
                if t > min_length - 11 and t < min_length:
                    offset = min_length - 10
                    time = "past"
                elif t == min_length:
                    offset = min_length
                    time = "current"
                else:
                    offset = 51
                    time = ""
                if time!= "":
                    state = self._trajectories[actor_id][t]
                    for key in state:
                        state_features[f"state/{time}/{key}"][i, t - offset] = state[key]

        is_idc = state_features["state/is_sdc"] == 1
        ego_route = np.concatenate([
            state_features["state/future/x"][is_idc],
            state_features["state/future/y"][is_idc],
        ])
        return state_features, ego_route

    def _select_colliding_trajectory_iterative(self, features, pred_score, pred_trajectory):
        trajs_OV = pred_trajectory[self.agents.index(self._adv_agent)]
        probs_OV = pred_score[self.agents.index(self._adv_agent)]
        probs_OV[6:] = probs_OV[6]
        probs_OV = np.exp(probs_OV)
        probs_OV = probs_OV / np.sum(probs_OV)
        res = np.zeros(pred_trajectory.shape[1])
        min_dist = np.full(pred_trajectory.shape[1], fill_value=1000000)
        ego = self.actors[self._ego_agent]
        adversary = (set(self.agents) - {self._ego_agent}).pop()
        adversary = self.actors[adversary]
        adv_width, adv_length = adversary.bounding_box.extent.y * 2, adversary.bounding_box.extent.x * 2
        width, length = ego.bounding_box.extent.y * 2, ego.bounding_box.extent.x * 2
        
        trajs_AV = pred_trajectory[self.agents.index(self._ego_agent)]
        probs_AV = pred_score[self.agents.index(self._ego_agent)]
        probs_AV[16:] = probs_AV[16]
        probs_AV = np.exp(probs_AV)
        probs_AV = probs_AV / np.sum(probs_AV)

        for j, prob_OV in enumerate(probs_OV):
            P1 = prob_OV
            traj_OV = trajs_OV[j][::5]
            traj_OV_plot = trajs_OV[j]
            full_adv_traj = get_full_trajectory(self.agents.index(self._adv_agent), features, future=traj_OV_plot)[:,::-1]
            full_adv_traj = np.concatenate([
                full_adv_traj,
                np.rad2deg(get_polyline_yaw(full_adv_traj)).reshape(-1, 1)
            ], axis=1)
            # CHeck if on roadgraph here:
            """visualize_traj(
                full_adv_traj[4:-4, 0],
                full_adv_traj[4:-4, 1],
                full_adv_traj[4:-4, 2],
                1.4,
                2.9,
                (5, 5, 5)
            )"""
            yaw_OV = get_polyline_yaw(trajs_OV[j])[::5].reshape(-1, 1)
            width_OV = adv_width
            length_OV = adv_length
            cos_theta = np.cos(yaw_OV)
            sin_theta = np.sin(yaw_OV)
            bbox_OV = np.concatenate([
                traj_OV,
                yaw_OV,
                traj_OV[:, 0].reshape(-1,
                                        1) + 0.5 * length_OV * cos_theta + 0.5 * width_OV * sin_theta,
                traj_OV[:, 1].reshape(-1,
                                        1) + 0.5 * length_OV * sin_theta - 0.5 * width_OV * cos_theta,
                traj_OV[:, 0].reshape(-1,
                                        1) + 0.5 * length_OV * cos_theta - 0.5 * width_OV * sin_theta,
                traj_OV[:, 1].reshape(-1,
                                        1) + 0.5 * length_OV * sin_theta + 0.5 * width_OV * cos_theta,
                traj_OV[:, 0].reshape(-1,
                                        1) - 0.5 * length_OV * cos_theta - 0.5 * width_OV * sin_theta,
                traj_OV[:, 1].reshape(-1,
                                        1) - 0.5 * length_OV * sin_theta + 0.5 * width_OV * cos_theta,
                traj_OV[:, 0].reshape(-1,
                                        1) - 0.5 * length_OV * cos_theta + 0.5 * width_OV * sin_theta,
                traj_OV[:, 1].reshape(-1,
                                        1) - 0.5 * length_OV * sin_theta - 0.5 * width_OV * cos_theta
            ], axis=1)

            for i, prob_AV in enumerate(probs_AV):
                P2 = prob_AV
                traj_AV = trajs_AV[i][::5]
                traj_AV_plot = trajs_AV[i]
                full_ego_traj = get_full_trajectory(self.agents.index(self._ego_agent), features, future=traj_AV_plot)[:,::-1]
                full_ego_traj = np.concatenate([
                    full_ego_traj,
                    np.rad2deg(get_polyline_yaw(full_ego_traj)).reshape(-1, 1)
                ], axis=1)
                # CHeck if on roadgraph here:
                """visualize_traj(
                    full_ego_traj[4:-4, 0],
                    full_ego_traj[4:-4, 1],
                    full_ego_traj[4:-4, 2],
                    1.4,
                    2.9,
                    (5, 0, 5)
                )"""

                yaw_AV = get_polyline_yaw(trajs_AV[i])[::5].reshape(-1, 1)
                width_AV = width
                length_AV = length
                cos_theta = np.cos(yaw_AV)
                sin_theta = np.sin(yaw_AV)

                bbox_AV = np.concatenate((traj_AV, yaw_AV, \
                                            traj_AV[:, 0].reshape(-1,
                                                                1) + 0.5 * length_AV * cos_theta + 0.5 * width_AV * sin_theta, \
                                            traj_AV[:, 1].reshape(-1,
                                                                1) + 0.5 * length_AV * sin_theta - 0.5 * width_AV * cos_theta, \
                                            traj_AV[:, 0].reshape(-1,
                                                                1) + 0.5 * length_AV * cos_theta - 0.5 * width_AV * sin_theta, \
                                            traj_AV[:, 1].reshape(-1,
                                                                1) + 0.5 * length_AV * sin_theta + 0.5 * width_AV * cos_theta, \
                                            traj_AV[:, 0].reshape(-1,
                                                                1) - 0.5 * length_AV * cos_theta - 0.5 * width_AV * sin_theta, \
                                            traj_AV[:, 1].reshape(-1,
                                                                1) - 0.5 * length_AV * sin_theta + 0.5 * width_AV * cos_theta, \
                                            traj_AV[:, 0].reshape(-1,
                                                                1) - 0.5 * length_AV * cos_theta + 0.5 * width_AV * sin_theta, \
                                            traj_AV[:, 1].reshape(-1,
                                                                1) - 0.5 * length_AV * sin_theta - 0.5 * width_AV * cos_theta),
                                            axis=1)

                P3 = 0
                uncertainty = 1.
                alpha = 0.99
                '''
                B-A  F-E
                | |  | |
                C-D  G-H
                '''
                for (Cx1, Cy1, yaw1, xA, yA, xB, yB, xC, yC, xD, yD), (
                        Cx2, Cy2, yaw2, xE, yE, xF, yF, xG, yG, xH, yH) in zip(bbox_AV, bbox_OV):
                    uncertainty *= alpha
                    ego_adv_dist = np.linalg.norm([Cx1 - Cx2, Cy1 - Cy2])
                    if ego_adv_dist < min_dist[j]:
                        min_dist[j] = ego_adv_dist
                    if ego_adv_dist >= np.linalg.norm(
                            [0.5 * length_AV, 0.5 * width_AV]) + np.linalg.norm(
                        [0.5 * length_OV, 0.5 * width_OV]):
                        pass
                    elif Intersect([xA, yA, xB, yB], [xE, yE, xF, yF]) or Intersect(
                            [xA, yA, xB, yB],
                            [xF, yF, xG, yG]) or \
                            Intersect([xA, yA, xB, yB], [xG, yG, xH, yH]) or Intersect(
                        [xA, yA, xB, yB],
                        [xH, yH, xE, yE]) or \
                            Intersect([xB, yB, xC, yC], [xE, yE, xF, yF]) or Intersect(
                        [xB, yB, xC, yC],
                        [xF, yF, xG, yG]) or \
                            Intersect([xB, yB, xC, yC], [xG, yG, xH, yH]) or Intersect(
                        [xB, yB, xC, yC],
                        [xH, yH, xE, yE]) or \
                            Intersect([xC, yC, xD, yD], [xE, yE, xF, yF]) or Intersect(
                        [xC, yC, xD, yD],
                        [xF, yF, xG, yG]) or \
                            Intersect([xC, yC, xD, yD], [xG, yG, xH, yH]) or Intersect(
                        [xC, yC, xD, yD],
                        [xH, yH, xE, yE]) or \
                            Intersect([xD, yD, xA, yA], [xE, yE, xF, yF]) or Intersect(
                        [xD, yD, xA, yA],
                        [xF, yF, xG, yG]) or \
                            Intersect([xD, yD, xA, yA], [xG, yG, xH, yH]) or Intersect(
                        [xD, yD, xA, yA],
                        [xH, yH, xE, yE]):
                        P3 = uncertainty
                        break

                res[j] += P1 * P2 * P3
            """world = CarlaDataProvider.get_world()
            spectator = world.get_spectator()
            ego_loc = ego.get_location()
            spectator.set_transform(carla.Transform(
                carla.Location(ego_loc.x, ego_loc.y, ego_loc.z + 60),
                carla.Rotation(pitch=-90)
            ))
            settings = world.get_settings()
            world.tick()"""
        if np.any(res):
            adv_traj_id = np.argmax(res)
        else:
            adv_traj_id = np.argmin(min_dist)

        """hist_AV = self.AV_history[:,:11, :2]

        #------------------------newcode-------------------------
        if self.objective_model is not None:
            total_scores = None
            for ego_hist, ego_traj in zip(hist_AV, trajs_AV):
                ego_full = np.concatenate((ego_hist, ego_traj), axis=0)
                ego_full = np.concatenate((ego_full, np.ones((len(ego_full), 1))), axis=1).astype(np.float64)
                clean_advs = []
                for traj_OV in trajs_OV:
                    traj_full = np.concatenate((self.storage[self.env.current_seed].get('adv_past'), traj_OV), axis=0)
                    traj_yaw = get_polyline_yaw(traj_full).reshape(-1, 1)
                    base_ones = np.ones((len(traj_full), 1))
                    # Want index (0, 1) = (x, y), (4) = (heading), (-1) = (valid)
                    traj_full = np.concatenate((traj_full, base_ones, base_ones, traj_yaw, base_ones), axis=1)
                    traj_full_clean = clean_traj(ego_full, traj_full)
                    clean_advs.append(torch.from_numpy(traj_full_clean).to(torch.float32))
                scores = self.objective_model(None, clean_advs, clean_advs)
                if total_scores is None:
                    total_scores = torch.clone(scores)
                else:
                    total_scores += scores
            if self.objective_model_mode == 'both':
                adv_traj_id = total_scores.mean(dim=-1).argmax().item()
            elif self.objective_model_mode == 'sc':
                adv_traj_id = total_scores[:, 0].argmax().item()
            elif self.objective_model_mode == 'diff':
                adv_traj_id = total_scores[:, 1].argmax().item()
            else:
                raise ValueError('Unexpected objective_model_mode')
        #--------------------------------------------------------"""

        adv_path = trajs_OV[adv_traj_id]
        adv_yaw = get_polyline_yaw(adv_path).reshape(-1, 1)
        adv_vel = np.linalg.norm(get_polyline_vel(adv_path), axis=1).reshape(-1, 1)

        ego_path = trajs_OV[0]
        ego_yaw = get_polyline_yaw(ego_path).reshape(-1, 1)
        ego_vel = np.linalg.norm(get_polyline_vel(ego_path), axis=1).reshape(-1, 1)

        ego_traj = get_full_trajectory(self.agents.index(self._ego_agent), features)[:, ::-1]
        adv_traj_original = get_full_trajectory(self.agents.index(self._adv_agent), features)[:, ::-1]
        full_adv_traj = get_full_trajectory(self.agents.index(self._adv_agent), features, future=adv_path)[:,
                        ::-1]
        ego_traj = np.concatenate([
            ego_traj,
            np.rad2deg(get_polyline_yaw(ego_traj)).reshape(-1, 1)
        ], axis=1)
        full_adv_traj = np.concatenate([
            full_adv_traj,
            np.rad2deg(get_polyline_yaw(full_adv_traj)).reshape(-1, 1)
        ], axis=1)
        adv_traj_original = np.concatenate([
            adv_traj_original,
            np.rad2deg(get_polyline_yaw(adv_traj_original)).reshape(-1, 1)
        ], axis=1)

        ego_width, ego_length = ego.bounding_box.extent.y * 2, ego.bounding_box.extent.x * 2
        adv_width, adv_length = adversary.bounding_box.extent.y * 2, adversary.bounding_box.extent.x * 2
        visualize_traj(
            ego_traj[4:-4, 0],
            ego_traj[4:-4, 1],
            ego_traj[4:-4, 2],
            ego_width,
            ego_length,
            (0, 5, 0)
        )
        visualize_traj(
            adv_traj_original[4:-4, 0],
            adv_traj_original[4:-4, 1],
            adv_traj_original[4:-4, 2],
            adv_width,
            adv_length,
            (0, 0, 5)
        )
        visualize_traj(
            full_adv_traj[4:-4, 0],
            full_adv_traj[4:-4, 1],
            full_adv_traj[4:-4, 2],
            adv_width,
            adv_length,
            (5, 0, 0)
        )

        world = CarlaDataProvider.get_world()
        spectator = world.get_spectator()
        ego_loc = ego.get_location()
        spectator.set_transform(carla.Transform(
            carla.Location(ego_loc.x, ego_loc.y, ego_loc.z + 60),
            carla.Rotation(pitch=-90)
        ))
        settings = world.get_settings()
        settings.synchronous_mode = False
        adv_traj = np.concatenate([adv_path[:, ::-1], adv_vel, adv_yaw], axis=1)
        ego_traj = np.concatenate([ego_path[:, ::-1], ego_vel, ego_yaw], axis=1)
        return adv_traj_id, adv_traj, ego_traj


