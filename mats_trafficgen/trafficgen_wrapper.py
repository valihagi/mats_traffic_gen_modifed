from __future__ import annotations

import enum
import logging
import math
import os
import pickle
import random
import tempfile
import time
from copy import deepcopy
from enum import Enum
from typing import Any

import carla
import gymnasium
import gymnasium.spaces
import numpy as np
import optree
import scenic.domains.driving.roads
import torch
from pettingzoo.utils.env import AgentID, ActionType, ObsType
from scenic.core.object_types import Point, OrientedPoint
from scenic.core.vectors import Vector
from scenic.domains.driving.roads import Network, Lane, VehicleType, LaneGroup, Road
from scenic.formats.opendrive import xodr_parser
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from scenic.simulators.carla.utils.utils import scenicToCarlaLocation, carlaToScenicPosition

from mats_gym.envs.scenario_env_wrapper import BaseScenarioEnvWrapper

from mats_trafficgen.trajectory_following import TrajectoryFollowingAgent
from trafficgen.trafficgen.traffic_generator.traffic_generator import TrafficGen
from trafficgen.trafficgen.traffic_generator.utils.data_utils import process_data_to_internal_format
from trafficgen.trafficgen.utils.config import load_config_init
from trafficgen.trafficgen.utils.utils import rotate

from mats_trafficgen.encodings import LaneEncoder, RoadEdgeEncoder, ActorEncoder, \
    TrafficLightEncoder, CrossWalkEncoder

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

class TrafficGenWrapper(BaseScenarioEnvWrapper):
    """
    Wrapper to add road information to the observation.
    Road information includes:
    - Identification of the current lane (road, section, lane)
    - Lane type and width
    - Lane change possibility
    """

    def __init__(
            self,
            env: BaseScenarioEnvWrapper,
            max_samples: int = 20000,
            max_radius: float = 50.0,
            line_resolution: float = 5.0,
            num_time_steps: int = 190,
            time_step: float = 0.1,
    ):
        """
        @param env: The environment to wrap.
        """
        self._max_samples = max_samples
        self._max_radius = max_radius
        self._line_resolution = line_resolution
        self._num_time_steps = num_time_steps
        self._time_step = time_step
        self._network: Network = None
        self._lane_ids: list[str] = []
        self._object_id = 0
        self._debug = True

        cfg = load_config_init("local")
        logging.debug("Load TrafficGen checkpoint.")
        self._trafficgen = TrafficGen(cfg)
        init_vis_dir = 'logs/output/vis/scene_initialized'
        if not os.path.exists(init_vis_dir):
            os.makedirs(init_vis_dir)
        tmp_pth = 'logs/output/initialized_tmp'
        if not os.path.exists(tmp_pth):
            os.makedirs(tmp_pth)

        self._trafficgen.init_model.eval()
        super().__init__(env)

    def _get_centerlines(self) -> dict[tuple, np.ndarray]:
        centerlines = {}
        for start, _ in self._topology:
            line = []
            prev = None
            dist = self._line_resolution
            previous_wps = start.previous(dist)
            while len(previous_wps) > 0 and dist > 0:
                previous_wps = start.previous(dist)
                prev = min(previous_wps,
                           key=lambda wp: wp.transform.location.distance(start.transform.location))
                dist -= 0.5

            wps = start.next_until_lane_end(self._line_resolution)
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
                line.append([location.x, location.y, type.value])
            line = np.array(line, dtype=np.float32)
            id = (start.road_id, start.lane_id)
            centerlines[id] = line
        return centerlines


    def from_list_to_array(self, inp_list, max_agent=32):
        agent = np.concatenate([x.get_inp(act=True) for x in inp_list], axis=0)
        agent = agent[:max_agent]
        agent_num = agent.shape[0]
        agent = np.pad(agent, ([0, max_agent - agent_num], [0, 0]))
        agent_mask = np.zeros([agent_num])
        agent_mask = np.pad(agent_mask, ([0, max_agent - agent_num]))
        agent_mask[:agent_num] = 1
        agent_mask = agent_mask.astype(bool)
        return agent, agent_mask



    def _visualize(self, lanes, edges, crosswalks, traffic_lights):
        world = CarlaDataProvider.get_world()
        color = {
            1: carla.Color(5, 0, 0),
            2: carla.Color(0, 5, 0),
            3: carla.Color(0, 0, 5),
            15: carla.Color(5, 5, 0),
            16: carla.Color(0, 5, 5),
            18: carla.Color(5, 0, 5),
        }
        for i in np.unique(lanes[:, 3]):
            xys = lanes[lanes[:, 3] == i][:, :3]
            for s, e in zip(xys[:-1], xys[1:]):
                world.debug.draw_arrow(
                    carla.Location(x=s[0].item(), y=s[1].item(), z=0.2),
                    carla.Location(x=e[0].item(), y=e[1].item(), z=0.2),
                    thickness=0.1,
                    color=color[int(s[2])],
                    life_time=0
                )
        for i in np.unique(edges[:, 3]):
            xys = edges[edges[:, 3] == i][:, :3]
            for s, e in zip(xys[:-1], xys[1:]):
                world.debug.draw_arrow(
                    carla.Location(x=s[0].item(), y=s[1].item(), z=0.2),
                    carla.Location(x=e[0].item(), y=e[1].item(), z=0.2),
                    thickness=0.1,
                    color=color[int(s[2])],
                    life_time=0
                )
        for i in np.unique(crosswalks[:, 3]):
            xys = crosswalks[crosswalks[:, 3] == i][:, :3]
            for s, e in zip(xys[:-1], xys[1:]):
                world.debug.draw_arrow(
                    carla.Location(x=s[0].item(), y=s[1].item(), z=0.5),
                    carla.Location(x=e[0].item(), y=e[1].item(), z=0.5),
                    thickness=0.1,
                    color=color[int(s[2])],
                    life_time=0
                )
            world.debug.draw_arrow(
                carla.Location(x=xys[-1][0].item(), y=xys[-1][1].item(), z=0.5),
                carla.Location(x=xys[0][0].item(), y=xys[0][1].item(), z=0.5),
                thickness=0.1,
                color=color[18],
                life_time=0
            )

        for traffic_light in traffic_lights:
            world.debug.draw_box(
                carla.BoundingBox(
                    carla.Location(x=traffic_light[1], y=traffic_light[2], z=0.2),
                    carla.Vector3D(x=0.5, y=0.5, z=0.5)
                ),
                carla.Rotation(yaw=0),
                thickness=0.1,
                color=carla.Color(0, 0, 5),
                life_time=0
            )


    def _spawn_actors(self, model_output, trajectories):
        world = CarlaDataProvider.get_world()
        ego = self.actors[self.agents[0]]
        ego_heading = -np.deg2rad(ego.get_transform().rotation.yaw)
        num_agents = trajectories.shape[1]
        agents = []
        for i in range(num_agents):
            agent = model_output[i+1]
            path = trajectories[:, i]
            color = np.random.randint(0, 5, 3).tolist()
            color = carla.Color(color[0], color[1], color[2])
            position, heading = agent[0:2], agent[4].item()
            x, y = rotate(position[0], position[1], ego_heading)
            heading += ego_heading

            y = -y
            x = ego.get_location().x + x
            y = ego.get_location().y + y
            heading = -heading

            if heading < -np.pi:
                heading += 2 * np.pi
            elif heading > np.pi:
                heading -= 2 * np.pi

            spawn_point = carla.Transform(
                location=carla.Location(x=x, y=y, z=0.2),
                rotation=carla.Rotation(yaw=np.rad2deg(heading))
            )

            bbox = carla.BoundingBox(spawn_point.location, carla.Vector3D(x=2.5, y=1, z=0.05))
            world.debug.draw_box(
                bbox,
                spawn_point.rotation,
                thickness=0.1,
                color=color,
                life_time=0
            )
            world.debug.draw_arrow(
                spawn_point.location,
                carla.Location(
                    x=spawn_point.location.x + 2 * np.cos(heading),
                    y=spawn_point.location.y + 2 * np.sin(heading),
                    z=0.2
                ),
                thickness=0.1,
                color=color,
                life_time=0
            )

            traj = []
            for t in range(path.shape[0]):
                x, y = path[t, :2]
                heading = path[t, 4]
                x, y = rotate(x, y, ego_heading)
                y = -y
                x = ego.get_location().x + x
                y = ego.get_location().y + y
                heading += ego_heading
                heading = -heading

                if heading < -np.pi:
                    heading += 2 * np.pi
                elif heading > np.pi:
                    heading -= 2 * np.pi
                tf = carla.Transform(
                    location=carla.Location(x=x, y=y, z=0.2),
                    rotation=carla.Rotation(yaw=np.rad2deg(heading))
                )
                speed = np.linalg.norm(path[t, 2:4]) * 3.6
                traj.append((tf.location, speed))

            for i, (start, end) in enumerate(list(zip(traj[:-1], traj[1:]))[::5]):
                t = 0.1 * i * 5
                start_loc = start[0]
                end_loc = end[0]
                world.debug.draw_arrow(
                    start_loc,
                    end_loc,
                    thickness=0.1,
                    color=color,
                    life_time=0
                )
                world.debug.draw_string(
                    start_loc + carla.Location(z=0.5),
                    f"{t:.1f}",
                    draw_shadow=True, color=color,
                    life_time=1000000
                )

            npc = CarlaDataProvider.request_new_actor("vehicle.audi.a2", spawn_point)
            if npc is not None:
                agent = TrajectoryFollowingAgent(npc, traj)
                agents.append(agent)
                #npc.set_autopilot(True)
                #tm: carla.TrafficManager = self.client.get_trafficmanager(CarlaDataProvider.get_traffic_manager_port())
                #tm.set_path(npc, [tf.location for tf in traj])

        return agents


    def init_map(self, use_init_model: bool = True, max_agents: int = 32):
        start = time.time()
        centerlines = self._get_centerlines()
        print(f"lane encoding time: {time.time() - start}")

        start = time.time()
        edges, _ = self._edge_encoder.encode(self.client)
        print(f"edge encoding time: {time.time() - start}")

        start = time.time()
        crosswalks, _ = self._crosswalk_encoder.encode(self.client)
        print(f"crosswalk encoding time: {time.time() - start}")

        start = time.time()
        actors, actor_info = self._actor_encoder.encode(self.client)
        print(f"actor encoding time: {time.time() - start}")

        start = time.time()
        traffic_lights = self._traffic_light_encoder.encode(self.client)
        print(f"traffic light encoding time: {time.time() - start}")

        start = time.time()

        # Add ids to lanes and edges
        lane_ids = list(sorted(centerlines.keys()))
        lanes = []
        for i, lane_id in enumerate(lane_ids):
            lane = centerlines[lane_id]
            lane = np.concatenate([lane, np.full((len(lane), 1), i)], axis=1)
            lanes.append(lane)

        lanes = np.concatenate(lanes)
        edges = np.concatenate([np.concatenate([edge, np.full((len(edge), 1), i + len(lanes))], axis=1) for i, edge in enumerate(edges)])
        crosswalks = np.concatenate([np.concatenate([crosswalk, np.full((len(crosswalk), 1), i + len(lanes) + len(edges))], axis=1) for i, crosswalk in enumerate(crosswalks)])

        # Associate traffic lights with lane ids
        traffic_light_features = []
        for lane_id in traffic_lights:
            if lane_id in centerlines:
                idx = lane_ids.index(lane_id)
                feat = np.concatenate([[idx], traffic_lights[lane_id]], axis=0)
                traffic_light_features.append(feat)
        traffic_lights = np.array(traffic_light_features)


        # Add ego to the front of the actor list
        ego_id = self.actors[self.agents[0]].id
        ego_loc = self.actors[self.agents[0]].get_location()
        ego_loc = np.array([ego_loc.x, -ego_loc.y])

        actors = np.array(actors)
        actors = np.concatenate([
            [actors[actor_info.index(ego_id)]],
            actors[:actor_info.index(ego_id)],
            actors[actor_info.index(ego_id) + 1:]
        ])

        #lanes = lanes[np.linalg.norm(lanes[:, :2] - ego_loc, axis=1) < self._max_radius]
        #edges = edges[np.linalg.norm(edges[:, :2] - ego_loc, axis=1) < self._max_radius]
        #crosswalks = crosswalks[np.linalg.norm(crosswalks[:, :2] - ego_loc , axis=1) < self._max_radius]
        #traffic_lights = traffic_lights[np.linalg.norm(traffic_lights[:, 1:3] - ego_loc, axis=1) < self._max_radius]
        self._visualize(lanes, edges, crosswalks, traffic_lights)

        # Transform coordinates
        actors = np.repeat(actors[np.newaxis], 20, axis=0)
        actors[..., 1] = -actors[..., 1]
        actors[..., 4] = -actors[..., 4]

        roadgraph = np.concatenate([lanes, edges, crosswalks], axis=0)
        roadgraph[..., 1] = -roadgraph[..., 1]


        traffic_lights =  np.repeat(traffic_lights[np.newaxis], 20, axis=0)
        traffic_lights[..., 2] = -traffic_lights[..., 2]



        scene = {
            "all_agent": actors, # necessary due to a bug in trafficgen
            "lane": roadgraph[np.linalg.norm(roadgraph[:, :2] - ego_loc, axis=1) < self._max_radius],
            "traffic_light":  traffic_lights,
            "unsampled_lane": roadgraph[:]
        }
        print(f"formatting time: {time.time() - start}")

        start = time.time()
        processed_scene = process_data_to_internal_format(scene)[0]
        print(f"processing time: {time.time() - start}")
        processed_scene = optree.tree_map(lambda x: np.expand_dims(x, axis=0), processed_scene)


        with torch.no_grad():
            processed_scene["agent_mask"][..., :18] = 1
            data = deepcopy(processed_scene)
            start = time.time()
            if True:
                model_output = self._trafficgen.place_vehicles_for_single_scenario(
                    batch=data,
                    index=0,
                    vis=True,
                    vis_dir="logs/output/vis/scene_initialized",
                    context_num=0
                )
                print(f"trafficgen init time: {time.time() - start}")
                model_output = optree.tree_map(lambda x: x.cpu().numpy() if isinstance(x, torch.Tensor) else x, model_output)
                agent, agent_mask = self.from_list_to_array(model_output['agent'])

                # agents = []
                # world = CarlaDataProvider.get_world()
                # ego_loc = self.actors[self.agents[0]].get_location()
                # ego_yaw = np.deg2rad(self.actors[self.agents[0]].get_transform().rotation.yaw)
                # for actor in world.get_actors().filter("vehicle.*"):
                #     loc = actor.get_location()
                #     yaw = actor.get_transform().rotation.yaw
                #     yaw = np.deg2rad(yaw)
                #     width, length = actor.bounding_box.extent.y * 2, actor.bounding_box.extent.x * 2
                #     x, y = loc.x - ego_loc.x, loc.y - ego_loc.y
                #     x, y = rotate(x, y, -ego_yaw)
                #     yaw = yaw - ego_yaw
                #     if yaw < -np.pi:
                #         yaw += 2 * np.pi
                #     elif yaw > np.pi:
                #         yaw -= 2 * np.pi
                #     #if yaw <= 0:
                #     #    yaw = -yaw
                #     #else:
                #     #    yaw = 2*np.pi - yaw
                #     agents.append([x, y, 0.0, 0.0, yaw, length, width])
                # agent = np.array(agents)
                # agent_mask = np.zeros((max_agents,), dtype=bool)
                # agent_mask[:len(agents)] = True

            output = {}
            output['context_num'] = 0
            output['all_agent'] = agent
            output['agent_mask'] = agent_mask
            output['lane'] = data['other']['unsampled_lane'][0]
            output['unsampled_lane'] = data['other']['unsampled_lane'][0]
            output['traf'] = np.repeat(data['other']['traf'][0], 190, axis=0)
            output['gt_agent'] = data['other']['gt_agent'][0]
            output['gt_agent_mask'] = data['other']['gt_agent_mask'][0]

            start = time.time()
            pred_i = self._trafficgen.inference_control(output, ego_gt=False)
            print(f"trafficgen act time: {time.time() - start}")


        return self._spawn_actors(agent, pred_i[:, 1:])

    def step(
        self, actions: dict[AgentID, ActionType]
    ) -> tuple[
        dict[AgentID, ObsType],
        dict[AgentID, float],
        dict[AgentID, bool],
        dict[AgentID, bool],
        dict[AgentID, dict],
    ]:
        world = CarlaDataProvider.get_world()
        spectator = world.get_spectator()
        ego = self.actors[self.agents[0]]
        tf = ego.get_transform()
        spectator.set_transform(carla.Transform(
            location=carla.Location(
                x=tf.location.x,
                y=tf.location.y,
                z=30
            ),
            rotation=carla.Rotation(
                pitch=-70,
            )
        ))
        for agent in self.npcs:
            ctrl = agent.run_step()
            agent._vehicle.apply_control(ctrl)
        obs, reward, done, truncated, info = super().step(actions)
        return obs, reward, done, truncated, info

    def reset(
            self, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict, dict[Any, dict]]:
        obs, info = super().reset(seed, options)
        map: carla.Map = CarlaDataProvider.get_map()
        town = map.name.split("/")[-1]
        self._network = Network.fromFile(f"scenarios/maps/{town}.xodr", useCache=True)
        self._lane_ids = [l.id for l in self._network.lanes]
        self._topology = map.get_topology()

        self._lane_encoder = LaneEncoder(
            map=CarlaDataProvider.get_map(),
            line_resolution=self._line_resolution,
            debug=self._debug,
        )
        self._edge_encoder = RoadEdgeEncoder(
            network=self._network,
            line_resolution=self._line_resolution,
            debug=self._debug,
        )
        self._actor_encoder = ActorEncoder()
        self._traffic_light_encoder = TrafficLightEncoder()
        self._crosswalk_encoder = CrossWalkEncoder(map=map)


        self.npcs = self.init_map(use_init_model=False)
        CarlaDataProvider.get_world().tick()
        world = CarlaDataProvider.get_world()
        settings = world.get_settings()
        settings.synchronous_mode = False
        obs = {
            agent: self.env.observe(agent)
            for agent in self.agents
        }

        #world.apply_settings(settings)
        return obs, info

