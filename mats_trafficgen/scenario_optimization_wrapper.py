from __future__ import annotations

import dataclasses
import enum
import logging
import math
import os
import pickle
import random
import tempfile
import time
from collections import deque
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
from agents.navigation.global_route_planner import GlobalRoutePlanner
from pettingzoo.utils.env import AgentID, ActionType, ObsType
from scenic.core.object_types import Point, OrientedPoint
from scenic.core.scenarios import Scene, Scenario
from scenic.core.vectors import Vector
from scenic.domains.driving.roads import Network, Lane, VehicleType, LaneGroup, Road
from scenic.formats.opendrive import xodr_parser
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from scenic.simulators.carla.utils.utils import scenicToCarlaLocation, carlaToScenicPosition

from mats_gym.envs.scenario_env_wrapper import BaseScenarioEnvWrapper

from cat.advgen.modeling.vectornet import VectorNet
from mats_trafficgen.map_features import MapFeatures
from mats_trafficgen.trajectory_following import TrajectoryFollowingAgent
from mats_trafficgen.trajectory_sampler import TrajectorySampler
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


@dataclasses.dataclass
class Candidate:
    scenario: dict
    ego_route: list[tuple[carla.Transform, carla.Vector3D]]
    npc_trajectories: dict[int, list[tuple[carla.Transform, carla.Vector3D]]]


class ScenarioOptimizationWrapper(BaseScenarioEnvWrapper):
    """
    Wrapper to add road information to the observation.
    Road information includes:
    - Identification of the current lane (road, section, lane)
    - Lane type and width
    - Lane change possibility
    """

    def __init__(
            self,
            args: Any,
            env: BaseScenarioEnvWrapper,
            ego_agent: AgentID,
            scenario: str,
            traffic_model: VectorNet,
            towns: list[str] | str = "Town05",
            num_scenarios_per_town: int = 2,
            max_samples: int = 20000,
            max_radius: float = 50.0,
            line_resolution: float = 0.5,
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
        self._traffic_model = traffic_model
        self._ego_agent = ego_agent

        self._map_features = MapFeatures(
            line_resolution=self._line_resolution,
            debug=True
        )
        self._sampler = TrajectorySampler(
            traffic_model=traffic_model,
            args=args,
            map_features=self._map_features,
        )

        self._population = []
        candidates = self._generate_initial_scenarios(
            scenario=scenario,
            towns=towns if isinstance(towns, list) else [towns],
            num_scenarios_per_town=num_scenarios_per_town
        )
        self._evaluation_queue = deque(candidates)
        self._evaluation_scenario = None
        super().__init__(env)

    def step(
            self, actions: dict[AgentID, ActionType]
    ) -> tuple[
        dict[AgentID, ObsType],
        dict[AgentID, float],
        dict[AgentID, bool],
        dict[AgentID, bool],
        dict[AgentID, dict],
    ]:
        obs, reward, done, truncated, info = self.env.step(actions)

        current_eval = self._evaluation_scenario
        for agent in self.actors:
            actor = self.actors[agent]
            if agent == self._ego_agent:
                current_eval.ego_route.append((actor.get_transform(), actor.get_velocity()))
            else:
                current_eval.npc_trajectories[agent].append((actor.get_transform(), actor.get_velocity()))
        return obs, reward, done, truncated, info

    def _place_vehicles(self, num_vehicles: int, radius: float) -> list[carla.Actor]:
        map = CarlaDataProvider.get_map()
        actor = self.actors[self._ego_agent]
        spawn_points = [tf for tf in map.get_spawn_points() if tf.location.distance(actor.get_location()) < radius]
        spawn_points = random.sample(spawn_points, num_vehicles)
        vehicles = []
        for i, spawn_point in enumerate(spawn_points):
            model = random.choice(["vehicle.audi.a2", "vehicle.bmw.grandtourer", "vehicle.tesla.model3"])
            vehicle = CarlaDataProvider.request_new_actor(
                model=model,
                spawn_point=spawn_point,
                rolename=f"npc_{i}"
            )
            vehicles.append(vehicle)
        return vehicles

    def _generate_initial_scenarios(self, scenario: str, towns: list[str] | str, num_scenarios_per_town: int = 10) -> list[Candidate]:
        scenarios = []
        for town in towns:
            params = {
                "map": f"scenarios/maps/{town}.xodr",
                "carla_map": town
            }
            with open(scenario, "r") as f:
                code = f.read()
            scenario: Scenario = scenic.scenarioFromString(
                string=code,
                params=params
            )
            for i in range(num_scenarios_per_town):
                scene, _ = scenario.generate()
                candidate = Candidate(
                    scenario={
                        "binary": scenario.sceneToBytes(scene),
                        "code": code,
                        "params": params
                    },
                    ego_route=[],
                    npc_trajectories=[]
                )
                scenarios.append(candidate)
        return scenarios

    def _sample_route(self, actor: carla.Actor, max_distance: float, resolution: float = 5) -> list[carla.Waypoint]:
        map = CarlaDataProvider.get_map()
        start = map.get_waypoint(actor.get_location(), project_to_road=True)
        route = [start]
        current_w = start
        last_lane_change = 6
        while route[-1].transform.location.distance(start.transform.location) < max_distance:
            # list of potential next waypoints
            potential_w = list(current_w.next(resolution))
            num_successors = len(potential_w)

            # check for available right driving lanes
            if current_w.lane_change & carla.LaneChange.Right and last_lane_change > 5:
                right_w = current_w.get_right_lane()
                if right_w and right_w.lane_type == carla.LaneType.Driving:
                    potential_w += list(right_w.next(resolution))

            # check for available left driving lanes
            if current_w.lane_change & carla.LaneChange.Left and last_lane_change > 5:
                left_w = current_w.get_left_lane()
                if left_w and left_w.lane_type == carla.LaneType.Driving:
                    potential_w += list(left_w.next(resolution))

            # choose a random waypoint to be the next
            idx = random.randint(0, len(potential_w) - 1)
            next_w = potential_w[idx]
            if idx < num_successors:
                last_lane_change += 1
            else:
                last_lane_change = 0
            current_w = next_w
            route.append(next_w)
        return route

    def reset(
            self, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict, dict[Any, dict]]:

        options = options or {}

        if self._evaluation_scenario is not None:
            pass


        if len(self._evaluation_queue) > 0:
            logging.debug(f"Evaluation queue not empty. Using next scenario.")
            scenario = self._evaluation_queue.popleft()
            options["scene"] = scenario.scenario
            obs, info = self.env.reset(seed, options)
            route = self._sample_route(
                actor=self.actors[self._ego_agent],
                max_distance=100,
                resolution=5
            )
            planner = GlobalRoutePlanner(CarlaDataProvider.get_map(), sampling_resolution=0.5)
            route = planner.trace_route(
                origin=route[0].transform.location,
                destination=route[-1].transform.location
            )
            for i in range(len(route) - 1):
                tf, next_tf = route[i][0].transform, route[i + 1][0].transform
                scenario.ego_route.append((tf, next_tf.location - tf.location))
            info[self._ego_agent]["route"] = route
            actor = self.actors[self._ego_agent]
            self._evaluation_scenario = scenario

        else:
            logging.debug(f"Evaluation queue empty. Generating new scenarios.")

            scenario = self._generate_scenario()
            options["scene"] = scenario


        # Get the map and reload the map features if the map has changed
        map = CarlaDataProvider.get_map()
        if self._map_features.town is None or self._map_features.town != map.name:
            self._map_features.reload_map(map)

        ego_actor = self.actors[self._ego_agent]
        trajectories, scores, goals = self._sampler.sample(
            world=CarlaDataProvider.get_world(),
            actor=ego_actor,
            trajectories={
                ego_actor.id: scenario.ego_route
            }
        )
        ego_goal = scenario.ego_route[-1][0].location
        ego_goal = np.array([ego_goal.y, ego_goal.x])
        closest_to_goal = np.argsort(np.linalg.norm(trajectories[:, -1] - ego_goal, axis=-1))[:10]
        trajectories = trajectories[closest_to_goal]
        world = CarlaDataProvider.get_world()

        for traj in trajectories:
            for start, end in zip(traj[:-1], traj[1:]):
                start = carla.Location(start[1], start[0], 0.2)
                end = carla.Location(end[1], end[0], 0.2)
                world.debug.draw_line(
                    start,
                    end,
                    thickness=0.05,
                    color=carla.Color(0, 0, 5),
                    life_time=1000
                )
        world.tick()

        return obs, info
