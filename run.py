import argparse
import logging
import math
import pickle
import random

import carla
import numpy as np
import scenic
import torch
from agents.navigation.basic_agent import BasicAgent
from agents.navigation.local_planner import RoadOption
from mats_gym.agents.meta_actions_agent import Action
from mats_gym.envs.renderers import camera_pov
from scenic.core.scenarios import Scenario
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenarios.basic_scenario import BasicScenario

import mats_gym
from mats_gym.envs import renderers
from mats_gym.wrappers import CarlaVisualizationWrapper, MetaActionWrapper, ReplayWrapper

from cat import advgen
from cat.advgen.adv_generator import AdvGenerator
from cat.advgen.modeling.vectornet import VectorNet
from cat.advgen.utils import add_argument
from mats_trafficgen.level_generator import LevelGenerator
from mats_trafficgen.scenario_optimization_wrapper import ScenarioOptimizationWrapper
from mats_trafficgen.trajectory_following import TrajectoryFollowingAgent

"""
This example shows how to use the CarlaVisualizationWrapper to create visualizations
inside the CARLA simulator. The visualization is done by adding a callback to the wrapper.
"""

NUM_EPISODES = 10


def make_agent(actor: carla.Actor, route: list[carla.Waypoint]):
    agent = BasicAgent(
        vehicle=actor,
        target_speed=35,
        opt_dict={
            "target_speed": 35,
            "ignore_traffic_lights": True
        },
    )
    world = CarlaDataProvider.get_world()
    for (a, _), (b, _) in zip(route[:-1], route[1:]):
        a = carla.Location(a.transform.location.x, a.transform.location.y, a.transform.location.z + 0.2)
        b = carla.Location(b.transform.location.x, b.transform.location.y, b.transform.location.z + 0.2)
        world.debug.draw_line(a, b, thickness=0.1, color=carla.Color(0, 5, 0), life_time=1000)
    agent.set_global_plan(route)
    return agent

def load_traffic_model(args, path: str):
    model = VectorNet(args).to("cpu")
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model

def main(args):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(filename)s - [%(levelname)s] - %(message)s",
    )

    # generator = LevelGenerator("scenarios/scenic/four_way_route_scenario.scenic")
    # config = generator()
    # import pickle

    # with open("level.pkl", "wb") as f:
    #    pickle.dump(config, f)

    SEED = 226
    NUM_SCENES_PER_TOWN = 1
    random.seed(SEED)
    np.random.seed(SEED)

    model = load_traffic_model(args, path="./cat/advgen/pretrained/densetnt.bin")
    env = mats_gym.scenic_env(
        scenario_specification=args.scenario,
        host="localhost",
        port=2000,
        seed=SEED,
        agent_name_prefixes=["ego"],
        render_mode="human",
        render_config=camera_pov(agent="ego"),
        max_time_steps=200
    )

    env = ScenarioOptimizationWrapper(
        args=args,
        env=env,
        scenario=args.scenario,
        num_scenarios_per_town=NUM_SCENES_PER_TOWN,
        ego_agent="ego",
        traffic_model=model,
    )

    for e in range(NUM_EPISODES):
        obs, info = env.reset()
        route = info["ego"]["route"][1:]
        agent = make_agent(actor=env.actors["ego"], route=route)
        done = False
        while not done:
            ctrl = agent.run_step()
            actions = {
                "ego": np.array([ctrl.throttle, ctrl.steer, ctrl.brake])
            }
            obs, reward, done, truncated, info = env.step(actions)
            done = all(done.values())
            env.render()
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=str, default="scenarios/base_scenarios/base.scenic")
    parser.add_argument('--towns', type=str, nargs='+', default=["Town05"])
    parser.add_argument('--OV_traj_num', type=int, default=64)
    parser.add_argument('--AV_traj_num', type=int, default=1)
    gen = AdvGenerator(parser, pretrained_path="./cat/advgen/pretrained/densetnt.bin")
    main(gen.args)
