import argparse
import logging
import pickle
import random

import carla
import mats_gym
import numpy as np
import scenic
from agents.navigation.basic_agent import BasicAgent
from mats_gym.envs.renderers import camera_pov
from scenic.core.scenarios import Scenario
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from mats_trafficgen.cat_wrapper import AdversarialTrainingWrapper
from cat.advgen.adv_generator import AdvGenerator
from mats_trafficgen.level_generator import LevelGenerator
from mats_trafficgen.trajectory_following import TrajectoryFollowingAgent

"""
This example shows how to use the CarlaVisualizationWrapper to create visualizations
inside the CARLA simulator. The visualization is done by adding a callback to the wrapper.
"""

NUM_EPISODES = 10

class FakeWaypoint:

    def __init__(self, transform):
        self.transform = transform
        map = CarlaDataProvider.get_map()
        self.wp = map.get_waypoint(transform.location)

    def __getattr__(self, item):
        if item == "transform":
            return self.transform
        return getattr(self.wp, item)


def get_agents(env):
    controllers = {}
    map = CarlaDataProvider.get_map()
    for i, agent in enumerate(env.agents):
        route = env.current_scenario.config.ego_vehicles[i].route
        controller = BasicAgent(
            env.actors[agent],
            target_speed=20.0,
            opt_dict={
                "use_bbs_detection": True,
                "base_vehicle_threshold": 15.0,
                "ignore_traffic_lights": True
            }
        )
        plan = []
        for tf, option in route:
            wp = FakeWaypoint(tf)
            plan.append((wp, option))
        controller.set_global_plan(plan)
        controllers[agent] = controller
    return controllers

def main(args):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(filename)s - [%(levelname)s] - %(message)s",
    )

    SEED = 226
    random.seed(SEED)
    np.random.seed(SEED)
    scenario: Scenario = scenic.scenarioFromFile("scenarios/scenic/one_adv_intersection.scenic")
    scene, _ = scenario.generate()
    env = mats_gym.scenic_env(
        host=args.carla_host,
        port=args.carla_port,
        seed=SEED,
        agent_name_prefixes=["ego", "adversary"],
        scenes=scene,
        render_mode="human",
        render_config=camera_pov(agent="ego"),
        max_time_steps=200
    )

    env = AdversarialTrainingWrapper(
        env=env,
        args=args,
        model_path="cat/advgen/pretrained/densetnt.bin",
        ego_agent="ego",
        adv_agents="adversary",
    )

    def joint_policy(agents):
        actions = {}
        for agent in agents:
            ctrl = agents[agent].run_step()
            actions[agent] = np.array([ctrl.throttle, ctrl.steer, ctrl.brake])
        return actions

    for e in range(NUM_EPISODES):
        obs, info = env.reset(options={
            "scene": scene
        })
        agents = get_agents(env)
        done = False
        while not done:
            actions = joint_policy(agents)
            obs, reward, done, truncated, info = env.step(actions)
            done = all(done.values())
            env.render()

        obs, info = env.reset(options={
            "scene": scene,
            "adversarial": True
        })

        traj = [
            (carla.Location(x=point[0], y=point[1]), point[2] * 3.6)
            for point in info["adversary"]["adv_trajectory"]
        ]
        adv_agent = TrajectoryFollowingAgent(
            vehicle=env.actors["adversary"],
            trajectory=traj
        )
        agents = get_agents(env)
        agents["adversary"] = adv_agent

        done = False
        while not done:
            actions = joint_policy(agents)
            obs, reward, done, truncated, info = env.step(actions)
            done = all(done.values())
            env.render()

        scene, _ = scenario.generate()
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--OV_traj_num', type=int, default=256)
    parser.add_argument('--AV_traj_num', type=int, default=1)
    parser.add_argument('--carla-host', type=str, default="localhost")
    parser.add_argument('--carla-port', type=int, default=2000)
    gen = AdvGenerator(parser, pretrained_path="./cat/advgen/pretrained/densetnt.bin")
    main(gen.args)
