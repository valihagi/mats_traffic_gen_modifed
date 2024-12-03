import argparse
import logging
import random

import mats_gym
import numpy as np
import scenic
from agents.navigation.basic_agent import BasicAgent
from mats_gym.envs.renderers import camera_pov
from mats_gym.wrappers import MetaActionWrapper
from scenic.core.scenarios import Scenario
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

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
        controller = BasicAgent(env.actors[agent], target_speed=40.0, opt_dict={"ignore_traffic_lights": True})
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

    env = mats_gym.scenic_env(
        host="localhost",
        port=2000,
        agent_name_prefixes=["ego", "adversary"],
        scenario_specification="scenarios/scenic/one_adv_intersection.scenic",
        render_mode="human",
        render_config=camera_pov(agent="ego")
    )

    from mats_trafficgen.trafficgen_wrapper import TrafficGenWrapper
    env = TrafficGenWrapper(
        env=env
    )

    def joint_policy(agents):
        actions = {}
        for agent in agents:
            ctrl = agents[agent].run_step()
            actions[agent] = np.array([ctrl.throttle, ctrl.steer, ctrl.brake]) * 0
        return actions

    for e in range(NUM_EPISODES):
        obs, info = env.reset()

        agents = get_agents(env)
        done = False
        while not done:
            actions = joint_policy(agents)
            obs, reward, done, truncated, info = env.step(actions)
            done = all(done.values())
            env.render()
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    main(parser.parse_args())
