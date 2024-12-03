from __future__ import annotations

import argparse
import logging
import os
import random

import carla
import numpy as np
import torch
from agents.navigation.basic_agent import BasicAgent
from srunner.autoagents.autonomous_agent import AutonomousAgent
from srunner.scenarioconfigs.route_scenario_configuration import (
    RouteScenarioConfiguration,
)
from srunner.scenarioconfigs.scenario_configuration import ScenarioConfiguration
from srunner.scenariomanager.timer import GameTime
from srunner.scenarios.route_scenario import RouteScenario
from srunner.tools.route_parser import RouteParser

import mats_gym
from mats_gym.envs import renderers
from mats_gym.scenarios.actor_configuration import ActorConfiguration

from adversarial_route_wrapper import AdversarialRouteWrapper
from cat.advgen.adv_generator import AdvGenerator
from cat.advgen.modeling.vectornet import VectorNet

"""
This example shows how run a leaderboard route scenario with a custom agent.
"""


def get_policy_for_agent(agent: BasicAgent):
    def policy(obs):
        control = agent.run_step()
        action = np.array([control.throttle, control.steer, control.brake])
        return action

    return policy


def scenario_fn(client: carla.Client, config: ScenarioConfiguration):
    scenario = RouteScenario(world=client.get_world(), config=config, debug_mode=1)
    return scenario


def make_configs(role_name: str, routes: str) -> list[RouteScenarioConfiguration]:
    configs = RouteParser.parse_routes_file(route_filename=routes)
    for config in configs:
        config.ego_vehicles = [
            ActorConfiguration(
                route=config.route,
                model="vehicle.lincoln.mkz2017",
                rolename=role_name,
                transform=None,
            )
        ]
    return configs


def main(args):
    # Set environment variable for the scenario runner root. It can be found in the virtual environment.
    os.environ["SCENARIO_RUNNER_ROOT"] = os.path.join(
        os.getcwd(), "../venv/lib/python3.10/site-packages"
    )
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(filename)s - [%(levelname)s] - %(message)s",
    )


    configs = make_configs(role_name="hero", routes="data/routes_debug.xml")
    env = mats_gym.raw_env(
        config=configs[0],
        scenario_fn=scenario_fn,
        render_mode="human",
        render_config=renderers.camera_pov(agent="hero"),
    )

    model = VectorNet(args).to("cpu")
    model.load_state_dict(torch.load("../cat/advgen/pretrained/densetnt.bin", "cpu"))
    env = AdversarialRouteWrapper(env, args=args, model=model)
    client = carla.Client("localhost", 2000)
    client.set_timeout(120.0)


    for config in configs[0:]:
        obs, info = env.reset(options={"scenario_config": config, "client": client})
        agent = BasicAgent(vehicle=env.actors["hero"])
        policy = get_policy_for_agent(agent)
        done = False
        while not done:
            # Use agent to get control for the current step
            actions = {agent: policy(o) for agent, o in obs.items()}
            obs, reward, done, truncated, info = env.step(actions)
            done = done["hero"]
            env.render()
            print("EVENTS: ", info["hero"]["events"])

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--OV_traj_num', type=int, default=256)
    parser.add_argument('--AV_traj_num', type=int, default=1)
    gen = AdvGenerator(parser, pretrained_path="../cat/advgen/pretrained/densetnt.bin")
    main(gen.args)
