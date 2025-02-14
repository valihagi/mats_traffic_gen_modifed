import argparse
import json
import logging
import pickle
import random
import os

import carla
import time
from active_doe_module.webapi_client import active_doe_client
import mats_gym
import numpy as np
from run_simulation import run_simulation
import scenic
from agents.navigation.basic_agent import BasicAgent
from mats_gym.envs.renderers import camera_pov
from scenic.core.scenarios import Scenario
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from mats_trafficgen.cat_wrapper import AdversarialTrainingWrapper
from cat.advgen.adv_generator import AdvGenerator
from mats_trafficgen.level_generator import LevelGenerator
from mats_trafficgen.trajectory_following import TrajectoryFollowingAgent
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
import rclpy
from std_msgs.msg import String
from time import sleep
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import Header
from pose_publisher import PosePublisher
from helpers import run_docker_command
from helpers import get_docker_ouptut
from helpers import run_docker_restart_command
from helpers import get_carla_point_from_scene
#from motion_state_subscriber import MotionStateSubscriber
import signal
from pprint import pprint
import math
from tqdm import tqdm
from scipy.stats import wasserstein_distance

import subprocess

autoware_container_name = "pensive_curie"
bridge_container_name = "pensive_hugle"

autoware_terminal = "/dev/pts/14"
bridge_terminal = "/dev/pts/15"
default_terminal = "/dev/pts/20"


"""
This example shows how to use the CarlaVisualizationWrapper to create visualizations
inside the CARLA simulator. The visualization is done by adding a callback to the wrapper.
"""

NUM_EPISODES = 10



def compute_WD(gt, other):
    gt_histogram, gt_bins = np.histogram(gt, bins=int(np.ceil(np.sqrt(len(gt)))))
    gt_histogram = gt_histogram + .1
    gt_histogram /= np.sum(gt_histogram)

    other_histogram, _ = np.histogram(other, bins=gt_bins)
    other_histogram = other_histogram + .1
    other_histogram /= np.sum(other_histogram)

    wd = wasserstein_distance(u_values=gt_bins[:-1], v_values=gt_bins[:-1],
                                u_weights=gt_histogram, v_weights=other_histogram)
    return wd
    

def main(args):
    rclpy.init()
    pose_publisher = PosePublisher()
    SEED = 226
    random.seed(SEED)
    np.random.seed(SEED)
    strategy = args.strategy

    # set scene to None if we have an xosc scenario
    scenario: Scenario = scenic.scenarioFromFile("scenarios/scenic/one_adv_intersection.scenic")
    scene, _ = scenario.generate()
    """for i in range(10):
        scene, _ = scenario.generate()
    
        pose_publisher.convert_from_carla_to_autoware(get_carla_point_from_scene(scene))
        print(pose_publisher.pose_msg)
    return"""
    
    """for x in scene.egoObject.route[-1].centerline:
        print(x)
        
    while(1):
        time.sleep(1)"""

    env = mats_gym.scenic_env(
        host=args.carla_host,
        port=args.carla_port,
        seed=SEED,
        agent_name_prefixes=["ego_vehicle", "adversary"],
        scenes=scene,
        render_mode="human",
        render_config=camera_pov(agent="ego_vehicle"),
        max_time_steps=2000
    )

    env = AdversarialTrainingWrapper(
        env=env,
        args=args,
        model_path="cat/advgen/pretrained/densetnt.bin",
        ego_agent="ego_vehicle",
        adv_agents="adversary",
    )
    
    if strategy == "doe":
        with open("active_doe_module/setup_scenic.json", "r") as fp:
            setup = json.load(fp)
    
        with active_doe_client(hostname="localhost", port=8011, use_sg=False) as doe_client:
            session=doe_client.initialize(setup=setup)
            if session is None:
                raise Exception("could not initialize session")
            measurements = [
                dict(
                    Variations={"adv_target_speed": 50},
                    Responses={"ttc": 1.0}
                )
            ]
            #client.insert_measurements(measurements=measurements)
            while True:

                candidates=doe_client.get_candidates(size=1, latest_models_required=True)
                print(candidates)
                measurements = []

                for candidate in candidates:
                    ##unpack candiates and insert them into env.scenario
                    env.parameters = candidate["Variations"]
                    obs, info = env.reset(options={
                        "scene": scene,
                        "parametrized": True
                        })

                    traj = [
                        (carla.Location(x=point[0], y=point[1]), point[2] * 3.6)
                        for point in info["adversary"]["adv_trajectory"]
                    ]

                    run_simulation(autoware_container_name=autoware_container_name,
                       bridge_container_name=bridge_container_name,
                       default_terminal=default_terminal,
                       autoware_terminal=autoware_terminal,
                       bridge_terminal=bridge_terminal,
                       env=env,
                       args=args,
                       scene=scene,
                       target_point=None,
                       strategy=strategy,
                       adv_path=traj,
                       pose_publisher=pose_publisher)
            return



    for e in range(NUM_EPISODES):
        traj = None
        if strategy == "random":
            obs, info = env.reset(options={
                "scene": scene,
                "random": True
            })
            traj = [
                (carla.Location(x=point[0], y=point[1]), point[2] * 3.6)
                for point in info["adversary"]["adv_trajectory"]
            ]
        elif strategy == "cat":
            if e != 0:
                scene, _ = scenario.generate()
            obs, info = env.reset(options={
                "scene": scene
            })

        run_simulation(autoware_container_name=autoware_container_name,
                       bridge_container_name=bridge_container_name,
                       default_terminal=default_terminal,
                       autoware_terminal=autoware_terminal,
                       bridge_terminal=bridge_terminal,
                       env=env,
                       args=args,
                       scene=scene,
                       target_point=None,
                       strategy=strategy,
                       adv_path=traj,
                       pose_publisher=pose_publisher)
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--OV_traj_num', type=int, default=256)
    parser.add_argument('--AV_traj_num', type=int, default=1)
    parser.add_argument('--carla-host', type=str, default="localhost")
    parser.add_argument('--carla-port', type=int, default=2000)
    parser.add_argument('--strategy', type=str, default="cat")
    gen = AdvGenerator(parser, pretrained_path="./cat/advgen/pretrained/densetnt.bin")
    main(gen.args)
