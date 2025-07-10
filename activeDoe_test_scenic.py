import argparse
import logging
import pickle
import random
import os

import json, time, os,sys
from requests.sessions import session
from active_doe_module.webapi_client import active_doe_client

import carla
import time
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
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
import rclpy
from std_msgs.msg import String
from time import sleep
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import Header
from pose_publisher import PosePublisher
from helpers import get_carla_point_from_xml, run_docker_command
from helpers import get_docker_ouptut
from helpers import run_docker_restart_command
from helpers import get_carla_point_from_scene
#from motion_state_subscriber import MotionStateSubscriber
import signal
from pprint import pprint
import math
from tqdm import tqdm

import subprocess

autoware_container_name = "bold_kapitsa"
bridge_container_name = "gracious_kowalevski"

autoware_terminal = "/dev/pts/14"
bridge_terminal = "/dev/pts/15"
default_terminal = "/dev/pts/20"


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
    
    
def kill_process(container_name, pid):
    command = (
        f"kill {pid}"
    )
    return run_docker_command(container_name, command, default_terminal)
    
    
    
def change_control_mode(container_name):
    command = (
        "cd /work/Valentin_dev/tumgeka_bridge/autoware_new/autoware/ && "
        "source install/setup.bash && "
        "ros2 service call /api/operation_mode/change_to_autonomous autoware_adapi_v1_msgs/srv/ChangeOperationMode {}"
    )
    return run_docker_command(container_name, command, default_terminal)
    
    
def check_is_stopped(container_name):
    command = (
        "cd /work/Valentin_dev/tumgeka_bridge/autoware_new/autoware/ && "
        "source install/setup.bash && "
        "ros2 service call /api/operation_mode/change_to_autonomous autoware_adapi_v1_msgs/srv/ChangeOperationMode {}"
    )
    return run_docker_command(container_name, command, default_terminal)
    
    
def run_autoware_simulation(container_name):
    command = (
        "cd /work/Valentin_dev/tumgeka_bridge/autoware_new/autoware/ && "
        "source install/setup.bash && "
        "ros2 launch autoware_launch e2e_simulator.launch.xml "
        "vehicle_model:=carla_t2_vehicle sensor_model:=carla_t2_sensor_kit "
        "map_path:=/work/Valentin_dev/tumgeka_bridge/Town10 launch_system:=false"
    )
    return run_docker_command(container_name, command, autoware_terminal)
    
    
def run_carla_aw_bridge(container_name):
    command = (
        "source install/setup.bash && "
        "ros2 launch carla_autoware_bridge carla_aw_bridge.launch.py "
        "port:=2000 passive:=True register_all_sensors:=False timeout:=180"
    )
    return run_docker_command(container_name, command, bridge_terminal)
    

def main(args):
    rclpy.init()
    pose_publisher = PosePublisher()
    SEED = 226
    random.seed(SEED)
    np.random.seed(SEED)
    """for i in range(10):
        scene, _ = scenario.generate()
    
        pose_publisher.convert_from_carla_to_autoware(get_carla_point_from_scene(scene))
        print(pose_publisher.pose_msg)
    return"""
    
    """for x in scene.egoObject.route[-1].centerline:
        print(x)
        
    while(1):
        time.sleep(1)"""
    scenario: Scenario = scenic.scenarioFromFile("scenarios/scenic/one_adv_intersection.scenic")
    scene, _ = scenario.generate()

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

    def joint_policy(agents):
        actions = {}
        for agent in agents:
            if agent != "ego_vehicle":
                ctrl = agents[agent].run_step()
                actions[agent] = np.array([ctrl.throttle, ctrl.steer, ctrl.brake])
                #actions[agent] = np.array([.50, 0, 0])
        return actions
    
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

                aw_process = run_autoware_simulation(autoware_container_name)
                obs, info = env.reset(options={
                "scene": scene,
                "parametrized": True
                })

                agents = get_agents(env)
                traj = [
                    (carla.Location(x=point[0], y=point[1]), point[2] * 3.6)
                    for point in info["adversary"]["adv_trajectory"]
                ]

                adv_agent = TrajectoryFollowingAgent(
                    vehicle=env.actors["adversary"],
                    trajectory=traj
                )
                agents["adversary"] = adv_agent
                        
                done = False
                CarlaDataProvider.get_world().tick()
                
                print("\n starting up...")
                for i in tqdm(range(100)):
                    CarlaDataProvider.get_world().tick()
                    time.sleep(.1)
                
                carla_aw_bridge_process = run_carla_aw_bridge(bridge_container_name)
                
                
                print("\n waiting for autoware...")
                for i in tqdm(range(550)):
                    time.sleep(.1)
                    CarlaDataProvider.get_world().tick()
                #motion_state_subscriber = MotionStateSubscriber(CarlaDataProvider.get_world())
                # Wait until the vehicle is stopped
                #motion_state_subscriber.wait_until_stopped()
                
                pose_publisher.convert_from_carla_to_autoware(get_carla_point_from_scene(scene))

                try:
                    rclpy.spin_once(pose_publisher, timeout_sec=2)
                except KeyboardInterrupt:
                    pass
                
                print("\n watiting for autonomous mode....")
                for i in tqdm(range(120)):
                    time.sleep(.1)
                    CarlaDataProvider.get_world().tick()
                    
                control_change_process = change_control_mode(autoware_container_name)
                
                print("\n starting autoware...")
                for i in tqdm(range(10)):
                    time.sleep(.1)
                    CarlaDataProvider.get_world().tick()
                
                while not done:
                    actions = joint_policy(agents)
                    obs, reward, done, truncated, info = env.step(actions)
                    time.sleep(.02)
                    done = all(done.values())
                    env.render()

                #retrieve all kpis and feed them to activeDoE
                kpis = env.get_ttc_as_dict()
                measurements.append(dict(
                    Index=candidate['Index'],
                    Variations=candidate['Variations'],
                    Responses=kpis
                    )
                )
                    
                print("----------------------")
                print("restarting containers")
                
                run_docker_restart_command(bridge_container_name, default_terminal)
                run_docker_restart_command(autoware_container_name, default_terminal)
            doe_client.insert_measurements(measurements=measurements)
        
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--OV_traj_num', type=int, default=64)
    parser.add_argument('--AV_traj_num', type=int, default=1)
    parser.add_argument('--carla-host', type=str, default="localhost")
    parser.add_argument('--carla-port', type=int, default=2000)
    gen = AdvGenerator(parser, pretrained_path="./cat/advgen/pretrained/densetnt.bin")
    main(gen.args)
