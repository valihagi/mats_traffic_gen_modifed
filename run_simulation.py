import argparse
import logging
import pickle
import random
import os
import subprocess

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
from active_doe_module.webapi_client import active_doe_client


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
    
    
def kill_process(container_name, pid, default_terminal):
    command = (
        f"kill {pid}"
    )
    return run_docker_command(container_name, command, default_terminal)
    
    
    
def change_control_mode(container_name, default_terminal):
    command = (
        "cd /home/u29r26/developing/autoware && "
        "source install/setup.bash && "
        "ros2 service call /api/operation_mode/change_to_autonomous autoware_adapi_v1_msgs/srv/ChangeOperationMode {}"
    )
    return run_docker_command(container_name, command, default_terminal)
    
    
def check_is_stopped(container_name, default_terminal):
    command = (
        "cd /home/u29r26/developing/autoware && "
        "source install/setup.bash && "
        "ros2 service call /api/operation_mode/change_to_autonomous autoware_adapi_v1_msgs/srv/ChangeOperationMode {}"
    )
    return run_docker_command(container_name, command, default_terminal)
    
    
def run_autoware_simulation(container_name, autoware_terminal):
    command = (
        "cd /home/u29r26/developing/autoware && "
        "source install/setup.bash && "
        "ros2 launch autoware_launch e2e_simulator.launch.xml "
        "vehicle_model:=carla_t2_vehicle sensor_model:=carla_t2_sensor_kit "
        "map_path:=/home/u29r26/developing/Town10 launch_system:=false"
    )
    return run_docker_command(container_name, command, autoware_terminal)
    
    
def run_carla_aw_bridge(container_name, bridge_terminal):
    command = (
        "source install/setup.bash && "
        "ros2 launch carla_autoware_bridge carla_aw_bridge.launch.py "
        "port:=2000 passive:=True register_all_sensors:=False timeout:=180"
    )
    return run_docker_command(container_name, command, bridge_terminal)

def joint_policy(agents):
        actions = {}
        for agent in agents:
            if agent != "ego_vehicle":
                ctrl = agents[agent].run_step()
                actions[agent] = np.array([ctrl.throttle, ctrl.steer, ctrl.brake])
                #actions[agent] = np.array([.50, 0, 0])
            else:
                actions[agent] = np.array([.50, 0, 0])
        return actions

def run_simulation(autoware_container_name, bridge_container_name, default_terminal, autoware_terminal,
                   bridge_terminal, env, args, scene, target_point, strategy, adv_path, pose_publisher):
    aw_process = run_autoware_simulation(autoware_container_name, autoware_terminal)
        
    if scene is not None:
        agents = get_agents(env)
    else:
        agents = {}
        agents["ego_vehicle"] = {"ego_vehicle"}
    if adv_path is not None:
        adv_agent = TrajectoryFollowingAgent(
            vehicle=env.actors["adversary"],
            trajectory=adv_path
        )
        agents["adversary"] = adv_agent

    client = carla.Client(args.carla_host, args.carla_port)
    
    done = False
    CarlaDataProvider.get_world().tick()
    
    print("\n starting up...")
    for i in tqdm(range(100)):
        CarlaDataProvider.get_world().tick()
        time.sleep(.1)
    
    carla_aw_bridge_process = run_carla_aw_bridge(bridge_container_name, bridge_terminal)
    
    
    print("\n waiting for autoware...")
    for i in tqdm(range(520)):
        time.sleep(.1)
        CarlaDataProvider.get_world().tick()
    #motion_state_subscriber = MotionStateSubscriber(CarlaDataProvider.get_world())
    # Wait until the vehicle is stopped
    #motion_state_subscriber.wait_until_stopped()
    if scene is not None:
        pose_publisher.convert_from_carla_to_autoware(get_carla_point_from_scene(scene))
    else:
        pose_publisher.convert_from_carla_to_autoware(target_point)

    try:
        rclpy.spin_once(pose_publisher, timeout_sec=2)
    except KeyboardInterrupt:
        pass
    
    print("\n watiting for autonomous mode....")
    for i in tqdm(range(60)):
        time.sleep(.1)
        CarlaDataProvider.get_world().tick()
        
    control_change_process = change_control_mode(autoware_container_name, default_terminal)
    
    print("\n starting autoware...")
    for i in tqdm(range(200)):
        #TODO implement wait for ego to have specific speed
        vel = env.actors["ego_vehicle"].get_velocity()
        speed = np.linalg.norm([vel.x, vel.y])
        if speed > 2:
            break
        time.sleep(.02)
        CarlaDataProvider.get_world().tick()
    print("-------Starting now:-------------")
    
    #----------------main loop--------------------------------------------
    counter = 0
    collision = False
    while not done:
        counter += 1
        actions = joint_policy(agents)
        obs, reward, done, truncated, info = env.step(actions)
        time.sleep(.02)
        if env.coll:
            collision = True
            break
        done = all(done.values())
        env.render()
        if counter > 350:
            done = True
    #--------------------------------------------------------------------------

    ttc = info["kpis"]["ttc"][-1]

    #print(f"yaw-wasserstein distance: {compute_WD(gt_yaw, other_yaw)}")
    #print(f"acc-wasserstein distance: {compute_WD(gt_acc, other_acc)}")
    print(f"min ttc: {ttc}")
    print(f"collision: {collision}")
        
    print("----------------------")
    print("restarting containers")
    
    run_docker_restart_command(bridge_container_name, default_terminal)
    run_docker_restart_command(autoware_container_name, default_terminal)


    #--------------------return if we are not within a CAT run-----------------------
    if strategy != "cat":
        # calc metrics and return them
        return
    
    aw_process = run_autoware_simulation(autoware_container_name, autoware_terminal)
    try:
        obs, info = env.reset(options={
            "scene": scene,
            "adversarial": True
        })
    except:
        subprocess.run(["bash", "/work/Valentin_dev/mats-trafficgen-main/Carla/CarlaUE4.sh"], check=True)
        obs, info = env.reset(options={
            "scene": scene,
            "adversarial": True
        })

    gt_yaw = info["kpis"]["adv_yaw"]
    gt_acc = info["kpis"]["adv_acc"]

    traj = [
        (carla.Location(x=point[0], y=point[1]), point[2] * 3.6)
        for point in info["adversary"]["adv_trajectory"]
    ]
    adv_agent = TrajectoryFollowingAgent(
        vehicle=env.actors["adversary"],
        trajectory=traj
    )
    if scene is not None:
        agents = get_agents(env)
    else:
        agents = {}
        agents["ego_vehicle"] = {"ego_vehicle"}
    agents["adversary"] = adv_agent
    
    carla_aw_bridge_process = run_carla_aw_bridge(bridge_container_name, bridge_terminal) 

    print("waiting for autoware....")
    for i in tqdm(range(580)):
        time.sleep(.1)
        CarlaDataProvider.get_world().tick()
    
    if scene is not None:
        pose_publisher.convert_from_carla_to_autoware(get_carla_point_from_scene(scene))
    else:
        pose_publisher.convert_from_carla_to_autoware(target_point)

    CarlaDataProvider.get_world().tick()
    try:
            rclpy.spin_once(pose_publisher, timeout_sec=2)
    except KeyboardInterrupt:
        pass
    
    print("waiting for autonomous mode....")
    for i in tqdm(range(60)):
        time.sleep(.1)
        CarlaDataProvider.get_world().tick()
        
    control_change_process = change_control_mode(autoware_container_name, default_terminal)
    
    print("starting autoware....")
    for i in tqdm(range(200)):
        vel = env.actors["ego_vehicle"].get_velocity()
        speed = np.linalg.norm([vel.x, vel.y])
        if speed > 2:
            break
        time.sleep(.02)
        CarlaDataProvider.get_world().tick()
    print("-------Starting now:-------------")

    done = False
    #-----------------------------main loop-----------------------------------
    while not done:
        actions = joint_policy(agents)
        obs, reward, done, truncated, info = env.step(actions)
        if env.coll:
            collision = True
            break
        done = all(done.values())
        env.render()
        time.sleep(.02)
    #---------------------------------------------------------------------------

    other_yaw = info["kpis"]["adv_yaw"]
    other_acc = info["kpis"]["adv_acc"]
    ttc = min(info["kpis"]["ttc"])

    #print(f"yaw-wasserstein distance: {compute_WD(gt_yaw, other_yaw)}")
    #print(f"acc-wasserstein distance: {compute_WD(gt_acc, other_acc)}")
    print(f"min ttc: {ttc}")
    print(f"collision: {collision}")
    
    print("----------------------")
    print("restarting containers")

    run_docker_restart_command(bridge_container_name, default_terminal)
    run_docker_restart_command(autoware_container_name, default_terminal)


def run_dummy_simulation(autoware_container_name, bridge_container_name, default_terminal, autoware_terminal,
                   bridge_terminal, env, args, scene, target_point, strategy, adv_path, pose_publisher):
        
    if scene is not None:
        agents = get_agents(env)
    else:
        agents = {}
        agents["ego_vehicle"] = {"ego_vehicle"}
    if adv_path is not None:
        adv_agent = TrajectoryFollowingAgent(
            vehicle=env.actors["adversary"],
            trajectory=adv_path
        )
        agents["adversary"] = adv_agent

    client = carla.Client(args.carla_host, args.carla_port)
    
    done = False
    
    #----------------main loop--------------------------------------------
    counter = 0
    while not done:
        counter += 1
        actions = joint_policy(agents)
        obs, reward, done, truncated, info = env.step(actions)
        time.sleep(.02)
        if env.coll:
            collision = True
            break
        done = all(done.values())
        env.render()
        if counter > 450:
            done = True
    #--------------------------------------------------------------------------



    #--------------------return if we are not within a CAT run-----------------------
    if strategy != "cat":
        # calc metrics and return them
        return
    

    obs, info = env.reset(options={
        "scene": scene,
        "adversarial": True
    })

    gt_yaw = info["kpis"]["adv_yaw"]
    gt_acc = info["kpis"]["adv_acc"]

    traj = [
        (carla.Location(x=point[0], y=point[1]), point[2] * 3.6)
        for point in info["adversary"]["adv_trajectory"]
    ]
    adv_agent = TrajectoryFollowingAgent(
        vehicle=env.actors["adversary"],
        trajectory=traj
    )
    if scene is not None:
        agents = get_agents(env)
    else:
        agents = {}
        agents["ego_vehicle"] = {"ego_vehicle"}
    agents["adversary"] = adv_agent


    done = False
    #-----------------------------main loop-----------------------------------
    while not done:
        actions = joint_policy(agents)
        obs, reward, done, truncated, info = env.step(actions)
        if env.coll:
            collision = True
            break
        done = all(done.values())
        env.render()
        time.sleep(.02)
    #---------------------------------------------------------------------------

    other_yaw = info["kpis"]["adv_yaw"]
    other_acc = info["kpis"]["adv_acc"]
    #ttc = min(info["kpis"]["ttc"])

    #print(f"yaw-wasserstein distance: {compute_WD(gt_yaw, other_yaw)}")
    #print(f"acc-wasserstein distance: {compute_WD(gt_acc, other_acc)}")
    #print(f"min ttc: {ttc}")
    print(f"collision: {collision}")
    
