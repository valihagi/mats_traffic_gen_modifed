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
from mats_trafficgen.going_straight import GoingStraightAgent
from mats_trafficgen.level_generator import LevelGenerator
from mats_trafficgen.trajectory_following import TrajectoryFollowingAgent
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
import rclpy
from std_msgs.msg import String
from time import sleep
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import Header
import json
from pose_publisher import PosePublisher
from helpers import run_docker_command, save_log_file
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
from enum import IntEnum

class WaitingTime(IntEnum):
    STARTTRIGGERSPEED = 2
    WAITFORAUTONOMOUS = 80
    MAXSTARTDELAY = 250
    MAXTIMESTEPS = 240
    COORDINATECHECK = 45.8
    COORDINATECHECK2 = 23.8
    COORDINATECHECK6 = -38
    COORDINATECHECK7 = 48.3

progress_file = "/workspace/shared/progress.txt"


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
        "cd /work/Valentin_dev/tumgeka_bridge/autoware_fixed/autoware && "
        "source install/setup.bash && "
        "ros2 service call /api/operation_mode/change_to_autonomous autoware_adapi_v1_msgs/srv/ChangeOperationMode {}"
    )
    return run_docker_command(container_name, command, default_terminal)

def init_gnss_again(container_name, default_terminal):
    command = (
        "cd /work/Valentin_dev/tumgeka_bridge/autoware_fixed/autoware && "
        "source install/setup.bash && "
        "ros2 service call /api/localization/initialize autoware_adapi_v1_msgs/srv/InitializeLocalization {}"
    )
    return run_docker_command(container_name, command, default_terminal)
    
    
def check_is_stopped(container_name, default_terminal):
    command = (
        "cd /work/Valentin_dev/tumgeka_bridge/autoware_fixed/autoware && "
        "source install/setup.bash && "
        "ros2 service call /api/operation_mode/change_to_autonomous autoware_adapi_v1_msgs/srv/ChangeOperationMode {}"
    )
    return run_docker_command(container_name, command, default_terminal)
    
    
def run_autoware_simulation(container_name, autoware_terminal):
    command = (
        "cd /work/Valentin_dev/tumgeka_bridge/autoware_fixed/autoware && "
        "source install/setup.bash && "
        "ros2 launch autoware_launch e2e_simulator.launch.xml "
        "vehicle_model:=carla_t2_vehicle sensor_model:=carla_t2_sensor_kit "
        "map_path:=/work/Valentin_dev/tumgeka_bridge/Town10 launch_system:=false"
    )
    # launch_system:=false
    return run_docker_command(container_name, command, autoware_terminal)
    
    
def run_carla_aw_bridge(container_name, bridge_terminal):
    command = (
        "source install/setup.bash && "
        "ros2 launch carla_autoware_bridge carla_aw_bridge.launch.py "
        "port:=2000 passive:=True register_all_sensors:=False timeout:=180"
    )
    return run_docker_command(container_name, command, bridge_terminal)

def joint_policy(agents, counter=None):
        actions = {}
        for agent in agents:
            if counter is not None:
                if agent != "ego_vehicle":
                    print("inhere")
                    if counter > 60:
                        actions[agent] = np.array([.65, 0, 0])
                    else:
                        actions[agent] = np.array([0, 0, 0])
                    return actions
            if agent != "ego_vehicle":
                ctrl = agents[agent].run_step()
                actions[agent] = np.array([ctrl.throttle, ctrl.steer, ctrl.brake])
            """else:
                actions[agent] = np.array([.72, -.09, 0])"""
        return actions

def run_simulation(autoware_container_name, bridge_container_name, carla_container_name, default_terminal, autoware_terminal,
                   bridge_terminal, env, args, scene, target_point, strategy, adv_path, pose_publisher, iteration, autoware_target_point=None, num_iterations=50, parameters=None,
                   test_xosc=None):
    #run_docker_restart_command(carla_container_name, default_terminal)
    """obs, info = env.reset(options={
                "scene": scene
            })"""
    print("sleeping now")
    print("test")
    aw_process = run_autoware_simulation(autoware_container_name, autoware_terminal)
    #run_docker_restart_command(autoware_container_name, default_terminal)
        
    if scene is not None:
        agents = get_agents(env)
    else:
        agents = {}
        agents["ego_vehicle"] = {"ego_vehicle"}
    if adv_path is not None:
        traj = [(carla.Location(x=point[0], y=point[1]), point[2] * 3.6) for point in adv_path]
        adv_agent = TrajectoryFollowingAgent(
            vehicle=env.actors["adversary"],
            trajectory=traj
        )
        agents["adversary"] = adv_agent
    elif strategy == "cat_iterative" or strategy == "cat" or strategy == "cat_no_odd":
        adv_agent = GoingStraightAgent()
        agents["adversary"] = adv_agent
        print("newAGent")

    #client = carla.Client(args.carla_host, args.carla_port)
    
    done = False
    CarlaDataProvider.get_world().tick()
    
    print("\n starting up...")
    for i in tqdm(range(100)):
        CarlaDataProvider.get_world().tick()
        time.sleep(.1)
    
    carla_aw_bridge_process = run_carla_aw_bridge(bridge_container_name, bridge_terminal)
    
    
    print("\n waiting for autoware...")
    for i in tqdm(range(450)):
        time.sleep(.1)
        CarlaDataProvider.get_world().tick()

    """init = init_gnss_again(autoware_container_name, default_terminal)

    print("\n init autoware again...")
    for i in tqdm(range(100)):
        time.sleep(.1)
        CarlaDataProvider.get_world().tick()"""
    #motion_state_subscriber = MotionStateSubscriber(CarlaDataProvider.get_world())
    # Wait until the vehicle is stopped
    #motion_state_subscriber.wait_until_stopped()
    if scene is not None:
        pose_publisher.convert_from_carla_to_autoware(get_carla_point_from_scene(scene))
    else:
        pose_publisher.convert_from_carla_to_autoware(target_point, autoware_target_point)

    try:
        rclpy.spin_once(pose_publisher, timeout_sec=2)
    except KeyboardInterrupt:
        pass
    for actor in CarlaDataProvider.get_world().get_actors():
        print(actor)
    
    print("\n watiting for autonomous mode....")
    for i in tqdm(range(WaitingTime.WAITFORAUTONOMOUS)):
        time.sleep(.1)
        CarlaDataProvider.get_world().tick()
        
    control_change_process = change_control_mode(autoware_container_name, default_terminal)
    
    print("\n starting autoware...")
    """for i in tqdm(range(WaitingTime.MAXSTARTDELAY)):
        #TODO implement wait for ego to have specific speed
        vel = env.actors["ego_vehicle"].get_velocity()
        speed = np.linalg.norm([vel.x, vel.y])
        if speed > WaitingTime.STARTTRIGGERSPEED:
            break
        time.sleep(.01)
        CarlaDataProvider.get_world().tick()"""
    for i in tqdm(range(WaitingTime.MAXSTARTDELAY + 200)):
        #TODO implement wait for ego to have specific speed
        pos = env.actors["ego_vehicle"].get_location()
        #if pos.y > WaitingTime.COORDINATECHECK:
        #if pos.x > WaitingTime.COORDINATECHECK2:
        #if pos.y < WaitingTime.COORDINATECHECK6:
        if pos.y < WaitingTime.COORDINATECHECK7:
            break
        time.sleep(.03)
        CarlaDataProvider.get_world().tick()
    print("-------Starting now:-------------")
    
    #----------------main loop--------------------------------------------
    counter = 0
    collision = False
    while not done:
        if (counter == 30 or counter == 80  or counter == 95 or counter == 105) and strategy == "cat_iterative":
            start = time.time()
            if counter > 50:
                _, adv_traj, ego_traj = env._generate_adversarial_route_iterative(vis=True)
            else:
                _, adv_traj, ego_traj = env._generate_adversarial_route_iterative(vis=True)
            if adv_traj is not None:
                traj = [
                    (carla.Location(x=point[0], y=point[1]), point[2] * 3.6)
                    for point in adv_traj
                ]
                adv_agent = TrajectoryFollowingAgent(
                    vehicle=env.actors["adversary"],
                    trajectory=traj
                )
                agents["adversary"] = adv_agent
            end = time.time()
            print(end - start)
            
        counter += 1
        actions = joint_policy(agents)
        obs, reward, done, truncated, info = env.step(actions)
        time.sleep(.005)
        if env.coll:
            collision = True
            break
        #done = all(done.values())
        env.render()
        if (counter -1 == 30 or counter -1 == 80  or counter -1 == 95 or counter -1 == 105) and strategy == "cat_iterative":
            print("sleeping")
            time.sleep(30)
            
        if counter > WaitingTime.MAXTIMESTEPS:
            done = True
        else:
            done = False
    #--------------------------------------------------------------------------

    #ttc = info["kpis"]["ttc"][-1]

    #print(f"yaw-wasserstein distance: {compute_WD(gt_yaw, other_yaw)}")
    #print(f"acc-wasserstein distance: {compute_WD(gt_acc, other_acc)}")
    #print(f"min ttc: {ttc}")
    #print(f"collision: {collision}")
        
    print("----------------------")
    print("restarting containers")
    
    #if iteration % 8 == 0:
    """print("Also restarting carla this iteartion to prevent it from segfaulting")
    run_docker_restart_command(carla_container_name, default_terminal)
    print("sleeping")
    time.sleep(20)
    print("waking up")"""
    run_docker_restart_command(autoware_container_name, default_terminal)
    run_docker_restart_command(bridge_container_name, default_terminal)


    #--------------------return if we are not within a CAT run-----------------------
    print(strategy)
    if True:
        save_log_file(env, info, parameters, "base_scenario", test_xosc)
    if strategy != "cat" and strategy != "cat_no_odd":
        #check if AW moved:
        ego_traj = env._trajectories["ego_vehicle"]
        started = math.hypot(ego_traj[0]["x"] - ego_traj[-1]["x"], ego_traj[0]["y"] - ego_traj[-1]["y"]) > 2
            
        # calc metrics and return them / also save trajectories that where executed
        if started:
            save_log_file(env, info, parameters, strategy, test_xosc)

        if not started:
            print("AW did not start!!")
            return False
        """if iteration % 8 == 0 and iteration > 0:
            with open(progress_file, "w") as f:
                f.write(str(1))
            print(f"[Main] Progress written: {1}")
            time.sleep(120)"""
        
        return True
    print(num_iterations)
    for iteration in range(num_iterations):
        print(f"Starting ADV scenario iteration {iteration} \n")
        aw_process = run_autoware_simulation(autoware_container_name, autoware_terminal)
        obs, info = env.reset(options={
            "scene": scene,
            "adversarial": True,
            "method": strategy
        })
        print("Reseting the environment")


        #gt_yaw = info["kpis"]["adv_yaw"]
        #gt_acc = info["kpis"]["adv_acc"]
        print("getting_traj")
        if info["adversary"]["adv_trajectory"] is None:
            print("no valid traj exiting...")
            run_docker_restart_command(autoware_container_name, default_terminal)
            run_docker_restart_command(bridge_container_name, default_terminal)
            return
        traj = [
            (carla.Location(x=point[0], y=point[1]), point[2] * 3.6)
            for point in info["adversary"]["adv_trajectory"]
        ]
        print("gottraj")
        adv_agent = TrajectoryFollowingAgent(
            vehicle=env.actors["adversary"],
            trajectory=traj
        )
        print("getting agents")
        if scene is not None:
            agents = get_agents(env)
        else:
            agents = {}
            agents["ego_vehicle"] = {"ego_vehicle"}
        agents["adversary"] = adv_agent
        print("got agents")
        
        carla_aw_bridge_process = run_carla_aw_bridge(bridge_container_name, bridge_terminal) 

        print("waiting for autoware....")
        for i in tqdm(range(520)):
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
        for i in tqdm(range(WaitingTime.WAITFORAUTONOMOUS)):
            time.sleep(.1)
            CarlaDataProvider.get_world().tick()
            
        control_change_process = change_control_mode(autoware_container_name, default_terminal)
        
        print("\n starting autoware...")
        """for i in tqdm(range(WaitingTime.MAXSTARTDELAY)):
            #TODO implement wait for ego to have specific speed
            vel = env.actors["ego_vehicle"].get_velocity()
            speed = np.linalg.norm([vel.x, vel.y])
            if speed > WaitingTime.STARTTRIGGERSPEED:
                break
            time.sleep(.01)
            CarlaDataProvider.get_world().tick()"""
        for i in tqdm(range(WaitingTime.MAXSTARTDELAY + 200)):
            #TODO implement wait for ego to have specific speed
            pos = env.actors["ego_vehicle"].get_location()
            #if pos.y > WaitingTime.COORDINATECHECK:
            #if pos.x > WaitingTime.COORDINATECHECK2:
            #if pos.y < WaitingTime.COORDINATECHECK6:
            if pos.y < WaitingTime.COORDINATECHECK7:
                break
            time.sleep(.03)
            CarlaDataProvider.get_world().tick()
        print("-------Starting now:-------------")

        done = False
        counter = 0
        #-----------------------------main loop-----------------------------------
        while not done:
            actions = joint_policy(agents)
            obs, reward, done, truncated, info = env.step(actions)
            if env.coll:
                collision = True
                break
            #done = all(done.values())
            env.render()
            counter += 1
            if counter > WaitingTime.MAXTIMESTEPS:
                done = True
            else:
                done = False
            time.sleep(.05)
        #---------------------------------------------------------------------------

        #other_yaw = info["kpis"]["adv_yaw"]
        #other_acc = info["kpis"]["adv_acc"]
        #ttc = min(info["kpis"]["ttc"])

        #print(f"yaw-wasserstein distance: {compute_WD(gt_yaw, other_yaw)}")
        #print(f"acc-wasserstein distance: {compute_WD(gt_acc, other_acc)}")
        #print(f"min ttc: {ttc}")
        #print(f"collision: {collision}")
        
        print("----------------------")
        print("restarting containers")

        """if iteration % 8 == 0:
            print("Also restarting carla this iteartion to prevent it from segfaulting")
            run_docker_restart_command(carla_container_name, default_terminal)
            time.sleep(5)"""
        run_docker_restart_command(autoware_container_name, default_terminal)
        run_docker_restart_command(bridge_container_name, default_terminal)

        save_log_file(env, info, parameters, strategy, test_xosc)
        """if iteration % 8 == 0 and iteration > 0:
            with open(progress_file, "w") as f:
                f.write(str(1))
            print(f"[Main] Progress written: {1}")
            time.sleep(120)"""


def run_dummy_simulation(autoware_container_name, bridge_container_name, carla_container_name, default_terminal, autoware_terminal,
                bridge_terminal, env, args, scene, target_point, strategy, adv_path, pose_publisher, iteration, autoware_target_point=None, parameters=None,
                test_xosc=None, num_iterations=50):
        
    if scene is not None:
        agents = get_agents(env)
    else:
        agents = {}
        agents["ego_vehicle"] = {"ego_vehicle"}
    if adv_path is not None:
        traj = [(carla.Location(x=point[0], y=point[1]), point[2] * 3.6) for point in adv_path]
        adv_agent = TrajectoryFollowingAgent(
            vehicle=env.actors["adversary"],
            trajectory=traj
        )
        agents["adversary"] = adv_agent
    elif strategy == "cat_iterative" or strategy == "cat" or strategy == "cat_no_odd":
        adv_agent = GoingStraightAgent()
        agents["adversary"] = adv_agent
        print("newAGent")

    client = carla.Client(args.carla_host, args.carla_port)
    
    done = False
    
    #----------------main loop--------------------------------------------
    counter = 0
    """for i in range(10000):
        time.sleep(.03)
        CarlaDataProvider.get_world().tick()"""
    while not done:
        if counter % 40 == 0 and counter < 110 and counter > 0 and strategy == "cat_iterative":
            start = time.time()
            _, adv_traj, ego_traj = env._generate_adversarial_route_iterative()
            traj = [
                (carla.Location(x=point[0], y=point[1]), point[2] * 3.6)
                for point in adv_traj
            ]
            adv_agent = TrajectoryFollowingAgent(
                vehicle=env.actors["adversary"],
                trajectory=traj
            )
            agents["adversary"] = adv_agent
            end = time.time()
            print(end - start)
        counter += 1
        actions = joint_policy(agents)
        obs, reward, done, truncated, info = env.step(actions)
        time.sleep(.05)
        if env.coll:
            collision = True
            break
        #done = all(done.values())
        env.render()
        if counter > WaitingTime.MAXTIMESTEPS:
            done = True
        else:
            done = False
    #--------------------------------------------------------------------------
    if False:
        with open(progress_file, "w") as f:
            f.write(str(1))
        print(f"[Main] Progress written: {1}")
        time.sleep(120)


    #--------------------return if we are not within a CAT run-----------------------
    if strategy != "cat":
        # calc metrics and return them
        return
    
    for iteration in range(num_iterations):
        print(f"Starting ADV scenario iteration {iteration} \n")
        obs, info = env.reset(options={
            "scene": scene,
            "adversarial": True
        })

        #gt_yaw = info["kpis"]["adv_yaw"]
        #gt_acc = info["kpis"]["adv_acc"]

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

        print("hererererere-----------------------")
        for i in range(len(traj)):
            print(traj[i])


        done = False
        #-----------------------------main loop-----------------------------------
        counter = 0
        while not done:
            counter += 1
            actions = joint_policy(agents)
            obs, reward, done, truncated, info = env.step(actions)
            if env.coll:
                collision = True
                break
            done = all(done.values())
            done = False
            if counter > 250:
                done = True
            env.render()
            time.sleep(.04)
        #---------------------------------------------------------------------------

        #other_yaw = info["kpis"]["adv_yaw"]
        #other_acc = info["kpis"]["adv_acc"]
        #ttc = min(info["kpis"]["ttc"])
        #save stuff here
        """ego_traj = env._trajectories["ego"]
        adv_traj = env._trajectories["adversary"]

        data = {
            "ego_traj": ego_traj,
            "adv_traj": adv_traj,
            "kpis": info["kpis"]
        }

        with open(f'/workspace/random_results/data{iteration}.json', 'w') as f:
            json.dump(data, f)"""

        #print(f"yaw-wasserstein distance: {compute_WD(gt_yaw, other_yaw)}")
        #print(f"acc-wasserstein distance: {compute_WD(gt_acc, other_acc)}")
        #print(f"min ttc: {ttc}")
        #print(f"collision: {collision}")
    
