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
    """
    Create and configure agent controllers for all agents in the environment.

    Sets up BasicAgent controllers for each agent with specified routes and
    configurations including speed targets and behavior options.

    Args:
        env: The MATS gym environment containing agents and scenario configuration.

    Returns:
        dict: A dictionary mapping agent names to their respective controller instances.
    """
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
    """
    Kill a specific process running in a Docker container.

    Executes a kill command for the specified process ID within the
    given Docker container.

    Args:
        container_name (str): Name of the Docker container.
        pid (str): Process ID to kill.
        default_terminal (str): Terminal device path for output redirection.

    Returns:
        subprocess.Popen: The subprocess object for the kill command.
    """
    command = (
        f"kill {pid}"
    )
    return run_docker_command(container_name, command, default_terminal)
    
    
    
def change_control_mode(container_name, default_terminal):
    """
    Change Autoware's operation mode to autonomous driving.

    Sends a ROS2 service call to switch Autoware to autonomous operation mode.

    Args:
        container_name (str): Name of the Docker container running Autoware.
        default_terminal (str): Terminal device path for output redirection.

    Returns:
        subprocess.Popen: The subprocess object for the command execution.
    """
    command = (
        "cd /work/Valentin_dev/tumgeka_bridge/autoware_fixed/autoware && "
        "source install/setup.bash && "
        "ros2 service call /api/operation_mode/change_to_autonomous autoware_adapi_v1_msgs/srv/ChangeOperationMode {}"
    )
    return run_docker_command(container_name, command, default_terminal)

def init_gnss_again(container_name, default_terminal):
    """
    Re-initialize GNSS localization in Autoware.

    Sends a ROS2 service call to reinitialize the localization system.

    Args:
        container_name (str): Name of the Docker container running Autoware.
        default_terminal (str): Terminal device path for output redirection.

    Returns:
        subprocess.Popen: The subprocess object for the command execution.
    """
    command = (
        "cd /work/Valentin_dev/tumgeka_bridge/autoware_fixed/autoware && "
        "source install/setup.bash && "
        "ros2 service call /api/localization/initialize autoware_adapi_v1_msgs/srv/InitializeLocalization {}"
    )
    return run_docker_command(container_name, command, default_terminal)
    
    
def check_is_stopped(container_name, default_terminal):
    """
    Check if the system is stopped and change to autonomous mode.

    Note: This function appears to duplicate change_control_mode functionality.

    Args:
        container_name (str): Name of the Docker container running Autoware.
        default_terminal (str): Terminal device path for output redirection.

    Returns:
        subprocess.Popen: The subprocess object for the command execution.
    """
    command = (
        "cd /work/Valentin_dev/tumgeka_bridge/autoware_fixed/autoware && "
        "source install/setup.bash && "
        "ros2 service call /api/operation_mode/change_to_autonomous autoware_adapi_v1_msgs/srv/ChangeOperationMode {}"
    )
    return run_docker_command(container_name, command, default_terminal)
    
    
def run_autoware_simulation(container_name, autoware_terminal):
    """
    Launch the Autoware simulation with specific vehicle and sensor models.

    Starts the Autoware end-to-end simulator with CARLA T2 vehicle and sensor kit.

    Args:
        container_name (str): Name of the Docker container running Autoware.
        autoware_terminal (str): Terminal device path for output redirection.

    Returns:
        subprocess.Popen: The subprocess object for the Autoware launch command.
    """
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
    """
    Launch the CARLA-Autoware bridge for communication between systems.

    Starts the bridge that connects CARLA simulation with Autoware,
    enabling data exchange between the simulator and autonomous driving stack.

    Args:
        container_name (str): Name of the Docker container running the bridge.
        bridge_terminal (str): Terminal device path for output redirection.

    Returns:
        subprocess.Popen: The subprocess object for the bridge launch command.
    """
    command = (
        "source install/setup.bash && "
        "ros2 launch carla_autoware_bridge carla_aw_bridge.launch.py "
        "port:=2000 passive:=True register_all_sensors:=False timeout:=180"
    )
    return run_docker_command(container_name, command, bridge_terminal)

def joint_policy(agents, counter=None):
    """
    Generate actions for all agents using their respective control policies.

    Computes control actions for each agent, with special handling for the ego vehicle
    and optional counter-based behavior modification.

    Args:
        agents (dict): Dictionary mapping agent names to their controller instances.
        counter (int, optional): Simulation step counter for time-based behavior changes.

    Returns:
        dict: Dictionary mapping agent names to their control actions as numpy arrays
            containing [throttle, steer, brake] values.
    """
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
    """
    Execute the main simulation with Autoware, CARLA, and adversarial agents.

    Orchestrates a complete simulation run including container management, agent setup,
    and scenario execution with different strategies (CAT, random, etc.).

    Args:
        autoware_container_name (str): Name of Docker container running Autoware.
        bridge_container_name (str): Name of Docker container running CARLA-Autoware bridge.
        carla_container_name (str): Name of Docker container running CARLA.
        default_terminal (str): Default terminal device path for command output.
        autoware_terminal (str): Autoware-specific terminal device path.
        bridge_terminal (str): Bridge-specific terminal device path.
        env: The MATS gym environment instance.
        args: Command-line arguments for simulation configuration.
        scene: Scenic scene object, if using scenario-based simulation.
        target_point (carla.Transform): Target pose for the ego vehicle.
        strategy (str): Simulation strategy ('cat', 'cat_iterative', 'random', etc.).
        adv_path (list, optional): Predefined adversarial trajectory points.
        pose_publisher: ROS2 pose publisher instance.
        iteration (int): Current iteration number.
        autoware_target_point (dict, optional): Autoware-specific target point.
        num_iterations (int, optional): Number of CAT iterations to run. Defaults to 50.
        parameters (list, optional): Additional simulation parameters.
        test_xosc (str, optional): XOSC test scenario identifier.

    Returns:
        bool or None: True if simulation completed successfully (for non-CAT strategies),
            None for CAT strategies, False if Autoware failed to start.
    """

    print("test")
    aw_process = run_autoware_simulation(autoware_container_name, autoware_terminal)
        
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
        """if (counter -1 == 30 or counter -1 == 80  or counter -1 == 95 or counter -1 == 105) and strategy == "cat_iterative":
            print("sleeping")
            time.sleep(30)"""
            
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
    """
    Execute a simplified simulation without full Autoware integration.

    Runs a basic simulation loop without the complexity of Docker container management
    and Autoware integration, primarily for testing and development purposes.

    Args:
        autoware_container_name (str): Name of Docker container running Autoware (unused in dummy mode).
        bridge_container_name (str): Name of Docker container running bridge (unused in dummy mode).
        carla_container_name (str): Name of Docker container running CARLA.
        default_terminal (str): Default terminal device path (unused in dummy mode).
        autoware_terminal (str): Autoware terminal device path (unused in dummy mode).
        bridge_terminal (str): Bridge terminal device path (unused in dummy mode).
        env: The MATS gym environment instance.
        args: Command-line arguments for simulation configuration.
        scene: Scenic scene object, if using scenario-based simulation.
        target_point (carla.Transform): Target pose for the ego vehicle (unused in dummy mode).
        strategy (str): Simulation strategy ('cat', 'cat_iterative', 'random', etc.).
        adv_path (list, optional): Predefined adversarial trajectory points.
        pose_publisher: ROS2 pose publisher instance (unused in dummy mode).
        iteration (int): Current iteration number.
        autoware_target_point (dict, optional): Autoware-specific target point (unused in dummy mode).
        parameters (list, optional): Additional simulation parameters (unused in dummy mode).
        test_xosc (str, optional): XOSC test scenario identifier (unused in dummy mode).
        num_iterations (int, optional): Number of CAT iterations to run. Defaults to 50.

    Returns:
        None: Function does not return a value.
    """
        
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
    
