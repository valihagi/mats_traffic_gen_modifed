import argparse
from datetime import datetime
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
from run_simulation import run_dummy_simulation, run_simulation
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
from helpers import create_random_traj, generate_even_timestamps, generate_parametrized_adversarial_route, generate_random_adversarial_route, generate_timestamps, run_docker_command, save_log_file, visualize_traj
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

progress_file = "/workspace/shared/progress.txt"

autoware_container_name = "objective_aryabhata"
bridge_container_name = "ecstatic_panini"
carla_container_name = "stupefied_villani"

autoware_terminal = "/dev/pts/8"
bridge_terminal = "/dev/pts/9"
default_terminal = "/dev/pts/10"


"""
This example shows how to use the CarlaVisualizationWrapper to create visualizations
inside the CARLA simulator. The visualization is done by adding a callback to the wrapper.
"""

NUM_EPISODES = 1000

def get_candidates():
    try:
        with open("/workspace/shared/doe_messages/candidates.json", "r") as f:
            return json.load(f)
    except:
        return None

def write_kpi(data):
    print(f"Writing {data}")
    with open("/workspace/shared/doe_messages/kpi.json", "w") as f:
        json.dump(data, f)

def clear_candiates():
    try:
        with open("/workspace/shared/doe_messages/candidates.json", "w") as f:
            json.dump([], f)
    except:
        return None


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

def get_all_vehicles(client):
    # Get the world (simulation environment)
    world = client.get_world()

    # Get all actors (objects in the simulation)
    actors = world.get_actors()

    # Filter out vehicles (cars)
    vehicles = [actor for actor in actors if actor.type_id.startswith('vehicle.')]

    return vehicles

    

def main(args):
    print("starting here")
    with open(progress_file, "w") as f:
        f.write(str(0))
        print(f"[Main] Progress written: {0}")
    rclpy.init()
    pose_publisher = PosePublisher()
    SEED = int(datetime.now().timestamp())
    random.seed(SEED)
    np.random.seed(SEED)
    strategy = args.strategy
    test_xosc = "test"

    env = mats_gym.openscenario_env(
        scenario_files=f"scenarios/open_scenario/{test_xosc}.xosc",
        host=args.carla_host,
        port=args.carla_port,
        seed=SEED,
        render_mode="human",
        render_config=camera_pov(agent="ego_vehicle"),
    )
    print("xosc here:")
    print(test_xosc)

    env = AdversarialTrainingWrapper(
        env=env,
        args=args,
        model_path="cat/advgen/pretrained/densetnt.bin",
        ego_agent="ego_vehicle",
        adv_agents="adversary",
    )
    env.agents.append('adversary')
    """vehicles = []
    client = carla.Client(args.carla_host, args.carla_port)
    while len(vehicles) == 0:
        CarlaDataProvider.get_world().tick()
        vehicles = get_all_vehicles(client)
        print(vehicles)
        adv_veh = None
        for veh in vehicles:
            if veh.type_id == "vehicle.audi.tt":
                adv_veh = veh
                break"""
    
    #dummy target point for ego, needs to be set for xosc scenarios
    # Hardcoded for now since ego behaviour is not specified in our xosc file
    #(76,17)
    
    
    """target_point = carla.Transform(
        carla.Location(55, -13.5, 0.0),  # Assuming Z = 0 for ground level
        carla.Rotation(yaw=angle_deg)
    )"""
    #other scenario
    if test_xosc == "test":
        direction_vector = (-1, 0)
        angle_rad = math.atan2(direction_vector[1], direction_vector[0])
        angle_deg = math.degrees(angle_rad)
        target_point = carla.Transform(
            carla.Location(82.5, -70, 0.0),  # Assuming Z = 0 for ground level
            carla.Rotation(yaw=angle_deg)
        )

    #test2.xosc
    if test_xosc == "test2":
        direction_vector = (0, -1)
        angle_rad = math.atan2(direction_vector[1], direction_vector[0])
        angle_deg = math.degrees(angle_rad)
        target_point = carla.Transform(
            carla.Location(43.5, -40, 0),  # Assuming Z = 0 for ground level
            carla.Rotation(yaw=angle_deg)
        )

    #test6.xosc
    if test_xosc == "test6":
        direction_vector = (1, 0)
        angle_rad = math.atan2(direction_vector[1], direction_vector[0])
        angle_deg = math.degrees(angle_rad)
        target_point = carla.Transform(
            carla.Location(-72.5, 68.5, 0),  # Assuming Z = 0 for ground level
            carla.Rotation(yaw=angle_deg)
        )

    """#test7.xosc OLD
    direction_vector = (0, -1)
    angle_rad = math.atan2(direction_vector[1], direction_vector[0])
    angle_deg = math.degrees(angle_rad)
    target_point = carla.Transform(
        carla.Location(-104.5, -73, 0),  # Assuming Z = 0 for ground level
        carla.Rotation(yaw=angle_deg)
    )"""

    #test7.xosc new
    if test_xosc == "test7":
        direction_vector = (1, 0)
        angle_rad = math.atan2(direction_vector[1], direction_vector[0])
        angle_deg = math.degrees(angle_rad)
        target_point = carla.Transform(
            carla.Location(-90, -16.5, 0),  # Assuming Z = 0 for ground level
            carla.Rotation(yaw=angle_deg)
        )

    autoware_target_point = None #"""{
            #'x': 71.0,
            #'y': -13.5,
            #'z': 0.0,  # Assuming Z remains the same
            #'x1': 0.0,
            #'y1': 0.0,
            #'z1': -0.9999924884745914,
            #'w': 0.003875950772917655
        #}"""
    print("Target point is:----------------:")
    print(target_point)
    #get same as in scenic scneario should work

    ran_counter = 0

    if strategy == "doe":
        print("USING Active DoE")
        with open("active_doe_module/setup_xosc.json", "r") as fp:
            setup = json.load(fp)
            
        """if len(start_design)>0:
            measurements = [
                dict(
                    Variations=variations,
                    Responses=run_simulation(variations=variations, setup=setup)
                ) for variations in start_design
            ]
            client.insert_measurements(measurements=measurements)"""

        
        json_file_number = 7

        logs = "/workspace/doe_logs/"

        json_file = f"{logs}meas{json_file_number}.json"
        samples_file = f"{logs}samples{json_file_number}.json"

        while True:
            while True:
                candidates = get_candidates()
                if candidates is None or candidates == []:
                    print("wait for next input")
                    time.sleep(5)
                    continue
                else:
                    break
            if candidates == [1]:
                print("received eng signal for doe")
                break
            print(candidates)
            measurements = []
            measurements_no_index = []

            for candidate in candidates:
                ran_counter += 1
                aw_started = False
                while not aw_started:
                    traj = None
                    ##unpack candiates and insert them into env.scenario
                    variations = candidate["Variations"]
                    print("parameters are: ")
                    print(variations)
                    obs, info = env.reset(options={
                        })
                    
                    CarlaDataProvider.get_world().tick()
                    
                    #times = generate_timestamps(100, 80, 1, )
                    """times = generate_even_timestamps(80, 18)
                    adv_traj, ego_traj, ego_width, ego_length = generate_parametrized_adversarial_route(env, 80, times)"""

                    adv = env.actors["adversary"]
                    adv_loc = adv.get_location()
                    random_offset = variations["start_pos_offset"]
                    adv_loc.x = adv_loc.x - random_offset
                    print(random_offset)
                    new_transform = carla.Transform(location=adv_loc, rotation=adv.get_transform().rotation)
                    adv.set_transform(new_transform)
                    
                    adv_traj, parameters = create_random_traj((adv_loc.x, -adv_loc.y), env._network, variations)
                    parameters.append(random_offset)

                    print(f"STARTING scenario... counter: {ran_counter}")
                    try:
                        # Your main code
                        print("RUNNING scenario...")

                        aw_started = run_simulation(autoware_container_name=autoware_container_name,
                        bridge_container_name=bridge_container_name,
                        carla_container_name=carla_container_name,
                        default_terminal=default_terminal,
                        autoware_terminal=autoware_terminal,
                        bridge_terminal=bridge_terminal,
                        env=env,
                        args=args,
                        scene=None,
                        iteration=ran_counter,
                        target_point=target_point,
                        strategy=strategy,
                        adv_path=adv_traj,
                        pose_publisher=pose_publisher,
                        autoware_target_point= autoware_target_point,
                        parameters=parameters,
                        test_xosc=test_xosc)
                        print(f"AW started: {aw_started}")
                    
                    except Exception as e:
                        print(f"EXCEPTION: {e}")
                    finally:
                        print("SCRIPT EXITED.")
                
                #get KPIS
                kpis = env.get_min_ttc_as_dict()
                measurements.append(dict(
                    Index=candidate['Index'],
                    Variations=candidate['Variations'],
                    Responses=kpis)
                )
                measurements_no_index.append(dict(
                    Variations=candidate['Variations'],
                    Responses=kpis)
                )
                print("-------------measurements below---------------------")
                print(type(measurements))
                print(measurements)
                print("-------------measurements above---------------------")

            if os.path.exists(json_file):
                with open(json_file, "r") as f:
                    try:
                        existing_data = json.load(f)
                    except json.JSONDecodeError:
                        existing_data = []
            else:
                existing_data = []

            # Append and save back
            existing_data.extend(measurements_no_index)  # assumes new_data is a list
            with open(json_file, "w") as f:
                json.dump(existing_data, f, indent=2)
            
            write_kpi(measurements)
            clear_candiates()

        print("exiting DoE")
        return
        
    if strategy == "doe_finished":
        print("USING Active DoE finished samples")
            
        json_file_number = 7

        json_file = f"/workspace/doe_logs/meas{json_file_number}.json"
        samples_file = f"/workspace/doe_logs/samples{json_file_number}.json"
        if os.path.exists(samples_file):
            with open(samples_file, "r") as f:
                data = json.load(f)

        data.sort(key=lambda candidate: candidate["Responses"]["min_ttc"])

        for i in range(25,30): 
            candidate = data[i]
            print(candidate["Responses"]["min_ttc"])
            ran_counter += 1
            aw_started = False
            while not aw_started:
                traj = None
                ##unpack candiates and insert them into env.scenario
                variations = candidate["Variations"]
                print("parameters are: ")
                print(variations)
                obs, info = env.reset(options={
                    })
                
                CarlaDataProvider.get_world().tick()
                
                #times = generate_timestamps(100, 80, 1, )
                """times = generate_even_timestamps(80, 18)
                adv_traj, ego_traj, ego_width, ego_length = generate_parametrized_adversarial_route(env, 80, times)"""

                adv = env.actors["adversary"]
                adv_loc = adv.get_location()
                random_offset = variations["start_pos_offset"]
                adv_loc.x = adv_loc.x - random_offset
                print(random_offset)
                new_transform = carla.Transform(location=adv_loc, rotation=adv.get_transform().rotation)
                adv.set_transform(new_transform)
                
                adv_traj, parameters = create_random_traj((adv_loc.x, -adv_loc.y), env._network, variations)
                parameters.append(random_offset)

                print(f"STARTING scenario... counter: {ran_counter}")
                try:
                    # Your main code
                    print("RUNNING scenario...")

                    aw_started = run_simulation(autoware_container_name=autoware_container_name,
                    bridge_container_name=bridge_container_name,
                    carla_container_name=carla_container_name,
                    default_terminal=default_terminal,
                    autoware_terminal=autoware_terminal,
                    bridge_terminal=bridge_terminal,
                    env=env,
                    args=args,
                    scene=None,
                    iteration=ran_counter,
                    target_point=target_point,
                    strategy=strategy,
                    adv_path=adv_traj,
                    pose_publisher=pose_publisher,
                    autoware_target_point= autoware_target_point,
                    parameters=parameters,
                    test_xosc=test_xosc)
                    print(f"AW started: {aw_started}")
                
                except Exception as e:
                    print(f"EXCEPTION: {e}")
                finally:
                    print("SCRIPT EXITED.")

        print("finished running all samples, exiting now...")

        return
    ran_counter = 0
    iteration_counter = 0
    already_reset = False
    while(ran_counter < NUM_EPISODES):
        iteration_counter += 1
        traj = None
        if strategy == "random":
            print("USING Random DoE")
            if not already_reset:
                for i in range(10):
                    try:
                        obs, info = env.reset(options={
                        })
                        print("Reseting the environment")
                        break
                    except:
                        print("Carla seems to be down, taking a short timeout and trying again...")
                        time.sleep(60)
            CarlaDataProvider.get_world().tick()
            already_reset = False
            #times = generate_timestamps(100, 80, .6, 3.5)
            #adv_traj, ego_traj, ego_width, ego_length, parameters = generate_random_adversarial_route(env, 80, times)

            #random offset from start position
            adv = env.actors["adversary"]
            adv_loc = adv.get_location()
            random_offset = random.uniform(-5, 5)
            adv_loc.x = adv_loc.x - 0#random_offset #+1.9
            print(random_offset)
            new_transform = carla.Transform(location=adv_loc, rotation=adv.get_transform().rotation)
            adv.set_transform(new_transform)
            
            adv_traj, parameters = create_random_traj((adv_loc.x, -adv_loc.y), env._network, test_xosc=test_xosc)
            parameters.append(random_offset)

            """env = mats_gym.openscenario_env(
                scenario_files="scenarios/open_scenario/test.xosc",
                host=args.carla_host,
                port=args.carla_port,
                seed=SEED,
                render_mode="human",
                render_config=camera_pov(agent="ego_vehicle"),
            )

            env = AdversarialTrainingWrapper(
                env=env,
                args=args,
                model_path="cat/advgen/pretrained/densetnt.bin",
                ego_agent="ego_vehicle",
                adv_agents="adversary",
            )
            env.agents.append('adversary')"""
            
            """obs, info = env.reset(options={
            })"""
            #check ego traj
            #env.check_on_roadgraph_old(ego_traj, iteration_counter)
            """if not env.check_on_roadgraph(adv_traj, iteration_counter):
                print("Created trajectory is not on the roadgraph and will therefore be skipped!")
                already_reset = True
                save_log_file(env, info, parameters, iteration_counter, in_odd=False)
                continue"""
            ran_counter += 1
        elif strategy == "cat":
            print("USING CAT")
            obs, info = env.reset(options={
            })
            adv_traj = None
            parameters = None

        elif strategy == "cat_no_odd":
            print("USING CAT_no_odd")
            obs, info = env.reset(options={
            })
            adv_traj = None
            parameters = None

        elif strategy == "cat_iterative":
            print("USING CAT iteratively")
            obs, info = env.reset(options={
            })
            adv_traj = None
            parameters = None
            ran_counter += 1

        else:
            print("unknown startegy please check the config.")
            return
        
        """world = CarlaDataProvider.get_world()
        ego = env.actors["adversary"]
        ego_loc = ego.get_location()
        color = carla.Color(*(100,0,0))
        arr = [[104.5, 27.5],
               [104.5, 19],
               [101, 15.5],
               [97.5, 13.5],
               [97.5, 17.5],
               [101, 20],
               [102.5, 22.5]]
        
        for point in arr:
            loc = carla.Location(point[0], point[1], z=5)
            world.debug.draw_point(loc, size=.5, color=color)

        spec = world.get_spectator()
        spec.set_transform(carla.Transform(
            carla.Location(ego_loc.x, ego_loc.y, ego_loc.z + 60),
            carla.Rotation(pitch=-90)
        ))

        settings = world.get_settings()
        world.apply_settings(settings)
        world.tick()l
        time.sleep(100)"""

        print(f"STARTING scenario... counter: {ran_counter}")
        try:
            # Your main code
            print("RUNNING scenario...")
        

            run_dummy_simulation(autoware_container_name=autoware_container_name,
                        bridge_container_name=bridge_container_name,
                        carla_container_name=carla_container_name,
                        default_terminal=default_terminal,
                        autoware_terminal=autoware_terminal,
                        bridge_terminal=bridge_terminal,
                        env=env,
                        args=args,
                        scene=None,
                        target_point=target_point,
                        strategy=strategy,
                        adv_path=adv_traj,
                        pose_publisher=pose_publisher,
                        iteration=ran_counter,
                        autoware_target_point= autoware_target_point,
                        parameters=parameters,
                        test_xosc=test_xosc)
            """kpis = env.get_min_ttc_as_dict()
            print(kpis)
            time.sleep(100)"""
            
        except Exception as e:
            print(f"EXCEPTION: {e}")
        finally:
            print("SCRIPT EXITED.")
        
        #get KPIS

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--OV_traj_num', type=int, default=64)
    parser.add_argument('--AV_traj_num', type=int, default=1)
    parser.add_argument('--carla-host', type=str, default="localhost")
    parser.add_argument('--carla-port', type=int, default=2000)
    parser.add_argument('--strategy', type=str, default="cat")
    gen = AdvGenerator(parser, pretrained_path="./cat/advgen/pretrained/densetnt.bin")
    main(gen.args)
