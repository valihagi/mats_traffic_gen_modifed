from datetime import datetime
import random
import subprocess
import os
import time
import io
import math
from TwoDimSSM import TTC_DRAC_MTTC
import carla
import bezier
import numpy as np
from TwoDimTTC import TTC
import pandas as pd
import shapely.geometry
from shapely.geometry import Point, Polygon
from scenic.domains.driving.roads import Lane
from scenic.domains.driving.roads import Network
from scipy.interpolate import interp1d

from cat.advgen.adv_generator import get_polyline_yaw
import xml.etree.ElementTree as ET
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
import json
from matplotlib import pyplot as plt


def run_docker_command(container_name, command, output_dest):
    exec_command = [
        "docker", "exec", "-it", container_name, "bash", "-c", command
    ]
    with open(output_dest, 'w') as f:
        process = subprocess.Popen(
            exec_command,
            stdout=f,
            stderr=f,
            universal_newlines=True,
            preexec_fn=os.setsid  # Create a new process group so it can be killed later
        )    
        
    return process
    
    
def run_docker_restart_command(container_name, output_dest):
    exec_command = [
        "docker", "restart", container_name
    ]
    with open(output_dest, 'w') as f:
        process = subprocess.Popen(
            exec_command,
            stdout=f,
            stderr=f,
            universal_newlines=True,
            preexec_fn=os.setsid  # Create a new process group so it can be killed later
        )
        
    stdout, stderr = process.communicate()
    
    if process.returncode == 0:
        print(f"Successfully restarted container: {container_name}")
        if stdout:
            print(stdout.decode())
    else:
        print(f"Failed to restart container: {container_name}")
        if stderr:
            print(stderr.decode())
        return
          
    return process
    

    
    
def get_docker_ouptut(process, tick_carla_client, world, wait_lines):
    # Capture and print output in real time
    for i in range(wait_lines):
        print(i)
        if tick_carla_client:
            world.tick()
        output = process.stdout.readline()
        if output == '':  # Empty output means end of the stream
            if process.poll() is not None:
                break
        if output:
            print(output.strip())  # Print the command output in real time
        time.sleep(.1)  # Avoid busy waiting

    # Check if the process terminated with an error
    #stderr_output = process.stderr.read()
    #if stderr_output:
        #print("Errors:\n", stderr_output)
        
        
def get_carla_point_from_scene(scene):
    length = len(scene.egoObject.route[-1].centerline)
    if length > 10:
        last_point = scene.egoObject.route[-1].centerline[6]
        second_last_point = scene.egoObject.route[-1].centerline[7]
    elif length > 5:
        last_point = scene.egoObject.route[-1].centerline[2]
        second_last_point = scene.egoObject.route[-1].centerline[3]
    elif length > 2:
        last_point = scene.egoObject.route[-1].centerline[1]
        second_last_point = scene.egoObject.route[-1].centerline[2]
    else:
        last_point = scene.egoObject.route[-1].centerline[0]
        second_last_point = scene.egoObject.route[-1].centerline[1]
    
    #print(f"last_point {last_point.x} / {last_point.y}")
    #print(f"second_to last_point {second_last_point.x} / {second_last_point.y}")
    direction_vector = (
        last_point.x - second_last_point.x,
        last_point.y - second_last_point.y
    )
    angle_rad = math.atan2(direction_vector[1], direction_vector[0])
    angle_deg = math.degrees(angle_rad)
    #print(f"yaw is {angle_deg}")
    
    return carla.Transform(
        carla.Location(last_point.x, last_point.y, 0.0),  # Assuming Z = 0 for ground level
        carla.Rotation(yaw=angle_deg)
    )

def get_carla_point_from_xml():
    """
    Only dummy code for now
    """

    return carla.Transform(
        carla.Location(x=92.189285, y=39.811680, z=-0.000798),
        carla.Rotation(pitch=-0.021788, yaw=-88.243736, roll=0.000969)
    )

def visualize_traj(x, y, yaw, width, length, color, world, map, skip=5):
    #world = CarlaDataProvider.get_world()
    #map = CarlaDataProvider.get_map()
    color = carla.Color(*color)
    ts = np.arange(0, 0.1 * len(x), 0.1)
    for t in range(0, len(x), skip):
        x_t, y_t, yaw_t = x[t], y[t], yaw[t]
        wp = map.get_waypoint(carla.Location(x=x_t.item(), y=y_t.item()))
        loc = carla.Location(x=x_t.item(), y=y_t.item(), z=wp.transform.location.z)
        bbox = carla.BoundingBox(
            loc,
            carla.Vector3D(x=length / 2, y=width / 2, z=0.05)
        )
        front = map.get_waypoint(carla.Location(
            x=x_t.item() + 0.5 * length * np.cos(yaw_t.item()),
            y=y_t.item() - 0.5 * length * np.sin(yaw_t.item())
        ))

        pitch = np.arcsin((front.transform.location.z - loc.z) / (length / 2))
        pitch = np.rad2deg(pitch)
        world.debug.draw_point(loc, size=0.1, color=color, life_time=0)
        world.debug.draw_box(bbox, carla.Rotation(yaw=yaw_t.item(), pitch=pitch),
                             thickness=0.1, color=color,
                             life_time=0)
        time = ts[t]
        loc += carla.Location(z=0.2)
        world.debug.draw_string(loc, f"{time:.1f}s", draw_shadow=True, color=color,
                                life_time=1000000)


def generate_random_adversarial_route(env, num_waypoints, times):
    random.seed(datetime.now().timestamp())
    vehicle = env.actors["adversary"]

    x_start, y_start = vehicle.get_location().x, vehicle.get_location().y
    end_x, end_y = create_random_end_point(vehicle)

    # need two random control points to create a cubic bezier curve
    x_ctrl_1, y_ctrl_1, length_factor1, width_offset1 = create_random_control_point(vehicle, end_x, end_y)
    x_ctrl_2, y_ctrl_2, length_factor2, width_offset2 = create_random_control_point(vehicle, end_x, end_y)

    nodes = np.asfortranarray([
    [x_start, x_ctrl_1, x_ctrl_2, end_x],
    [y_start, y_ctrl_1, y_ctrl_2, end_y],
    ])
    
    curve = bezier.Curve.from_nodes(nodes)
    
    # calculate waypoints on the curve
    trajectory = []
    traj = []
    random_max_speed = random.uniform(5, 15)
    random_acc = random.uniform(5, 15)
    for i in np.linspace(0, 1, num_waypoints):
        speed = min(random_max_speed, (i + 1) ** random_acc)
        point = curve.evaluate(i)
        trajectory.append([point[0][0], point[1][0], speed])
        traj.append([point[0][0], point[1][0]])

    ego_width, ego_length = vehicle.bounding_box.extent.y * 2, vehicle.bounding_box.extent.x * 2
    ego_traj = np.concatenate([
        traj,
        np.rad2deg(get_polyline_yaw(traj)).reshape(-1, 1)
    ], axis=1)

    parameters = [end_x, end_y,length_factor1, width_offset1,length_factor2,width_offset2]

    return trajectory, ego_traj, ego_width, ego_length, parameters


def generate_parametrized_adversarial_route(env, num_waypoints, times):
    vehicle = env.actors["adversary"]

    x_start, y_start = vehicle.get_location().x, vehicle.get_location().y

    transform = vehicle.get_transform()
    x, y = transform.location.x, transform.location.y
    distance = env.parameters["distance_to_target"] # 50 to 55
    
    angle_offset = env.parameters["angle_to_target"] #random.uniform(-math.pi / 2, math.pi / 2)
    random_angle = math.radians(transform.rotation.yaw) + angle_offset

    end_x = x + distance * math.cos(random_angle)
    end_y = y + distance * math.sin(random_angle)


    # need two random control points to create a cubic bezier curve
    distance = env.parameters["distance_to_control"] # 20 to 30
    
    angle_offset = env.parameters["angle_to_control"] #random.uniform(-math.pi / 2, math.pi / 2)
    random_angle = math.radians(transform.rotation.yaw) + angle_offset
    x_ctrl_1 = x + distance * math.cos(random_angle)
    y_ctrl_1 = y + distance * math.sin(random_angle)

    nodes = np.asfortranarray([
    [x_start, x_ctrl_1, end_x],
    [y_start, y_ctrl_1, end_y],
    ])
    
    curve = bezier.Curve.from_nodes(nodes)
    
    # calculate waypoints on the curve
    trajectory = []
    traj = []
    for i in np.linspace(0, 1, num_waypoints):
        speed = min(6.7, (i + 1) ** 15)
        point = curve.evaluate(i)
        trajectory.append([point[0][0], point[1][0], speed])
        traj.append([point[0][0], point[1][0]])

    ego_width, ego_length = vehicle.bounding_box.extent.y * 2, vehicle.bounding_box.extent.x * 2
    ego_traj = np.concatenate([
        traj,
        np.rad2deg(get_polyline_yaw(traj)).reshape(-1, 1)
    ], axis=1)

    xml_file = "scenarios/open_scenario/Catalogs/Trajectories/TrajectoryCatalog2.xosc"  # Replace with your actual file name
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Define new coordinates for each vertex
    new_coordinates = traj

    for i, vertex in enumerate(root.findall(".//Polyline/Vertex")):
        # Update the time attribute
        vertex.set("time", str(times[i]))  # Example: Set time based on index

        # Find the WorldPosition element inside Position
        world_position = vertex.find("./Position/WorldPosition")
        if world_position is not None:
            world_position.set("x", str(new_coordinates[i][0]))
            world_position.set("y", str(new_coordinates[i][1]))


    # Save the modified XML back to a file
    modified_xml_file = "scenarios/open_scenario/Catalogs/Trajectories/TrajectoryCatalog.xosc"
    tree.write(modified_xml_file, encoding="utf-8", xml_declaration=True)

    print(f"Updated XML saved as {modified_xml_file}")

    return trajectory, ego_traj, ego_width, ego_length

def generate_trajectories_from_position(start_position, heading_deg, road_network: Network, max_depth=3):
    """
    Generate all possible trajectories from a starting position and heading on a Scenic road network.

    Args:
        start_position (tuple): (x, y) world position of the vehicle.
        heading_deg (float): Heading angle in degrees (0 = east, 90 = north, etc.).
        road_network (Network): Scenic road network.
        max_depth (int): Max depth to search through successor lanes.

    Returns:
        list of list of (x, y): Each trajectory is a list of (x, y) tuples.
    """
    min_distance = 30
    max_distance = 80
    car_point = Point(start_position)

    all_trajectories = []

    # Utility: Compute distance between two (x, y) points
    def dist(p1, p2):
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

    # Compute length of a path
    def compute_length(points):
        return sum(dist(points[i], points[i - 1]) for i in range(1, len(points)))

    # Remove overlapping (prefix-contained) trajectories
    def remove_overlapping_subtrajectories(trajectories):
        sorted_trajectories = sorted(trajectories, key=len, reverse=True)
        filtered = []
        for traj in sorted_trajectories:
            if not any(traj == other[:len(traj)] for other in filtered if len(traj) <= len(other)):
                filtered.append(traj)
        return filtered

    # Step 1: Find the starting lane
    closest_lane = None
    for lane in road_network.lanes:
        if lane.containsPoint(car_point):
            closest_lane = lane
            break

    if not closest_lane or not closest_lane.centerline:
        return []

    # Step 2: Get the start index on the centerline **forward only**
    centerline_points = [(pt[0], pt[1]) for pt in closest_lane.centerline.points]
    dists = [dist(start_position, pt) for pt in centerline_points]
    closest_index = dists.index(min(dists))

    # Step 3: Skip centerline points behind the start_position
    # Use heading to determine direction (optional — currently we just go forward from closest)
    forward_points = centerline_points[closest_index:]

    def dfs(current_lane: Lane, path_so_far, distance_so_far, depth):
        if depth > max_depth or distance_so_far >= max_distance:
            return

        centerline = [(pt[0], pt[1]) for pt in current_lane.centerline.points]
        if not centerline:
            return

        # Decide starting point list
        if not path_so_far:
            # We're at the start — use sliced forward points
            new_path = [start_position]
            last_point = start_position
            points_to_add = forward_points
        else:
            new_path = path_so_far[:]
            last_point = new_path[-1]
            points_to_add = [(pt[0], pt[1]) for pt in current_lane.centerline.points][1:]

        current_dist = distance_so_far

        for pt in points_to_add:
            step_dist = dist(last_point, pt)
            if current_dist + step_dist > max_distance:
                break
            new_path.append(pt)
            current_dist += step_dist
            last_point = pt

            if min_distance <= current_dist <= max_distance:
                all_trajectories.append(new_path[:])

        # Continue to successors
        for maneuver in current_lane.maneuvers:
            next_lane = maneuver.connectingLane or maneuver.endLane
            if next_lane and next_lane.centerline:
                dfs(next_lane, new_path, current_dist, depth + 1)

    # Step 4: Begin traversal from the start position and trimmed centerline
    dfs(closest_lane, [], 0.0, 0)

    return remove_overlapping_subtrajectories(all_trajectories)

def resample_trajectory_to_equal_distance(trajectory_xy, num_points=180):
    """
    Resample a trajectory to have `num_points` with equal arc-length spacing,
    preserving the first and last points.

    Args:
        trajectory_xy (list or np.ndarray): Array of shape (N, 2) with [x, y] points.
        num_points (int): Number of resampled points (default: 80).

    Returns:
        np.ndarray: Resampled trajectory of shape (num_points, 2).
    """
    trajectory_xy = np.asarray(trajectory_xy)

    # Compute cumulative arc-length (distance)
    deltas = np.diff(trajectory_xy, axis=0)
    segment_lengths = np.linalg.norm(deltas, axis=1)
    arc_lengths = np.concatenate([[0], np.cumsum(segment_lengths)])

    total_length = arc_lengths[-1]
    desired_arc_lengths = np.linspace(0, total_length, num_points)

    # Interpolate x and y independently based on arc-length
    interp_x = interp1d(arc_lengths, trajectory_xy[:, 0], kind='linear')
    interp_y = interp1d(arc_lengths, trajectory_xy[:, 1], kind='linear')

    resampled_x = interp_x(desired_arc_lengths)
    resampled_y = interp_y(desired_arc_lengths)

    return np.stack((resampled_x, resampled_y), axis=1)

def create_random_traj(ego_loc, network, parameters=None):
    trajs = generate_trajectories_from_position(ego_loc, 0, network)
    if parameters is not None:
        idx = parameters["turn_idx"]
        random_max_speed = parameters["max_speed"]
        random_acc = parameters["acceleration"]
    else:
        idx = random.randint(0, len(trajs) - 1)
        random_max_speed = random.uniform(2.5, 15)
        random_acc = random.uniform(.4, 1.5)

    traj1 = resample_trajectory_to_equal_distance(trajs[idx])
    traj1 = traj1[:120]
    
    trajectory = []
    for i in range(len(traj1)):
        speed = min(random_max_speed, (i + 1) ** random_acc)
        point = traj1[i]
        trajectory.append([point[0], -point[1], speed])
    return trajectory, [idx, random_acc, random_max_speed]

def generate_timestamps(total_distance, num_points, acc, vmax):
    # Phase 1: Acceleration phase
    t_accel = vmax / acc  # Time to reach vmax
    d_accel = 0.5 * acc * t_accel**2  # Distance covered during acceleration

    if d_accel > total_distance / 2:
        # If we reach half the total distance before vmax, adjust the calculation
        t_accel = np.sqrt(total_distance / acc)
        vmax = acc * t_accel
        d_accel = 0.5 * acc * t_accel**2
        t_const = 0
    else:
        # Phase 2: Constant velocity phase
        d_const = total_distance - 2 * d_accel  # Remaining distance
        t_const = d_const / vmax  # Time spent at constant velocity

    # Phase 3: Deceleration phase (same as acceleration)
    t_decel = t_accel
    total_time = 2 * t_accel + t_const

    # Generate time values
    timestamps = []
    distances = np.linspace(0, total_distance, num_points)  # Equally spaced trajectory points

    for d in distances:
        if d < d_accel:
            # Acceleration phase: solve d = 0.5 * acc * t^2 for t
            t = np.sqrt(2 * d / acc)
        elif d < d_accel + d_const:
            # Constant velocity phase: solve d = d_accel + vmax * (t - t_accel) for t
            t = t_accel + (d - d_accel) / vmax
        else:
            # Deceleration phase: solve d = total_distance - 0.5 * acc * (t_decel - (t - t_accel - t_const))^2
            remaining_d = total_distance - d
            t = total_time - np.sqrt(2 * remaining_d / acc)
        
        timestamps.append(t)

    return np.round(timestamps, 3)


def generate_even_timestamps(num_points, timespan):
    return np.linspace(0, timespan, num_points)

def create_random_end_point(vehicle):
    min_dist = 30
    max_dist = 60
    transform = vehicle.get_transform()
    x, y = transform.location.x, transform.location.y
    epsilon = 0.1
    angle_offset = random.uniform(-math.pi / 2 + epsilon, math.pi / 2 - epsilon) # epsilon to not allow fully 90 degree
    random_angle = math.radians(transform.rotation.yaw) + angle_offset
    distance = random.uniform(min_dist * max(0.5,math.cos(angle_offset)), max_dist * max(0.5,math.cos(angle_offset)))

    new_x = x + distance * math.cos(random_angle)
    new_y = y + distance * math.sin(random_angle)

    return new_x, new_y


def create_random_control_point(vehicle, target_x, target_y):
    width=30
    transform = vehicle.get_transform()
    yaw = math.radians(transform.rotation.yaw)
    distance = 5
    start_x, start_y = transform.location.x + distance * math.cos(yaw), transform.location.y + distance * math.sin(yaw)
    target_x, target_y = target_x - distance * math.cos(yaw), target_y - distance * math.sin(yaw)
    dx = target_x - start_x
    dy = target_y - start_y
    length = math.sqrt(dx**2 + dy**2)

    if length == 0:
        raise ValueError("Start and end points are the same; rectangle cannot be created.")

    # Normalize direction vector
    dx /= length
    dy /= length

    # Compute perpendicular vector
    perp_x = -dy  # Rotate by 90 degrees
    perp_y = dx

    # Half width for correct placement
    half_width = width / 2

    # Compute the four corners of the rectangle
    bottom_left = (start_x - half_width * perp_x, start_y - half_width * perp_y)
    bottom_right = (start_x + half_width * perp_x, start_y + half_width * perp_y)
    top_left = (target_x - half_width * perp_x, target_y - half_width * perp_y)
    top_right = (target_x + half_width * perp_x, target_y + half_width * perp_y)

    # Generate a random point inside the rectangle
    random_length_factor = random.uniform(0, 1)  # Random position along the length
    random_width_offset = random.uniform(-half_width, half_width)  # Random width offset

    # Interpolate along the direction
    random_x = start_x + random_length_factor * (target_x - start_x) + random_width_offset * perp_x
    random_y = start_y + random_length_factor * (target_y - start_y) + random_width_offset * perp_y
    return random_x, random_y, random_length_factor, random_width_offset

def get_vehicle_data(vehicle, near_miss=False):
    transform = vehicle.get_transform()
    location = transform.location
    rotation = transform.rotation
    velocity = vehicle.get_velocity()
    acceleration = vehicle.get_acceleration()
    bbox = vehicle.bounding_box

    # Normalize heading vector
    yaw = np.deg2rad(rotation.yaw)
    hx = np.cos(yaw)
    hy = np.sin(yaw)

    # Acceleration along heading direction
    acc_vector = np.array([acceleration.x, acceleration.y])
    heading_vector = np.array([hx, hy])
    acc_along_heading = np.dot(acc_vector, heading_vector)

    if near_miss:
        length = bbox.extent.x * 2.8
        width = bbox.extent.y * 2.8
    else:
        length = bbox.extent.x * 2.0
        width = bbox.extent.y * 2.0

    return {
        'x': location.x,
        'y': location.y,
        'vx': velocity.x,
        'vy': velocity.y,
        'hx': hx,
        'hy': hy,
        'acc': acc_along_heading,
        'length': length,
        'width': width
    }

def calculate_risk_coefficient(ego_vehicle, adv_vehicle, G=1.0, k=1.0, epsilon=1e-6):
    """
    Compute the risk coefficient (DRP) from an obstacle vehicle to an ego vehicle.

    Parameters:
        ego_vehicle (carla.Actor): The ego vehicle (target of risk computation)
        obstacle_vehicle (carla.Actor): The obstacle vehicle
        G (float): Scaling coefficient (default 1.0)
        k (float): Angle sensitivity coefficient (default 1.0)

    Returns:
        float: The risk coefficient at current time t
    """
    ego_loc = ego_vehicle.get_transform().location
    obs_loc = adv_vehicle.get_transform().location
    
    # Compute the distance vector from the obstacle to the ego vehicle
    dx = ego_loc.x - obs_loc.x
    dy = ego_loc.y - obs_loc.y
    d_vec = np.array([dx, dy])
    
    # Euclidean distance (norm of d_vec)
    d_norm = np.linalg.norm(d_vec) + epsilon  # Avoid division by zero

    # Get obstacle velocity vector (only x and y components)
    obs_velocity = adv_vehicle.get_velocity()
    v_vec = np.array([obs_velocity.x, obs_velocity.y])
    speed = np.linalg.norm(v_vec)
    
    # If the obstacle is static, we use a potential field only (exp_factor = 1)
    if speed < epsilon:
        exp_factor = 1.0
    else:
        # Normalize the velocity vector to get the unit direction
        v_unit = v_vec / (speed + epsilon)
        # Normalize the distance vector
        d_unit = d_vec / d_norm
        # Compute cosine of the angle between velocity direction and the distance vector
        cos_theta = np.clip(np.dot(v_unit, d_unit), -1.0, 1.0)
        # The exponential factor inside the DRP formula
        exp_factor = math.exp(k * speed * cos_theta)
    
    # The term d_vec/||d_vec||^2 has a magnitude of 1/||d_vec||
    # Hence, the scalar risk contribution is:
    risk = (1.0 / d_norm) * exp_factor
    
    return G * risk


def calculate_ttc(vehicle1, vehicle2):
    # Extract data from both vehicles
    data1 = get_vehicle_data(vehicle1)
    data2 = get_vehicle_data(vehicle2)

    # Create DataFrame in required format
    df = pd.DataFrame([{
        'x_i': data1['x'],
        'y_i': data1['y'],
        'vx_i': data1['vx'],
        'vy_i': data1['vy'],
        'hx_i': data1['hx'],
        'hy_i': data1['hy'],
        'acc_i': data1['acc'],
        'length_i': data1['length'],
        'width_i': data1['width'],
        'x_j': data2['x'],
        'y_j': data2['y'],
        'vx_j': data2['vx'],
        'vy_j': data2['vy'],
        'hx_j': data2['hx'],
        'hy_j': data2['hy'],
        'acc_j': data2['acc'],
        'length_j': data2['length'],
        'width_j': data2['width'],
    }])

    # TTC calculation
    ttc_result = TTC_DRAC_MTTC(df)

    return ttc_result["TTC"], ttc_result['DRAC'], ttc_result['MTTC']

def calculate_near_miss_ttc(vehicle1, vehicle2):
    # Extract data from both vehicles
    data1 = get_vehicle_data(vehicle1, near_miss=True)
    data2 = get_vehicle_data(vehicle2, near_miss=True)

    # Create DataFrame in required format
    df = pd.DataFrame([{
        'x_i': data1['x'],
        'y_i': data1['y'],
        'vx_i': data1['vx'],
        'vy_i': data1['vy'],
        'hx_i': data1['hx'],
        'hy_i': data1['hy'],
        'acc_i': data1['acc'],
        'length_i': data1['length'],
        'width_i': data1['width'],
        'x_j': data2['x'],
        'y_j': data2['y'],
        'vx_j': data2['vx'],
        'vy_j': data2['vy'],
        'hx_j': data2['hx'],
        'hy_j': data2['hy'],
        'acc_j': data2['acc'],
        'length_j': data2['length'],
        'width_j': data2['width'],
    }])

    # TTC calculation
    ttc_result = TTC_DRAC_MTTC(df)

    return ttc_result["TTC"], ttc_result['DRAC'], ttc_result['MTTC']

def calculate_entry_exit_times(conflict_area, traj, timestep):
    # makes way more sense to calculate this in post processing!!
    inside_prev = False
    entry_time = None
    entry_exit_times = []

    for i, pos in enumerate(traj):
        point = Point(pos)
        inside_now = conflict_area.contains(point)

        if inside_now and not inside_prev:
            entry_time = i * timestep  # Entered zone TODO
        elif not inside_now and inside_prev and entry_time is not None:
            exit_time = i * timestep  # Exited zone
            entry_exit_times.append((entry_time, exit_time))
            entry_time = None

        inside_prev = inside_now

    # Handle case where agent ends inside the zone
    if inside_prev and entry_time is not None:
        exit_time = len(traj) * timestep
        entry_exit_times.append((entry_time, exit_time))

    return entry_exit_times

def calculate_pet(agent1_traj, agent2_traj, conflict_zone_points, timestep_duration):
    """
    Calculates PET (Post-Encroachment Time) between two full trajectories after scenario execution.

    Parameters:
        agent1_traj: list of (x, y) tuples
        agent2_traj: list of (x, y) tuples
        conflict_zone_points: list of (x, y) points defining the conflict zone polygon
        timestep_duration: time per step in seconds

    Returns:
        PET in seconds, or None if no zone conflict occurred, or 0 if vehicles in the zone at the same time
    """
    zone = Polygon(conflict_zone_points)

    a1_times = calculate_entry_exit_times(agent1_traj, zone, timestep_duration)
    a2_times = calculate_entry_exit_times(agent2_traj, zone, timestep_duration)

    if not a1_times and not a2_times:
        return None

    # Check for overlapping intervals (simultaneous presence)
    for a1_entry, a1_exit in a1_times:
        for a2_entry, a2_exit in a2_times:
            if a1_entry < a2_exit and a2_entry < a1_exit:
                return 0

    # Find PETs
    pet_list = []
    for a1_entry, a1_exit in a1_times:
        for a2_entry, _ in a2_times:
            if a2_entry > a1_exit:
                pet = a2_entry - a1_exit
                pet_list.append(pet)
                break

    return min(pet_list) if pet_list else None

def calc_euclidian_distance(vehicle1, vehicle2):
    bbox1 = calc_bounding_box_coordinates(vehicle1)
    bbox2 = calc_bounding_box_coordinates(vehicle2)

    poly1 = Polygon(bbox1)
    poly2 = Polygon(bbox2)
    return poly1.distance(poly2)

def calc_bounding_box_coordinates(vehicle):
    transform = vehicle.get_transform()
    bbox = vehicle.bounding_box

    # Extract heading (convert to radians) and position
    heading_deg = transform.rotation.yaw
    heading_rad = np.radians(heading_deg)
    x = transform.location.x
    y = transform.location.y

    # Half-dimensions of the vehicle
    dx = bbox.extent.x  # Length / 2
    dy = bbox.extent.y  # Width / 2

    cos_theta = np.cos(heading_rad)
    sin_theta = np.sin(heading_rad)

    # Compute the 4 corners using the vehicle's center, heading, and size
    corners = [
        (x + dx * cos_theta - dy * sin_theta, y + dx * sin_theta + dy * cos_theta),  # front-right
        (x + dx * cos_theta + dy * sin_theta, y + dx * sin_theta - dy * cos_theta),  # front-left
        (x - dx * cos_theta + dy * sin_theta, y - dx * sin_theta - dy * cos_theta),  # rear-left
        (x - dx * cos_theta - dy * sin_theta, y - dx * sin_theta + dy * cos_theta),  # rear-right
    ]

    return corners


def plot_stuff(surface_xyz, surface_dir, edges_xyz, markings_xyz, trajectory, idx, string):
    plt.figure(figsize=(10, 10))

    # Plot road surface points (e.g., gray)
    plt.scatter(surface_xyz[:, 0], surface_xyz[:, 1], c='gray', s=2, label='Road Surface')

    # Only plot every Nth arrow for road surface directions
    N = 20  # Change this depending on density
    sampled_idx = np.arange(0, surface_xyz.shape[0], N)
    sampled_surface_xyz = surface_xyz[sampled_idx]
    sampled_surface_dir = surface_dir[sampled_idx, :2]
    
    arrow_scale = 5.0
    plt.quiver(
        sampled_surface_xyz[:, 0], sampled_surface_xyz[:, 1], 
        sampled_surface_dir[:, 0], sampled_surface_dir[:, 1], 
        angles='xy', scale_units='xy', scale=1 / arrow_scale, color='black', width=0.002, alpha=0.7, label='Lane Direction'
    )
    
    # Plot road edges (e.g., red)
    plt.scatter(edges_xyz[:, 0], edges_xyz[:, 1], c='red', s=5, label='Road Edges')
    
    # Plot road markings (e.g., blue)
    plt.scatter(markings_xyz[:, 0], markings_xyz[:, 1], c='blue', s=3, label='Road Markings')
    
    # Plot trajectory as a line (e.g., green)
    traj_xy = trajectory[:, :2]  # Ignore yaw/direction for plotting
    traj_yaw_deg = trajectory[:, 2]
    plt.plot(traj_xy[:, 0], traj_xy[:, 1], c='green', linewidth=2, label='Trajectory')

    # Plot trajectory headings as arrows
    traj_yaw_rad = np.deg2rad(traj_yaw_deg)
    arrow_length = 1.5
    dx = np.cos(traj_yaw_rad) * arrow_length
    dy = np.sin(traj_yaw_rad) * arrow_length

    # Downsample the trajectory points for better readability
    traj_sampled_idx = np.arange(0, len(traj_xy), N)
    sampled_traj_xy = traj_xy[traj_sampled_idx]
    sampled_dx = dx[traj_sampled_idx]
    sampled_dy = dy[traj_sampled_idx]
    
    plt.quiver(sampled_traj_xy[:, 0], sampled_traj_xy[:, 1], sampled_dx, sampled_dy, 
            angles='xy', scale_units='xy', scale=1, color='green', width=0.002, alpha=0.9, label='Trajectory Direction')


    
    plt.legend()
    plt.axis('equal')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Roadgraph and Trajectory Visualization')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"/workspace/random_results/plots_old/{string}_plt{idx}.png")
    plt.close()

def plot_trajectory_vs_network(trajectory, network, invalid_points, idx, invalid_reasons, string):
    """
    Plots the given trajectory against Scenic's Network information.
    
    Args:
        trajectory (list of tuples): [(x, y, yaw), ...] points.
        network (scenic.core.network.Network): Scenic network object.
        invalid_points (list of tuples): [(x, y), ...] invalid trajectory points.
        idx (int): Index for saving plots.
        string (str): File name identifier.
    """

    plt.figure(figsize=(10, 10))

    # --- 1. Plot drivable area (MultiPolygon support) ---
    drivable_area = network.drivableRegion.polygons  # ✅ Corrected from `drivableAreaPolygon`
    
    """if isinstance(drivable_area, shapely.geometry.MultiPolygon):
        for polygon in drivable_area.geoms:  # Iterate over each polygon
            #if polygon.centroid.distance(shapely.geometry.Point(trajectory[0][0], trajectory[0][1])) < 80:
            x, y = polygon.exterior.xy
            plt.fill(x, y, color='red', alpha=0.5, label="Drivable Area" if 'Drivable Area' not in plt.gca().get_legend_handles_labels()[1] else "")
    elif isinstance(drivable_area, shapely.geometry.Polygon):
        #if drivable_area.centroid.distance(shapely.geometry.Point(trajectory[0][0], trajectory[0][1])) < 80:
        x, y = drivable_area.exterior.xy
        plt.fill(x, y, color='red', alpha=0.5, label="Drivable Area")"""

    # --- 2. Plot all lanes and centerlines ---
    for lane in network.lanes:
        if lane.polygon.centroid.distance(shapely.geometry.Point(trajectory[0][0], trajectory[0][1])) < 100:
            # Plot lane boundaries
            x, y = lane.polygon.exterior.xy
            plt.plot(x, y, 'black', linewidth=1, alpha=0.7)  # Lane borders

            # Plot lane centerline
            centerline_x, centerline_y = zip(*lane.centerline.points)
            plt.plot(centerline_x, centerline_y, 'blue', linestyle='dashed', linewidth=1, alpha=0.6, label="Lane Centerline" if lane == network.lanes[0] else "")

    # --- 3. Plot trajectory points ---
    traj_x, traj_y, _ = zip(*trajectory)
    plt.scatter(traj_x, traj_y, c='green', s=20, label="Valid Trajectory")

    # --- 4. Highlight invalid trajectory points ---
    if invalid_points:
        invalid_x, invalid_y = zip(*invalid_points)
        for (x, y, reason) in zip(invalid_x, invalid_y, invalid_reasons):
            plt.scatter(x, y, c='red', s=50, marker='x')
            plt.text(x, y, "", fontsize=9, color='red')


    # --- 5. Formatting & Save ---
    plt.axis("equal")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Trajectory vs Scenic Network")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"/workspace/random_results/plots/{string}_network_plot{idx}.png")
    plt.show()
    plt.close()

def compute_bounding_box_corners(x, y, half_width, half_length, heading_deg):
    """
    Compute the 4 corners of a car's bounding box given its center, size, and heading.

    Args:
        center (tuple): (x, y) center of the car.
        half_width (float): Half of the car's width (left/right from center).
        half_length (float): Half of the car's length (front/back from center).
        heading_deg (float): Heading angle in degrees (0° = east, 90° = north).

    Returns:
        list of tuples: List of 4 (x, y) corner points in order:
                        [front-left, front-right, rear-right, rear-left]
    """
    theta = math.radians(heading_deg)

    # Unit direction vectors
    dx = math.cos(theta)
    dy = math.sin(theta)

    # Perpendicular vectors (to the left)
    perp_dx = -dy
    perp_dy = dx

    # Compute 4 corners
    front_left = (
        x + dx * half_length + perp_dx * half_width,
        y + dy * half_length + perp_dy * half_width
    )
    front_right = (
        x + dx * half_length - perp_dx * half_width,
        y + dy * half_length - perp_dy * half_width
    )
    rear_right = (
        x - dx * half_length - perp_dx * half_width,
        y - dy * half_length - perp_dy * half_width
    )
    rear_left = (
        x - dx * half_length + perp_dx * half_width,
        y - dy * half_length + perp_dy * half_width
    )

    return [front_left, front_right, rear_right, rear_left]

def save_log_file(env, info, parameters, approach, in_odd=True):
    # calc final metrics here:
    filename = f'/workspace/results/{approach}_results/succesfull_runs/data{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    if in_odd:
        ego_traj = env._trajectories["ego_vehicle"]
        adv_traj = env._trajectories["adversary"]
        valid = "valid"
        if all(abs(d["velocity_x"]) < .8 and abs(d["velocity_y"]) < .8 for d in ego_traj):
            valid = "invalid"
            filename = f'/workspace/{approach}_results/aw_didnt_start/data{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    else:
        ego_traj = []
        adv_traj = []
        valid = "invalid by odd"
        filename = f'/workspace/{approach}_results/not_in_odd/data{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    #info["kpis"]["lane_adherence"] = env.score_lane_adherence([[d["x"], d["y"], d["bbox_yaw"]] for d in adv_traj])
    data = {
        "ego_traj": ego_traj,
        "adv_traj": adv_traj,
        "kpis": info["kpis"],
        "valid": valid,
        "parameters": parameters
    }

    with open(filename, 'w') as f:
        json.dump(data, f)