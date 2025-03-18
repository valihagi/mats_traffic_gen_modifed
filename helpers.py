from datetime import datetime
import random
import subprocess
import os
import time
import io
import math
import carla
import bezier
import numpy as np

from cat.advgen.adv_generator import get_polyline_yaw
import xml.etree.ElementTree as ET
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider


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
    x_ctrl_1, y_ctrl_1 = create_random_control_point(vehicle)
    x_ctrl_2, y_ctrl_2 = create_random_control_point(vehicle)

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

    return trajectory, ego_traj, ego_width, ego_length


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
    max_dist = 55
    transform = vehicle.get_transform()
    x, y = transform.location.x, transform.location.y
    epsilon = 0.1
    angle_offset = random.uniform(-math.pi / 2 + epsilon, math.pi / 2 - epsilon) # epsilon to not allow fully 90 degree
    random_angle = math.radians(transform.rotation.yaw) + angle_offset
    distance = random.uniform(min_dist * max(0.5, math.cos(angle_offset)), max_dist * max(0.5,math.cos(angle_offset)))

    new_x = x + distance * math.cos(random_angle)
    new_y = y + distance * math.sin(random_angle)

    return new_x, new_y


def create_random_control_point(vehicle):
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
    return random_x, random_y
    
