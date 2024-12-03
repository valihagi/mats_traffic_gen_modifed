import subprocess
import os
import time
import io
import math
import carla

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
    
