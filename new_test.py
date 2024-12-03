# Python program to demonstrate
# main() function
import argparse
import logging
import pickle
import random
import os


import time

import subprocess


# Defining main function
def main():
    exec_command = ["docker", "exec", "-it", "distracted_pare", "bash", "-c", "source install/setup.bash && ros2 launch carla_autoware_bridge carla_aw_bridge.launch.py port:=2000 passive:=True register_all_sensors:=False"]
    process = subprocess.Popen(
        exec_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        preexec_fn=os.setsid  # Create a new process group so it can be killed later
    )

    for i in range(100):
        print(i)
        output = process.stdout.readline()
        if output == b'':
            print("empty output")
        if output == b'' and process.poll() is not None:
            break
        if output:
            print(output.strip())  # Print the command output in real time
        time.sleep(.2)  # Sleep for a while to avoid busy waiting
        #error_output = process.stderr.readline()
        #if error_output:
            #print("Error:", error_output.strip())

    if input("Press Q if you want to quit") == "q":
        process.terminate()

# Using the special variable 
# __name__
if __name__=="__main__":
    main()