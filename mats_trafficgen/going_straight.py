import dataclasses
from typing import NamedTuple

import carla
import numpy as np
from agents.navigation.basic_agent import BasicAgent
from agents.navigation.controller import VehiclePIDController



class GoingStraightAgent(BasicAgent):


    @dataclasses.dataclass
    class Waypoint:
        transform: carla.Transform

    def __init__(self):
        pass

    def run_step(self):
        print("i am going!!")
        ctrl = carla.VehicleControl()
        ctrl.throttle = .45
        ctrl.brake = .0
        ctrl.steer = 0
        return ctrl
