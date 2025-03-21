import dataclasses
from typing import NamedTuple

import carla
import numpy as np
from agents.navigation.basic_agent import BasicAgent
from agents.navigation.controller import VehiclePIDController



class TrajectoryFollowingAgent(BasicAgent):


    @dataclasses.dataclass
    class Waypoint:
        transform: carla.Transform

    def __init__(self, vehicle, trajectory: list[tuple[carla.Location, float]], opt_dict={}, map_inst=None,
                 grp_inst=None):
        self.path, self.velocities = zip(*trajectory)
        super().__init__(vehicle, self.velocities[0], opt_dict, map_inst, grp_inst)
        self._dt = 0.05
        self._current = 0
        self._controller = VehiclePIDController(
            vehicle=self._vehicle,
            args_lateral={'K_P': 1.95, 'K_I': 0.05, 'K_D': 0.2, 'dt': self._dt},
            args_longitudinal={'K_P': 1.0, 'K_I': 0.05, 'K_D': 0, 'dt': self._dt}
        )
        self._max_throt = 0.5
        self._max_brake = 0.3
        self._max_steer = 0.8

    def run_step(self):
        if self._current >= len(self.path) - 1:
            #TODO need to decide whether it should just continue straight or it should brake
            ctrl = carla.VehicleControl()
            ctrl.throttle = 0
            ctrl.brake = 1
            ctrl.steer = 0
            return ctrl

        target_v, target_loc = self.velocities[self._current], self.path[self._current]
        tf = carla.Transform(target_loc, carla.Rotation())
        ctrl = self._controller.run_step(
            waypoint=self.Waypoint(tf),
            target_speed=target_v
        )

        current_loc = self._vehicle.get_location()
        distances = np.array([current_loc.distance(loc) for loc in self.path[self._current:]]) - 3
        self._current += np.abs(distances).argmin()
        self._current = min(self._current, len(self.path))
        return ctrl
