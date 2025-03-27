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
            args_longitudinal={'K_P': 1.0, 'K_I': 0.05, 'K_D': 0.2, 'dt': self._dt}
        )
        self._max_throt = 0.75
        self._max_brake = 0.3
        self._max_steer = 0.8

    def run_step(self):
        if self._current >= len(self.path) - 1:
            #TODO need to decide whether it should just continue straight or it should brake
            ctrl = carla.VehicleControl()
            ctrl.throttle = 0
            ctrl.brake = .2
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
    
import dataclasses
import numpy as np
import carla

class TrajectoryFollowingAgentNew:

    @dataclasses.dataclass
    class Waypoint:
        location: carla.Location
        yaw: float  # in degrees
        speed: float  # m/s

    def __init__(self, vehicle, trajectory: list[tuple[carla.Location, float]], K=2.0, K_soft=1.0):
        self._vehicle = vehicle
        self._dt = 0.05
        self._current = 0
        self.K = K
        self.K_soft = K_soft
        self._max_steer = 0.8
        self._max_throttle = 0.75
        self._max_brake = 0.5

        # Smooth speed tracking
        self._smoothed_speed = 0.0
        self._speed_smooth_alpha = 0.2

        # Process trajectory into waypoints with yaw
        self.waypoints = []
        for i, (loc, speed) in enumerate(trajectory):
            if i < len(trajectory) - 1:
                next_loc = trajectory[i + 1][0]
                dx = next_loc.x - loc.x
                dy = next_loc.y - loc.y
                yaw = np.degrees(np.arctan2(dy, dx))
            else:
                yaw = self.waypoints[-1].yaw if self.waypoints else 0.0
            self.waypoints.append(self.Waypoint(loc, yaw, speed))

    def run_step(self):
        if self._current >= len(self.waypoints):
            return carla.VehicleControl(throttle=0.0, brake=0.3, steer=0.0)

        vehicle_transform = self._vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        vehicle_yaw = vehicle_transform.rotation.yaw
        vel = self._vehicle.get_velocity()
        vehicle_speed = np.linalg.norm([vel.x, vel.y]) # m/s

        # Advance to the closest future waypoint
        while self._current < len(self.waypoints) - 1 and \
                vehicle_location.distance(self.waypoints[self._current].location) < 2.0:
            self._current += 1

        target_wp = self.waypoints[self._current]

        # Lateral control: Stanley method
        heading_error = np.radians(target_wp.yaw - vehicle_yaw)
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))  # normalize to [-π, π]

        # Cross-track error
        dx = target_wp.location.x - vehicle_location.x
        dy = target_wp.location.y - vehicle_location.y
        crosstrack_error = dx * np.sin(np.radians(vehicle_yaw)) - dy * np.cos(np.radians(vehicle_yaw))

        steer = heading_error + np.arctan2(self.K * crosstrack_error, self.K_soft + vehicle_speed)
        steer = np.clip(steer, -self._max_steer, self._max_steer)

        # Longitudinal control: smoothed speed tracking
        target_speed = target_wp.speed
        self._smoothed_speed = (self._speed_smooth_alpha * target_speed +
                                (1 - self._speed_smooth_alpha) * self._smoothed_speed)
        speed_error = self._smoothed_speed - vehicle_speed

        throttle = np.clip(speed_error * 0.5, 0.0, self._max_throttle)
        brake = np.clip(-speed_error * 0.5, 0.0, self._max_brake)

        control = carla.VehicleControl(throttle=throttle, brake=brake, steer=steer)
        return control

