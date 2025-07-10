from __future__ import annotations
param map = localPath('../maps/Town02.xodr')  # or other CARLA map that definitely works
param carla_map = 'Town02'
model scenic.simulators.carla.model

DISTANCE_TO_INTERSECTION = Uniform(10, 10) * -1
DISTANCE_TO_INTERSECTION_ADV = Uniform(10, 10) * -1
NUM_VEHICLES = 2

class RouteFollowingCar(Car):
    route: list[Lane]

def is_3way_intersection(inter) -> bool:
    left_turns = filter(lambda i: i.type == ManeuverType.LEFT_TURN, inter.maneuvers)
    all_single_lane = all(len(lane.adjacentLanes) == 1 for lane in inter.incomingLanes)
    return len(left_turns) >=3 and inter.is3Way


intersection = Uniform(*network.intersections)
maneuvers = intersection.maneuvers

maneuver = Uniform(*filter(lambda m: m.type == ManeuverType.LEFT_TURN, maneuvers))
route = [maneuver.startLane, maneuver.connectingLane, maneuver.endLane]
ego = RouteFollowingCar following roadDirection from maneuver.startLane.centerline[-1] for DISTANCE_TO_INTERSECTION,
    with route route,
    with rolename f"ego_vehicle",
    with color Color(0,1,0),
    with blueprint "vehicle.volkswagen.t2_2021"

adv_maneuver = Uniform(*maneuver.conflictingManeuvers)
adv_route = [adv_maneuver.startLane, adv_maneuver.connectingLane, adv_maneuver.endLane]
adv = RouteFollowingCar following roadDirection from adv_maneuver.startLane.centerline[-1] for DISTANCE_TO_INTERSECTION_ADV,
    with route adv_route,
    with rolename f"adversary",
    with color Color(1,0,0),
    with blueprint "vehicle.lincoln.mkz_2017"
    






terminate after 20 seconds
