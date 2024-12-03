from __future__ import annotations
param map = localPath('../maps/Town05.xodr')  # or other CARLA map that definitely works
param carla_map = 'Town05'
model scenic.simulators.carla.model

DISTANCE_TO_INTERSECTION = Uniform(5, 10) * -1
NUM_VEHICLES = 2

class RouteFollowingCar(Car):
    route: list[Lane]

def is_4way_intersection(inter) -> bool:
    left_turns = filter(lambda i: i.type == ManeuverType.LEFT_TURN, inter.maneuvers)
    all_single_lane = all(len(lane.adjacentLanes) == 1 for lane in inter.incomingLanes)
    return len(left_turns) >=4 and inter.is4Way

four_way_intersections = filter(is_4way_intersection, network.intersections)
intersection = Uniform(*four_way_intersections)
maneuvers = intersection.maneuvers

maneuver = Uniform(*filter(lambda m: m.type == ManeuverType.LEFT_TURN, maneuvers))
route = [maneuver.startLane, maneuver.connectingLane, maneuver.endLane]

ego = RouteFollowingCar following roadDirection from maneuver.startLane.centerline[-1] for DISTANCE_TO_INTERSECTION,
    with route route,
    with rolename "ego",
    with color Color(0,1,0),
    with blueprint "vehicle.lincoln.mkz_2017"

terminate after 20 seconds