param map = localPath('../maps/Town05.xodr')  # or other CARLA map that definitely works
param carla_map = 'Town05'
model scenic.simulators.carla.model

lane = Uniform(*network.lanes)
spawn_point = OrientedPoint in lane.centerline
ego = Car at spawn_point,
    with rolename "ego",
    with color Color(0,1,0),
    with blueprint "vehicle.lincoln.mkz_2017"

terminate after 20 seconds