import carla
import numpy as np
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

def get_full_trajectory(id, features, with_yaw=False, future=None):
    trajs = []
    for time in ["past", "current", "future"]:
        x = features[f"state/{time}/x"][id]
        y = features[f"state/{time}/y"][id]
        traj = [x, y]
        if with_yaw:
            traj.append(features[f"state/{time}/bbox_yaw"][id])
        trajs.append(np.stack(traj, axis=1))
    if future is not None:
        trajs[-1] = future
    return np.concatenate(trajs, axis=0)

def visualize_traj(x, y, yaw, width, length, color, skip=5):
    world = CarlaDataProvider.get_world()
    map = CarlaDataProvider.get_map()
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