import carla
import numpy as np
import torch

from cat.advgen.adv_utils import process_data
from cat.advgen.modeling.vectornet import VectorNet
from mats_trafficgen.map_features import MapFeatures, TrafficLightStates, AgentTypes
import tensorflow as tf
import optree

class TrajectorySampler:
    def __init__(
            self,
            traffic_model: VectorNet,
            args,
            map_features: MapFeatures,
            max_samples: int = 20000,
            max_radius: float = 50.0,
            line_resolution: float = 5.0,
            num_time_steps: int = 91,
            time_step: float = 0.1,
            history_length: int = 10,
    ):

        self._traffic_model = traffic_model
        self._args = args
        self._max_samples = max_samples
        self._max_radius = max_radius
        self._line_resolution = line_resolution
        self._num_time_steps = num_time_steps
        self._time_step = time_step
        self._map_features = map_features
        self._history_length = history_length

    def _encode_traffic_light_states(
            self,
            traffic_lights: dict[int, carla.TrafficLight],
            states: dict[int, list[carla.TrafficLightState]],
            lane_ids: list[int]
    ) -> dict[str, np.ndarray]:
        ids = sorted(traffic_lights.keys())
        num_steps = len(states[ids[0]])
        num_features = sum([len(traffic_lights[tl_id].get_stop_waypoints()) for tl_id in ids])
        traffic_light_features = {
            "state": np.zeros((num_steps, num_features), dtype=np.int64),
            "valid": np.zeros((num_steps, num_features), dtype=np.int64),
            "id": np.zeros((num_steps, num_features), dtype=np.int64),
            "x": np.zeros((num_steps, num_features), dtype=np.float32),
            "y": np.zeros((num_steps, num_features), dtype=np.float32),
            "z": np.zeros((num_steps, num_features), dtype=np.float32)
        }

        ids = sorted(traffic_lights.keys())
        num_steps = len(states[ids[0]])
        for t in range(num_steps):
            i = 0
            for _, tl_id in enumerate(ids):
                state = states[tl_id][t]
                if state == carla.TrafficLightState.Red:
                    state = TrafficLightStates.LANE_STATE_STOP
                elif state == carla.TrafficLightState.Yellow:
                    state = TrafficLightStates.LANE_STATE_CAUTION
                elif state == carla.TrafficLightState.Green:
                    state = TrafficLightStates.LANE_STATE_GO
                else:
                    state = TrafficLightStates.LANE_STATE_UNKNOWN
                stop_points: list[carla.Waypoint] = traffic_lights[tl_id].get_stop_waypoints()
                for stop_wp in stop_points:
                    stop_loc = stop_wp.transform.location
                    lane_id = 100 * stop_wp.road_id + stop_wp.lane_id
                    if lane_id in lane_ids:
                        traffic_light_features["id"][t, i] = lane_ids.index(lane_id)
                        traffic_light_features["valid"][t, i] = 1
                    else:
                        traffic_light_features["id"][t, i] = -1
                        traffic_light_features["valid"][t, i] = 0
                    traffic_light_features["state"][t, i] = state.value
                    traffic_light_features["x"][t, i] = stop_loc.y # x and y are swapped
                    traffic_light_features["y"][t, i] = stop_loc.x
                    traffic_light_features["z"][t, i] = stop_loc.z
                    i += 1
        current = {
            f"traffic_light_state/current/{k}": v[-1] for k, v in traffic_light_features.items()
        }
        past = {
            f"traffic_light_state/past/{k}": v[:-1] for k, v in traffic_light_features.items()
        }
        return {
            **current,
            **past
        }

    def _transform_yaw(self, yaw: np.ndarray) -> np.ndarray:
        ur_quad = np.bitwise_and(yaw >= 0, yaw < np.pi / 2)
        lr_quad = np.bitwise_and(yaw >= np.pi / 2, yaw < np.pi)
        ll_quad = np.bitwise_and(yaw >= -np.pi, yaw < -np.pi / 2)
        ul_quad = np.bitwise_and(yaw >= -np.pi / 2, yaw < 0)
        yaw[ur_quad] = np.pi / 2 - yaw[ur_quad]
        yaw[lr_quad] = np.pi / 2 - yaw[lr_quad]
        yaw[ll_quad] = np.pi / 2 - yaw[ll_quad] - 2 * np.pi
        yaw[ul_quad] = np.pi / 2 - yaw[ul_quad]
        return yaw

    def _encode_agent_states(
            self,
            history_length: int,
            ego_id: int,
            actors: dict[int, carla.Vehicle],
            trajectories: dict[int, list[tuple[carla.Transform, carla.Vector3D]]],
            max_agents: int = 32
    ) -> dict[str, np.ndarray]:
        num_actors = len(actors) if len(actors) < max_agents else max_agents
        trajectory_length = len(trajectories[ego_id])
        future_length = trajectory_length - history_length - 1
        state_features = {
            "state/id": np.full([num_actors, ], -1, dtype=np.int64),
            "state/type": np.full([num_actors, ], 0, dtype=np.int64),
            "state/is_sdc": np.full([num_actors, ], 0, dtype=np.int64),
            'state/tracks_to_predict': np.full([num_actors, ], 0, dtype=np.int64),
            'state/current/bbox_yaw': np.full([num_actors, 1], -1, dtype=np.float32),
            'state/current/height': np.full([num_actors, 1], -1, dtype=np.float32),
            'state/current/length': np.full([num_actors, 1], -1, dtype=np.float32),
            'state/current/valid': np.full([num_actors, 1], 0, dtype=np.int64),
            'state/current/vel_yaw': np.full([num_actors, 1], -1, dtype=np.float32),
            'state/current/velocity_x': np.full([num_actors, 1], -1, dtype=np.float32),
            'state/current/velocity_y': np.full([num_actors, 1], -1, dtype=np.float32),
            'state/current/width': np.full([num_actors, 1], -1, dtype=np.float32),
            'state/current/x': np.full([num_actors, 1], -1, dtype=np.float32),
            'state/current/y': np.full([num_actors, 1], -1, dtype=np.float32),
            'state/current/z': np.full([num_actors, 1], -1, dtype=np.float32),
            'state/past/bbox_yaw': np.full([num_actors, history_length], -1, dtype=np.float32),
            'state/past/height': np.full([num_actors, history_length], -1, dtype=np.float32),
            'state/past/length': np.full([num_actors, history_length], -1, dtype=np.float32),
            'state/past/valid': np.full([num_actors, history_length], 0, dtype=np.int64),
            'state/past/vel_yaw': np.full([num_actors, history_length], -1, dtype=np.float32),
            'state/past/velocity_x': np.full([num_actors, history_length], -1, dtype=np.float32),
            'state/past/velocity_y': np.full([num_actors, history_length], -1, dtype=np.float32),
            'state/past/width': np.full([num_actors, history_length], -1, dtype=np.float32),
            'state/past/x': np.full([num_actors, history_length], -1, dtype=np.float32),
            'state/past/y': np.full([num_actors, history_length], -1, dtype=np.float32),
            'state/past/z': np.full([num_actors, history_length], -1, dtype=np.float32),
            'state/future/bbox_yaw': np.full([num_actors, future_length], -1, dtype=np.float32),
            'state/future/height': np.full([num_actors, future_length], -1, dtype=np.float32),
            'state/future/length': np.full([num_actors, future_length], -1, dtype=np.float32),
            'state/future/valid': np.full([num_actors, future_length], 0, dtype=np.int64),
            'state/future/vel_yaw': np.full([num_actors, future_length], -1, dtype=np.float32),
            'state/future/velocity_x': np.full([num_actors, future_length], -1, dtype=np.float32),
            'state/future/velocity_y': np.full([num_actors, future_length], -1, dtype=np.float32),
            'state/future/width': np.full([num_actors, future_length], -1, dtype=np.float32),
            'state/future/x': np.full([num_actors, future_length], -1, dtype=np.float32),
            'state/future/y': np.full([num_actors, future_length], -1, dtype=np.float32),
            'state/future/z': np.full([num_actors, future_length], -1, dtype=np.float32)
        }

        for i, (id, actor) in enumerate(sorted(actors.items(), key=lambda x: x[0])):
            bbox: carla.BoundingBox = actor.bounding_box.extent
            length, width, height = bbox.x * 2, bbox.y * 2, bbox.z * 2
            state_features["state/id"][i] = id
            state_features["state/type"][i] = AgentTypes.from_carla_type(actor).value
            state_features["state/is_sdc"][i] = 1 if id == ego_id else 0
            state_features["state/tracks_to_predict"][i] = 1 if id == ego_id else 0
            for t, (transform, velocity) in enumerate(trajectories[id]):
                if t < history_length:
                    time = "past"
                elif t == history_length:
                    time = "current"
                    t = t - history_length
                else:
                    time = "future"
                    t = t - history_length - 1

                loc = transform.location
                vel = velocity
                rotation = transform.rotation
                yaw = self._transform_yaw(np.array(np.deg2rad(rotation.yaw))) # yaw is inverted
                state_features[f"state/{time}/length"][i, t] = length
                state_features[f"state/{time}/width"][i, t] = width
                state_features[f"state/{time}/height"][i, t] = height
                state_features[f"state/{time}/x"][i, t] = loc.y # x and y are swapped
                state_features[f"state/{time}/y"][i, t] = loc.x
                state_features[f"state/{time}/z"][i, t] = loc.z
                state_features[f"state/{time}/velocity_x"][i, t] = vel.y
                state_features[f"state/{time}/velocity_y"][i, t] = vel.x
                state_features[f"state/{time}/bbox_yaw"][i, t] = yaw.item()
                state_features[f"state/{time}/vel_yaw"][i, t] = np.arctan2(vel.y, vel.x)
                state_features[f"state/{time}/valid"][i, t] = 1

        return state_features


    def sample(
            self,
            world: carla.World,
            actor: carla.Vehicle,
            max_traffic_lights: int = 16,
            dynamic_map_features: dict[int, list[carla.TrafficLightState]] | None = None,
            trajectories: dict[int, list[tuple[carla.Transform, carla.Vector3D]]] | None = None
    ) -> list[tuple[carla.Transform, float]]:

        # get roadgraph and swap x and y
        roadgraph = self._map_features.roadgraph(
            world=world,
            anchor=actor.get_location(),
            num_samples=self._max_samples
        )
        lane_ids = roadgraph.pop("lane_ids")
        roadgraph["xyz"] = roadgraph["xyz"][:, [1, 0, 2]]
        roadgraph["dir"] = roadgraph["dir"][:, [1, 0, 2]]

        # encode traffic light states
        if not dynamic_map_features:
            traffic_lights = {tl.id: tl for tl in world.get_actors().filter("traffic.traffic_light")}
            dynamic_map_features = {
                id: [carla.TrafficLightState.Green for _ in range(self._history_length + 1)] for id in traffic_lights
            }
        else:
            traffic_lights = {
                id: world.get_actor(id) for id in dynamic_map_features
            }
        if len(traffic_lights) > max_traffic_lights:
            actor_loc = actor.get_location()
            traffic_lights = list(sorted(traffic_lights.items(), key=lambda x: actor_loc.distance(x[1].get_location())))
            traffic_lights = {id: tl for id, tl in traffic_lights[:max_traffic_lights]}
            dynamic_map_features = {id: dynamic_map_features[id] for id in traffic_lights}
        dynamic_map_features = self._encode_traffic_light_states(
            traffic_lights=traffic_lights,
            states=dynamic_map_features,
            lane_ids=lane_ids
        )

        # encode agent states
        if not trajectories:
            trajectories = {
                actor.id: [(actor.get_transform(), actor.get_velocity()) for _ in range(self._num_time_steps)]
            }

        actors = {
            id: world.get_actor(id) for id in trajectories
        }
        agent_features = self._encode_agent_states(
            ego_id=actor.id,
            history_length=self._history_length,
            actors=actors,
            trajectories=trajectories
        )

        features = {f"roadgraph_samples/{k}": v for k, v in roadgraph.items()}
        features.update(dynamic_map_features)
        features.update(agent_features)
        features["scenario/id"] = np.array(["template"])
        features["state/objects_of_interest"] = features['state/tracks_to_predict'].copy()

        # process features
        def to_tensor(x: np.ndarray):
            if x.dtype == np.float64:
                x = x.astype(np.float32)
            return tf.convert_to_tensor(x)

        data = optree.tree_map(to_tensor, features)
        batch_data = process_data(data, self._args)
        with torch.no_grad():
            pred_trajectory, pred_score, indices = self._traffic_model(batch_data[0], 'cpu')
        goals = batch_data[0][0]["goals_2D"][indices[0]]
        return pred_trajectory[0], pred_score[0], goals
