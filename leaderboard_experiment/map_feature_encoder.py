import enum
from copy import deepcopy

import carla
import numpy as np
from scenic.domains.driving.roads import Network
from scenic.simulators.carla.utils.utils import scenicToCarlaLocation


class RoadGraphTypes(enum.Enum):
    UNKNOWN = 0
    LANE_FREEWAY = 1
    LANE_SURFACE_STREET = 2
    LANE_BIKE_LANE = 3
    ROAD_LINE_BROKEN_SINGLE_WHITE = 6
    ROAD_LINE_SOLID_SINGLE_WHITE = 7
    ROAD_LINE_SOLID_DOUBLE_WHITE = 8
    ROAD_LINE_BROKEN_SINGLE_YELLOW = 9
    ROAD_LINE_BROKEN_DOUBLE_YELLOW = 10
    ROAD_LINE_SOLID_SINGLE_YELLOW = 11
    ROAD_LINE_SOLID_DOUBLE_YELLOW = 12
    ROAD_LINE_PASSING_DOUBLE_YELLOW = 13
    ROAD_EDGE_BOUNDARY = 15
    ROAD_EDGE_MEDIAN = 16
    STOP_SIGN = 17
    CROSSWALK = 18
    SPEED_BUMP = 19


class AgentTypes(enum.Enum):
    VEHICLE = 1
    PEDESTRIAN = 2
    CYCLIST = 3
    OTHERS = 4

    @staticmethod
    def from_carla_type(actor: carla.Actor) -> "AgentTypes":
        if actor.type_id.startswith("vehicle"):
            if actor.attributes["base_type"] == "bicycle":
                return AgentTypes.CYCLIST
            return AgentTypes.VEHICLE
        elif actor.type_id.startswith("walker"):
            return AgentTypes.PEDESTRIAN
        else:
            return AgentTypes.OTHERS


class TrafficLightStates(enum.Enum):
    LANE_STATE_UNKNOWN = 0
    LANE_STATE_ARROW_STOP = 1
    LANE_STATE_ARROW_CAUTION = 2
    LANE_STATE_ARROW_GO = 3
    LANE_STATE_STOP = 4
    LANE_STATE_CAUTION = 5
    LANE_STATE_GO = 6
    LANE_STATE_FLASHING_STOP = 7
    LANE_STATE_FLASHING_CAUTION = 8


class MapFeatureEncoder:

    def __init__(
            self,
            map_dir: str,
            line_resolution: float = 5.0,
            max_samples: int = None,
            max_radius: float = None,
            debug: bool = False
    ):
        self._network = None
        self._map = None
        self._topology = None
        self._debug = debug
        self._cache = None
        self._max_samples = max_samples
        self._max_radius = max_radius
        self._resolution = line_resolution
        self._map_dir = map_dir
        self._traffic_light_states = []

    def reload_map(self, map: carla.Map):
        self._map = map
        town = map.name.split("/")[-1]
        self._topology = map.get_topology()
        self._network = Network.fromFile(f"{self._map_dir}/{town}.xodr", useCache=True)
        self._cache = None

    @property
    def town(self):
        return self._map.name if self._map is not None else None

    def get_features(
            self,
            world: carla.World,
            map: carla.Map,
            ego_vehicle: carla.Actor,
            actors: list[carla.Actor],
            trajectories: list[dict[int, tuple[carla.Transform, carla.Vector3D]]],
            traffic_lights: list[carla.TrafficLight],
            traffic_light_states: list[dict[int, carla.TrafficLightState]],
            num_roadgraph_samples: int = 2000
    ):
        if self.town != map.name:
            self.reload_map(map)
        traffic_light_feats = self.traffic_lights(traffic_lights, traffic_light_states)
        actor_feats = self.actors(actors, trajectories)
        road_graph = self.roadgraph(world, anchor=ego_vehicle.get_location(), num_samples=num_roadgraph_samples)

        lane_ids = road_graph.pop("roadgraph_samples/lane_ids")
        for lane_id in lane_ids:
            mask = traffic_light_feats["traffic_light_state/current/id"] == lane_id
            traffic_light_feats["traffic_light_state/current/id"][mask] = lane_ids.index(lane_id)
            mask = traffic_light_feats["traffic_light_state/past/id"] == lane_id
            traffic_light_feats["traffic_light_state/past/id"][mask] = lane_ids.index(lane_id)

        road_graph.update(traffic_light_feats)
        road_graph.update(actor_feats)
        return road_graph

    def roadgraph(
            self,
            world: carla.World,
            anchor: carla.Location = None,
            num_samples: int = None,
            from_cache: bool = False,
            with_lane_centerlines: bool = True,
            with_road_markings: bool = True,
            with_road_edges: bool = True,
            with_crosswalks: bool = True
    ) -> dict[str, np.ndarray]:

        if from_cache and self._cache is not None:
            return deepcopy(self._cache)

        dir, id, type, valid, xyz = [], [], [], [], []
        num_features = 0
        lanes = self._get_centerlines()
        lane_ids = list(sorted(lanes.keys()))

        elements, ids = [], []
        if with_lane_centerlines:
            elements.extend(lanes[id] for id in lane_ids)
        if with_road_edges:
            elements.extend(self._get_road_edges(world))
        if with_crosswalks:
            elements.extend(self._get_crosswalks())
        if with_road_markings:
            elements.extend(self._get_road_markings())

        for line in elements:
            assert line["xyz"].shape[0] == line["dir"].shape[0]
            xyz.append(line["xyz"])
            type.append(line["type"].reshape(-1, 1))
            valid.append(line["valid"].reshape(-1, 1))
            dir.append(line["dir"])
            length = line["type"].shape[0]
            id.append(np.full([length, 1], num_features, dtype=np.int64))
            num_features += 1

        road_graph = {
            "dir": np.concatenate(dir, axis=0),
            "id": np.concatenate(id, axis=0),
            "type": np.concatenate(type, axis=0),
            "valid": np.concatenate(valid, axis=0),
            "xyz": np.concatenate(xyz, axis=0),
        }
        self._cache = deepcopy(road_graph)

        if anchor is not None and num_samples is not None and road_graph["xyz"].shape[
            0] > num_samples:
            samples = {}
            points = road_graph["xyz"]
            pos = np.array([anchor.x, anchor.y, anchor.z])
            dists = np.linalg.norm(points - pos, axis=1)
            idxs = np.argsort(dists)[:num_samples]
            idxs.sort()
            for k, feats in road_graph.items():
                samples[k] = feats[idxs]

            new_lane_ids = []
            for new_id, old_id in enumerate(sorted(np.unique(samples["id"]))):
                samples["id"][samples["id"] == old_id] = new_id
                if old_id < len(lane_ids):
                    new_lane_ids.append(lane_ids[old_id])
            lane_ids = new_lane_ids
            road_graph = samples

        road_graph["lane_ids"] = lane_ids

        return {f"roadgraph_samples/{k}": v for k, v in road_graph.items()}

    def traffic_lights(
            self,
            traffic_lights: list[carla.TrafficLight],
            traffic_light_states: list[dict]
    ) -> dict[str, np.ndarray]:
        num_lights = len(traffic_light_states[0])
        num_steps = len(traffic_light_states)
        features = {
            'traffic_light_state/current/state': np.full([1, num_lights], -1, dtype=np.int64),
            'traffic_light_state/current/valid': np.full([1, num_lights], 0, dtype=np.int64),
            'traffic_light_state/current/id': np.full([1, num_lights], -1, dtype=np.int64),
            'traffic_light_state/current/x': np.full([1, num_lights], -1, dtype=np.float32),
            'traffic_light_state/current/y': np.full([1, num_lights], -1, dtype=np.float32),
            'traffic_light_state/current/z': np.full([1, num_lights], -1, dtype=np.float32),
            'traffic_light_state/past/state': np.full([num_steps - 1, num_lights], -1,
                                                      dtype=np.int64),
            'traffic_light_state/past/valid': np.full([num_steps - 1, num_lights], 0,
                                                      dtype=np.int64),
            'traffic_light_state/past/x': np.full([num_steps - 1, num_lights], -1,
                                                  dtype=np.float32),
            'traffic_light_state/past/y': np.full([num_steps - 1, num_lights], -1,
                                                  dtype=np.float32),
            'traffic_light_state/past/z': np.full([num_steps - 1, num_lights], -1,
                                                  dtype=np.float32),
            'traffic_light_state/past/id': np.full([num_steps - 1, num_lights], -1, dtype=np.int64),
        }
        lane_ids = list(sorted(traffic_light_states[0].keys()))
        for traffic_light in traffic_lights:
            for wp in traffic_light.get_stop_waypoints():
                for t, states in enumerate(traffic_light_states):
                    state = states[(wp.road_id, wp.lane_id)]
                    if state == carla.TrafficLightState.Red:
                        state = TrafficLightStates.LANE_STATE_STOP
                    elif state == carla.TrafficLightState.Yellow:
                        state = TrafficLightStates.LANE_STATE_CAUTION
                    elif state == carla.TrafficLightState.Green:
                        state = TrafficLightStates.LANE_STATE_GO
                    else:
                        state = TrafficLightStates.LANE_STATE_UNKNOWN
                    loc = wp.transform.location
                    lane_id = 100 * wp.road_id + wp.lane_id + 100
                    time = "past" if t < len(traffic_light_states) - 1 else "current"
                    t = t if t < len(traffic_light_states) - 1 else 0
                    i = lane_ids.index((wp.road_id, wp.lane_id))
                    features[f"traffic_light_state/{time}/state"][t, i] = state.value
                    features[f"traffic_light_state/{time}/valid"][t, i] = 1
                    features[f"traffic_light_state/{time}/id"][t, i] = lane_id
                    features[f"traffic_light_state/{time}/x"][t, i] = loc.x
                    features[f"traffic_light_state/{time}/y"][t, i] = loc.y
                    features[f"traffic_light_state/{time}/z"][t, i] = loc.z
        return features

    def actors(self, actors: list[carla.Actor],
               trajectories: list[dict[int, tuple]]) -> dict[str, np.ndarray]:
        num_actors = len(actors)
        past_length = len(trajectories) - 1
        future_length = 80
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
            'state/past/bbox_yaw': np.full([num_actors, past_length], -1, dtype=np.float32),
            'state/past/height': np.full([num_actors, past_length], -1, dtype=np.float32),
            'state/past/length': np.full([num_actors, past_length], -1, dtype=np.float32),
            'state/past/valid': np.full([num_actors, past_length], 0, dtype=np.int64),
            'state/past/vel_yaw': np.full([num_actors, past_length], -1, dtype=np.float32),
            'state/past/velocity_x': np.full([num_actors, past_length], -1, dtype=np.float32),
            'state/past/velocity_y': np.full([num_actors, past_length], -1, dtype=np.float32),
            'state/past/width': np.full([num_actors, past_length], -1, dtype=np.float32),
            'state/past/x': np.full([num_actors, past_length], -1, dtype=np.float32),
            'state/past/y': np.full([num_actors, past_length], -1, dtype=np.float32),
            'state/past/z': np.full([num_actors, past_length], -1, dtype=np.float32),
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

        for i, actor in enumerate(sorted(actors, key=lambda a: a.id)):
            state_features["state/id"][i] = actor.id
            state_features["state/type"][i] = AgentTypes.from_carla_type(actor).value
            state_features["state/is_sdc"][i] = 0
            state_features["state/tracks_to_predict"][i] = 1
            bbox: carla.BoundingBox = actor.bounding_box.extent
            length, width, height = bbox.x * 2, bbox.y * 2, bbox.z * 2
            for t in range(len(trajectories)):
                states = trajectories[t]
                offset = 0
                if t < past_length:
                    time = "past"
                elif t == past_length:
                    offset = past_length
                    time = "current"
                else:
                    offset = past_length + 1
                    time = "future"

                tf, vel = states[actor.id]["transform"], states[actor.id]["velocity"]
                rotation = tf.rotation
                vel_yaw = actor.get_angular_velocity().z
                state_features[f"state/{time}/bbox_yaw"][i, t - offset] = np.deg2rad(rotation.yaw)
                state_features[f"state/{time}/height"][i, t - offset] = height
                state_features[f"state/{time}/length"][i, t - offset] = length
                state_features[f"state/{time}/width"][i, t - offset] = width
                state_features[f"state/{time}/x"][i, t - offset] = tf.location.x
                state_features[f"state/{time}/y"][i, t - offset] = tf.location.y
                state_features[f"state/{time}/z"][i, t - offset] = tf.location.z
                state_features[f"state/{time}/velocity_x"][i, t - offset] = vel.x
                state_features[f"state/{time}/velocity_y"][i, t - offset] = vel.y
                state_features[f"state/{time}/vel_yaw"][i, t - offset] = np.deg2rad(vel_yaw)
                state_features[f"state/{time}/valid"][i, t - offset] = 1

        return state_features

    def _get_crosswalks(self) -> list[dict[str, np.ndarray]]:
        crosswalks = self._map.get_crosswalks()
        crosswalk_encodings = []
        idx = 0
        while idx < len(crosswalks) - 4:
            start = idx
            idx += 1
            while crosswalks[idx] != crosswalks[start]:
                idx += 1
            end = idx
            idx += 1
            crosswalk = crosswalks[
                        start:end + 1]  # crosswalks are 5 points, first point is repeated at the end
            xyz = np.array([
                [loc.x, loc.y, loc.z + 0.1]
                for loc in crosswalk
            ], dtype=np.float32)
            dir = (xyz[1:] - xyz[:-1]) / np.linalg.norm(xyz[1:] - xyz[:-1], axis=1, keepdims=True)
            dir = np.concatenate([dir, dir[-1:]], axis=0)
            type = np.full((xyz.shape[0],), RoadGraphTypes.CROSSWALK.value, dtype=np.int64)
            valid = np.ones_like(type, dtype=np.int64)
            crosswalk_encodings.append({
                "dir": dir,
                "type": type,
                "valid": valid,
                "xyz": xyz  # last point is repeated
            })

        return crosswalk_encodings

    def _get_traffic_light_states(self, traffic_light: carla.TrafficLight) -> list[
        dict[str, np.ndarray]]:
        tl_state = traffic_light.get_state()
        if tl_state == carla.TrafficLightState.Red:
            tl_state = TrafficLightStates.LANE_STATE_STOP
        elif tl_state == carla.TrafficLightState.Yellow:
            tl_state = TrafficLightStates.LANE_STATE_CAUTION
        elif tl_state == carla.TrafficLightState.Green:
            tl_state = TrafficLightStates.LANE_STATE_GO
        else:
            tl_state = TrafficLightStates.LANE_STATE_UNKNOWN

        states = []
        stop_points: list[carla.Waypoint] = traffic_light.get_stop_waypoints()
        for stop_wp in stop_points:
            stop_loc = stop_wp.transform.location
            lane_id = 100 * stop_wp.road_id + stop_wp.lane_id
            states.append({
                "state": tl_state.value,
                "valid": 1,
                "id": lane_id,
                "x": stop_loc.x,
                "y": stop_loc.y,
                "z": stop_loc.z
            })
        return states

    def _get_centerlines(self) -> dict[tuple, dict[str, np.ndarray]]:
        centerlines = {}
        for start, _ in self._topology:
            types, xyz = [], []
            prev = None
            dist = self._resolution
            previous_wps = start.previous(dist)
            while len(previous_wps) > 0 and dist > 0:
                previous_wps = start.previous(dist)
                prev = min(previous_wps,
                           key=lambda wp: wp.transform.location.distance(start.transform.location))
                dist -= 0.5

            wps = start.next_until_lane_end(self._resolution)
            if prev is not None:
                wps = [prev, *wps]
            for wp in wps:
                if wp.lane_type == carla.LaneType.Driving:
                    type = RoadGraphTypes.LANE_SURFACE_STREET
                elif wp.lane_type == carla.LaneType.Biking:
                    type = RoadGraphTypes.LANE_BIKE_LANE
                else:
                    type = RoadGraphTypes.UNKNOWN
                location = wp.transform.location
                xyz.append([location.x, location.y, location.z])
                types.append(type.value)
            xyz = np.array(xyz, dtype=np.float32)
            types = np.array(types, dtype=np.int64)
            valid = np.ones_like(types, dtype=np.int64)
            dir = (xyz[1:] - xyz[:-1]) / (
                    np.linalg.norm(xyz[1:] - xyz[:-1], axis=1, keepdims=True) + 1e-6)
            carla.Rotation()
            dir = np.concatenate([dir, dir[-1:]], axis=0)
            id = 100 * start.road_id + start.lane_id + 100
            centerlines[id] = {
                "dir": dir,
                "type": types,
                "valid": valid,
                "xyz": xyz
            }
        return centerlines

    def _get_road_markings(self) -> list[dict[str, np.ndarray]]:
        def get_type(type, color) -> RoadGraphTypes:
            rg_type = RoadGraphTypes.UNKNOWN
            if type == carla.LaneMarkingType.Broken:
                if color == carla.LaneMarkingColor.White:
                    rg_type = RoadGraphTypes.ROAD_LINE_BROKEN_SINGLE_WHITE
                elif color == carla.LaneMarkingColor.Yellow:
                    rg_type = RoadGraphTypes.ROAD_LINE_BROKEN_SINGLE_YELLOW
            elif type == carla.LaneMarkingType.Solid:
                if color == carla.LaneMarkingColor.White:
                    rg_type = RoadGraphTypes.ROAD_LINE_SOLID_SINGLE_WHITE
                elif color == carla.LaneMarkingColor.Yellow:
                    rg_type = RoadGraphTypes.ROAD_LINE_SOLID_SINGLE_YELLOW
            elif type == carla.LaneMarkingType.BrokenBroken:
                if color == carla.LaneMarkingColor.Yellow:
                    rg_type = RoadGraphTypes.ROAD_LINE_BROKEN_DOUBLE_YELLOW
            elif type == carla.LaneMarkingType.SolidSolid:
                if color == carla.LaneMarkingColor.White:
                    rg_type = RoadGraphTypes.ROAD_LINE_SOLID_DOUBLE_WHITE
                elif color == carla.LaneMarkingColor.Yellow:
                    rg_type = RoadGraphTypes.ROAD_LINE_SOLID_DOUBLE_YELLOW
            return rg_type

        def extract_line(line: list[carla.Waypoint]):
            return np.stack([
                [wp.transform.location.x, wp.transform.location.y, wp.transform.location.z]
                for wp in line
            ])

        def parallel_line(line: list[carla.Waypoint], left: bool):
            widths = np.array([wp.lane_width for wp in line])
            line = extract_line(line)
            dir = (line[1:] - line[:-1]) / np.linalg.norm(line[1:] - line[:-1], axis=1,
                                                          keepdims=True)
            dir = np.concatenate([dir, dir[-1:]], axis=0)
            angle = np.arctan2(dir[:, 1], dir[:, 0])
            offset = widths * (1 if left else -1) * 0.5
            shift = np.stack([
                offset * np.sin(angle),
                offset * -np.cos(angle),
                np.zeros_like(angle)
            ], axis=1)
            return line + shift, dir

        markings = []
        lanes = []
        # for road in self._network.roads:
        #     lanes.extend(road.lanes)
        # for intersection in self._network.intersections:
        #     for maneuver in filter(lambda m: m.type == ManeuverType.STRAIGHT, intersection.maneuvers):
        #         if maneuver.connectingLane is not None:
        #             lanes.append(maneuver.connectingLane)
        # lanes = [
        #    [self._map.get_waypoint(scenicToCarlaLocation(p, world=world)) for p in lane]
        #    for lane in lanes
        # ]

        for start, _ in self._topology:
            lanes.append([
                wp for wp in start.next_until_lane_end(self._resolution)
                if not wp.is_intersection
            ])

        for lane in lanes:
            centerline = lane
            if len(centerline) < 2:
                continue

            left = [
                wp for wp in centerline
                if wp.left_lane_marking.type != carla.LaneMarkingType.NONE
            ]

            right = [
                wp for wp in centerline
                if wp.right_lane_marking.type != carla.LaneMarkingType.NONE
            ]

            if len(left) > 1:
                xyz, dir = parallel_line(left, True)
                types = np.array([
                    get_type(wp.left_lane_marking.type, wp.left_lane_marking.color).value
                    for wp in left
                ], dtype=np.int64)
                markings.append({
                    "dir": dir,
                    "type": types,
                    "valid": np.ones_like(types, dtype=np.int64),
                    "xyz": xyz
                })

            if len(right) > 1:
                xyz, dir = parallel_line(right, False)
                types = np.array([
                    get_type(wp.right_lane_marking.type, wp.right_lane_marking.color).value
                    for wp in right
                ], dtype=np.int64)
                markings.append({
                    "dir": dir,
                    "type": types,
                    "valid": np.ones_like(types, dtype=np.int64),
                    "xyz": xyz
                })
        return markings

    def _get_road_edges(self, world: carla.World) -> list[dict[str, np.ndarray]]:
        lines = []
        for road in self._network.roads:
            edges = [
                (road.leftEdge, RoadGraphTypes.ROAD_EDGE_BOUNDARY),
                (road.rightEdge, RoadGraphTypes.ROAD_EDGE_BOUNDARY),
                (road.centerline, RoadGraphTypes.ROAD_EDGE_MEDIAN)
            ]
            for edge, type in edges:
                line = []
                for point in edge.pointsSeparatedBy(self._resolution):
                    loc = scenicToCarlaLocation(point, world=world)
                    if len(line) > 0 and abs(line[-1][2] - loc.z) > 0.5:
                        loc.z = line[-1][2]
                    line.append([loc.x, loc.y, loc.z])
                if len(line) < 2:
                    continue
                xyz = np.array(line, dtype=np.float32)
                type = np.full([xyz.shape[0], 1], type.value, dtype=np.int64)
                valid = np.ones_like(type, dtype=np.int64)
                dir = (xyz[1:] - xyz[:-1]) / np.linalg.norm(xyz[1:] - xyz[:-1], axis=1,
                                                            keepdims=True)
                dir = np.concatenate([dir, dir[-1:]], axis=0)
                lines.append({
                    "dir": dir,
                    "type": type,
                    "valid": valid,
                    "xyz": xyz
                })
        return lines
