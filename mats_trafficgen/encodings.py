import math
from typing import Any

import carla
import numpy as np
import scenic
from scenic.core.regions import PolylineRegion
from scenic.domains.driving.roads import Network
from scenic.simulators.carla.utils.utils import scenicToCarlaLocation


class LaneEncoder:

    def __init__(self, map: carla.Map, line_resolution: float = 5.0, debug: bool = False):
        self._map = map
        self._line_resolution = line_resolution
        self._debug = debug
        self._topology = map.get_topology()

    def encode(self, client: carla.Client) -> tuple[list[np.ndarray], Any]:
        lanes = []
        infos = []
        for start, _ in self._topology:
            center, widths = [], []
            prev = None
            dist = self._line_resolution
            previous_wps = start.previous(dist)
            while len(previous_wps) > 0 and dist > 0:
                previous_wps = start.previous(dist)
                prev = min(previous_wps, key=lambda wp: wp.transform.location.distance(start.transform.location))
                dist -= 0.5

            wps = start.next_until_lane_end(self._line_resolution)
            if prev is not None:
                wps = [prev, *wps]
            for wp in wps:
                wp: carla.Waypoint
                if wp.lane_type == carla.LaneType.Driving:
                    type = 2
                elif wp.lane_type == carla.LaneType.Biking:
                    type = 3
                else:
                    type = 0
                location = wp.transform.location
                data = np.array([location.x, location.y, type], dtype=np.float32)
                center.append(data)
                widths.append(wp.lane_width)
            lanes.append(np.stack(center, axis=0))
            widths = np.array(widths, dtype=np.float32)
            infos.append({
                "lane_id": (start.road_id, start.lane_id),
                "width": np.array([widths / 2, widths / 2], dtype=np.float32),
            })
        return lanes, infos


class RoadEdgeEncoder:

    def __init__(self, network: Network, line_resolution: float = 5.0, debug: bool = False):
        self._map = map
        self._line_resolution = line_resolution
        self._network = network
        self._debug = debug

    def encode(self, client: carla.Client) -> tuple[list[np.ndarray], Any]:
        world = client.get_world()
        edges = []

        for road in self._network.roads:
            left_edge = self._discretize(
                polyline=road.leftEdge,
                type=15,
                world=world
            )
            right_edge = self._discretize(
                polyline=road.rightEdge,
                type=15,
                world=world
            )
            center_line = self._discretize(
                polyline=road.centerline,
                type=16,
                world=world
            )
            edges.extend([left_edge, right_edge, center_line])
        return edges, None

    def _discretize(self, polyline: PolylineRegion, type: int, world: carla.World):
        line = []
        for point in polyline.pointsSeparatedBy(self._line_resolution):
            loc = scenicToCarlaLocation(point, world=world)
            line.append([loc.x, loc.y, type])
        return np.array(line, dtype=np.float32)


class CrossWalkEncoder:

    def __init__(self, map: carla.Map):
        self._map = map

    def encode(self, client: carla.Client) -> tuple[list[np.ndarray], Any]:
        crosswalks = self._map.get_crosswalks()
        crosswalk_encodings = []
        type = 18
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
            data = np.array([
                [loc.x, loc.y, type]
                for loc in crosswalk
            ], dtype=np.float32)
            crosswalk_encodings.append(data)
        return crosswalk_encodings, None


class TrafficLightEncoder:

    def _get_state(self, traffic_light: carla.TrafficLight) -> int:
        if traffic_light.state == carla.TrafficLightState.Red:
            state = 1
        elif traffic_light.state == carla.TrafficLightState.Yellow:
            state = 2
        elif traffic_light.state == carla.TrafficLightState.Green:
            state = 3
        else:
            state = 0
        return state

    def encode(self, client: carla.Client) -> dict:
        world: carla.World = client.get_world()
        traffic_lights = world.get_actors().filter("traffic.traffic_light")
        dynamic_states = {}
        states = {
            carla.TrafficLightState.Red: 1,
            carla.TrafficLightState.Yellow: 2,
            carla.TrafficLightState.Green: 3,
            carla.TrafficLightState.Off: 0,
        }

        # for t in range(self._num_time_steps):
        #    current_time = t * self._time_step
        #    traffic_light_states = []

        for traffic_light in traffic_lights:
            stop_points: list[carla.Waypoint] = traffic_light.get_stop_waypoints()
            for stop_wp in stop_points:
                stop_loc = stop_wp.transform.location
                lane_id = (stop_wp.road_id, stop_wp.lane_id)
                dynamic_states[lane_id] = np.array([
                    stop_loc.x, stop_loc.y, 0, states[carla.TrafficLightState.Green], 1
                ], dtype=np.float32)
        return dynamic_states


class ActorEncoder:

    def __init__(self):
        pass

    def _get_type(self, actor: carla.Actor) -> int:
        type_id = actor.type_id
        if type_id.startswith("vehicle"):
            if actor.attributes["base_type"] == "bicycle":
                return 3
            else:
                return 1
        elif type_id.startswith("walker"):
            return 2
        return 4

    def encode(self, client: carla.Client) -> tuple[list[np.ndarray], Any]:
        world = client.get_world()
        actors = []
        actors.extend(world.get_actors().filter("vehicle.*"))
        actors.extend(world.get_actors().filter("walker.*"))
        actor_encodings, ids = [], []
        for actor in actors:
            loc = actor.get_location()
            vel = actor.get_velocity()
            heading = math.radians(actor.get_transform().rotation.yaw)
            bbox: carla.BoundingBox = actor.bounding_box.extent
            length, width = bbox.x * 2, bbox.y * 2
            type = self._get_type(actor)
            data = np.array([
                loc.x, loc.y, vel.x, vel.y, heading, length, width, type, True
            ], dtype=np.float32)
            actor_encodings.append(data)
            ids.append(actor.id)
        return actor_encodings, ids
