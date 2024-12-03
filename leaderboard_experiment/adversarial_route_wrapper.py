import carla
import numpy as np
import optree
import torch
import tensorflow as tf
from agents.navigation.local_planner import RoadOption
from leaderboard.scenarios.route_scenario import RouteScenario
from leaderboard.utils.route_parser import RouteParser
from mats_gym.envs.scenario_env_wrapper import BaseScenarioEnvWrapper
from mats_gym.scenarios.actor_configuration import ActorConfiguration
from pettingzoo.utils.env import AgentID, ObsType, ActionType
from srunner.scenarioconfigs.route_scenario_configuration import RouteScenarioConfiguration
from srunner.scenarioconfigs.scenario_configuration import ScenarioConfiguration
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from cat.advgen.adv_generator import get_polyline_yaw, Intersect, get_polyline_vel
from cat.advgen.adv_utils import process_data
from util import visualize_traj, get_full_trajectory
from map_feature_encoder import MapFeatureEncoder


class AdversarialRouteWrapper(BaseScenarioEnvWrapper):

    def __init__(self, env, model, args):
        super().__init__(env)
        self._args = args
        self._model = model
        self._intersections = []
        self._monitored_agents = []
        self._monitored_traffic_lights = []
        self._trajectories = []
        self._traffic_light_states = []
        self._current_intersection = None
        self._is_monitoring = False

    def _find_intersections_along_route(self, route: list[tuple[carla.Location, RoadOption]]) -> list[int]:
        intersections = []
        map = CarlaDataProvider.get_map()

        for i, (tf, _) in enumerate(route):
            waypoint = map.get_waypoint(tf.location)
            if waypoint.is_junction:
                junction = waypoint.get_junction()
                if len(intersections) == 0 or junction.id != intersections[-1].id:
                    intersections.append(waypoint.get_junction())
        return intersections

    def _trigger_monitoring(self, intersection: carla.Junction, radius: float = 20.0):
        world: carla.World = CarlaDataProvider.get_world()
        intersection_center = intersection.bounding_box.location
        agents = []
        for actor in world.get_actors().filter("vehicle.*"):
            actor_location = actor.get_location()
            if intersection_center.distance(actor_location) < radius:
                agents.append(actor)
        self._monitored_traffic_lights = world.get_traffic_lights_in_junction(intersection.id)
        self._monitored_agents = agents
        self._trajectories = []
        self._traffic_light_states = []
        self._is_monitoring = True

    def _update_monitors(self):
        states = {}
        for agent in self._monitored_agents:
            tf = agent.get_transform()
            velocity = agent.get_velocity()
            states[agent.id] = {
                "transform": tf,
                "velocity": velocity
            }
        self._trajectories.append(states)

        lane_states = {}
        for tl in self._monitored_traffic_lights:
            for wp in tl.get_stop_waypoints():
                lane_id = (wp.road_id, wp.lane_id)
                lane_states[lane_id] = tl.state

        self._traffic_light_states.append(lane_states)


    def step(
        self, actions: dict[AgentID, ActionType]
    ) -> tuple[
        dict[AgentID, ObsType],
        dict[AgentID, float],
        dict[AgentID, bool],
        dict[AgentID, bool],
        dict[AgentID, dict],
    ]:
        obs, rewards, terminated, truncated, info = self.env.step(actions)

        location: carla.Location = self.actors[self.agents[0]].get_location()
        if len(self._intersections) > 0:
            next_intersection: carla.Junction = self._intersections[0]
            if next_intersection.bounding_box.location.distance(location) < 20:
                self._intersections.pop(0)
                self._trigger_monitoring(next_intersection)

        if self._is_monitoring:
            self._update_monitors()

        if len(self._trajectories) == 22:
            self._trajectories = self._trajectories[::2]
            self._traffic_light_states = self._traffic_light_states[::2]
            self._is_monitoring = False
            trajectories = self._generate_adversarial_trajectories()

        return obs, rewards, terminated, truncated, info

    def _generate_adversarial_trajectories(self) -> list:
        features = self._feature_encoder.get_features(
            world=CarlaDataProvider.get_world(),
            map=CarlaDataProvider.get_map(),
            ego_vehicle=self.actors[self.agents[0]],
            actors=self._monitored_agents,
            traffic_lights=self._monitored_traffic_lights,
            trajectories=self._trajectories,
            traffic_light_states=self._traffic_light_states
        )
        ego_trajectory = self._extrapolate_ego_trajectory()
        ego_id = self.actors[self.agents[0]].id
        mask = features["state/id"] == ego_id

        velocities = [(b.transform.location - a.transform.location) * 10 for a, b in zip(ego_trajectory[:-1], ego_trajectory[1:])]
        velocities.append(velocities[-1])
        velocities = np.array([[v.x, v.y] for v in velocities])

        yaws = np.deg2rad(np.array([wp.transform.rotation.yaw for wp in ego_trajectory]))
        yaws[yaws < -2 * np.pi] += 2 * np.pi
        yaws[yaws > 2 * np.pi] -= 2 * np.pi
        ang_velocities = [yaws[i] - yaws[i-1] for i in range(1, len(yaws))]
        ang_velocities.append(ang_velocities[-1])
        ang_velocities = np.array(ang_velocities) * 10

        features["state/future/x"][mask, :] = np.array([wp.transform.location.x for wp in ego_trajectory])
        features["state/future/y"][mask, :] = np.array([wp.transform.location.y for wp in ego_trajectory])
        features["state/future/z"][mask, :] = np.array([wp.transform.location.z for wp in ego_trajectory])
        features["state/future/bbox_yaw"][mask, :] = yaws
        features["state/future/velocity_x"][mask, :] = velocities[:, 0]
        features["state/future/velocity_y"][mask, :] = np.array(velocities[:, 1])
        features["state/future/vel_yaw"][mask, :] = ang_velocities
        features["state/future/width"][mask, :] = features["state/current/width"][mask, 0]
        features["state/future/length"][mask, :] = features["state/current/length"][mask, 0]
        features["state/future/height"][mask, :] = features["state/current/height"][mask, 0]
        features["state/is_sdc"][mask] = 1
        features["tracks_to_predict/track_id"] = np.array([ego_id])
        features["scenario/id"] = np.array(["template"])
        features["state/objects_of_interest"] = features['state/tracks_to_predict'].copy()

        for time in ["past", "current", "future"]:
            x, y = features[f"state/{time}/x"], features[f"state/{time}/y"]
            vx, vy = features[f"state/{time}/velocity_x"], features[f"state/{time}/velocity_y"]

            features[f"state/{time}/x"] = y
            features[f"state/{time}/y"] = x
            features[f"state/{time}/velocity_x"] = vy
            features[f"state/{time}/velocity_y"] = vx

            yaw = features[f"state/{time}/bbox_yaw"]
            ur_quad = np.bitwise_and(yaw >= 0, yaw < np.pi / 2)
            lr_quad = np.bitwise_and(yaw >= np.pi / 2, yaw < np.pi)
            ll_quad = np.bitwise_and(yaw >= -np.pi, yaw < -np.pi / 2)
            ul_quad = np.bitwise_and(yaw >= -np.pi / 2, yaw < 0)

            yaw[ur_quad] = np.pi / 2 - yaw[ur_quad]
            yaw[lr_quad] = np.pi / 2 - yaw[lr_quad]
            yaw[ll_quad] = np.pi / 2 - yaw[ll_quad] - 2 * np.pi
            yaw[ul_quad] = np.pi / 2 - yaw[ul_quad]

            features[f"state/{time}/bbox_yaw"] = yaw
            features[f"state/{time}/vel_yaw"] = -features[f"state/{time}/vel_yaw"]

        for time in ["past", "current"]:
            x, y = features[f"traffic_light_state/{time}/x"], features[
                f"traffic_light_state/{time}/y"]

            features[f"traffic_light_state/{time}/x"] = y
            features[f"traffic_light_state/{time}/y"] = x

        features["roadgraph_samples/xyz"] = features["roadgraph_samples/xyz"][:, [1, 0, 2]]
        features["roadgraph_samples/dir"] = features["roadgraph_samples/dir"][:, [1, 0, 2]]

        def to_tensor(x: np.ndarray):
            if x.dtype == np.float64:
                x = x.astype(np.float32)
            return tf.convert_to_tensor(x)

        data = optree.tree_map(to_tensor, features)
        batch_data = process_data(data, self._args)
        with torch.no_grad():
            pred_trajectory, pred_score, _ = self._model(batch_data[0], 'cpu')
        first_agent = np.argmax(1-mask)
        advs = np.arange(len(mask))[~mask]

        best_score, best_traj = -1, None
        for adv in advs:
            traj_id, adv_traj, score = self._select_adversarial_trajectory(adv, pred_trajectory, pred_score, features)
            if score > best_score:
                best_score = score
                best_traj = adv_traj

        world = CarlaDataProvider.get_world()
        spectator = world.get_spectator()
        ego_loc = self.actors["hero"].get_location()
        spectator.set_transform(carla.Transform(
            carla.Location(ego_loc.x, ego_loc.y, ego_loc.z + 50),
            carla.Rotation(pitch=-90)
        ))
        settings = world.get_settings()
        settings.synchronous_mode = False
        return adv_traj

    def _select_adversarial_trajectory(self, adversary_idx, pred_trajectory, pred_score, features) -> list:
        trajs_OV = pred_trajectory[adversary_idx]
        probs_OV = pred_score[adversary_idx]
        probs_OV[6:] = probs_OV[6]
        probs_OV = np.exp(probs_OV)
        probs_OV = probs_OV / np.sum(probs_OV)
        res = np.zeros(pred_trajectory.shape[1])
        min_dist = np.full(pred_trajectory.shape[1], fill_value=1000000)
        ego_idx = np.argmax(features["state/id"])
        adv_width, adv_length = features["state/current/width"][adversary_idx].item(), features["state/current/length"][adversary_idx].item()
        width, length = features["state/current/width"][ego_idx].item(), features["state/current/length"][ego_idx].item()
        trajs_AV = np.concatenate([
            features["state/future/x"][adversary_idx].reshape(-1, 1),
            features["state/future/y"][adversary_idx].reshape(-1, 1)
        ], axis=1)
        trajs_AV = np.expand_dims(trajs_AV, axis=0)
        for j, prob_OV in enumerate(probs_OV):
            P1 = prob_OV
            traj_OV = trajs_OV[j][::5]
            yaw_OV = get_polyline_yaw(trajs_OV[j])[::5].reshape(-1, 1)
            width_OV = adv_width
            length_OV = adv_length
            cos_theta = np.cos(yaw_OV)
            sin_theta = np.sin(yaw_OV)
            bbox_OV = np.concatenate([
                traj_OV,
                yaw_OV,
                traj_OV[:, 0].reshape(-1,
                                      1) + 0.5 * length_OV * cos_theta + 0.5 * width_OV * sin_theta,
                traj_OV[:, 1].reshape(-1,
                                      1) + 0.5 * length_OV * sin_theta - 0.5 * width_OV * cos_theta,
                traj_OV[:, 0].reshape(-1,
                                      1) + 0.5 * length_OV * cos_theta - 0.5 * width_OV * sin_theta,
                traj_OV[:, 1].reshape(-1,
                                      1) + 0.5 * length_OV * sin_theta + 0.5 * width_OV * cos_theta,
                traj_OV[:, 0].reshape(-1,
                                      1) - 0.5 * length_OV * cos_theta - 0.5 * width_OV * sin_theta,
                traj_OV[:, 1].reshape(-1,
                                      1) - 0.5 * length_OV * sin_theta + 0.5 * width_OV * cos_theta,
                traj_OV[:, 0].reshape(-1,
                                      1) - 0.5 * length_OV * cos_theta + 0.5 * width_OV * sin_theta,
                traj_OV[:, 1].reshape(-1,
                                      1) - 0.5 * length_OV * sin_theta - 0.5 * width_OV * cos_theta
            ], axis=1)

            probs_AV = [1.]
            for i, prob_AV in enumerate(probs_AV):
                P2 = prob_AV
                traj_AV = trajs_AV[i][::5]
                yaw_AV = get_polyline_yaw(trajs_AV[i])[::5].reshape(-1, 1)
                width_AV = width
                length_AV = length
                cos_theta = np.cos(yaw_AV)
                sin_theta = np.sin(yaw_AV)

                bbox_AV = np.concatenate((traj_AV, yaw_AV, \
                                          traj_AV[:, 0].reshape(-1,
                                                                1) + 0.5 * length_AV * cos_theta + 0.5 * width_AV * sin_theta, \
                                          traj_AV[:, 1].reshape(-1,
                                                                1) + 0.5 * length_AV * sin_theta - 0.5 * width_AV * cos_theta, \
                                          traj_AV[:, 0].reshape(-1,
                                                                1) + 0.5 * length_AV * cos_theta - 0.5 * width_AV * sin_theta, \
                                          traj_AV[:, 1].reshape(-1,
                                                                1) + 0.5 * length_AV * sin_theta + 0.5 * width_AV * cos_theta, \
                                          traj_AV[:, 0].reshape(-1,
                                                                1) - 0.5 * length_AV * cos_theta - 0.5 * width_AV * sin_theta, \
                                          traj_AV[:, 1].reshape(-1,
                                                                1) - 0.5 * length_AV * sin_theta + 0.5 * width_AV * cos_theta, \
                                          traj_AV[:, 0].reshape(-1,
                                                                1) - 0.5 * length_AV * cos_theta + 0.5 * width_AV * sin_theta, \
                                          traj_AV[:, 1].reshape(-1,
                                                                1) - 0.5 * length_AV * sin_theta - 0.5 * width_AV * cos_theta),
                                         axis=1)

                P3 = 0
                uncertainty = 1.
                alpha = 0.99
                '''
                B-A  F-E
                | |  | |
                C-D  G-H
                '''
                for (Cx1, Cy1, yaw1, xA, yA, xB, yB, xC, yC, xD, yD), (
                        Cx2, Cy2, yaw2, xE, yE, xF, yF, xG, yG, xH, yH) in zip(bbox_AV, bbox_OV):
                    uncertainty *= alpha
                    ego_adv_dist = np.linalg.norm([Cx1 - Cx2, Cy1 - Cy2])
                    if ego_adv_dist < min_dist[j]:
                        min_dist[j] = ego_adv_dist
                    if ego_adv_dist >= np.linalg.norm(
                            [0.5 * length_AV, 0.5 * width_AV]) + np.linalg.norm(
                        [0.5 * length_OV, 0.5 * width_OV]):
                        pass
                    elif Intersect([xA, yA, xB, yB], [xE, yE, xF, yF]) or Intersect(
                            [xA, yA, xB, yB],
                            [xF, yF, xG, yG]) or \
                            Intersect([xA, yA, xB, yB], [xG, yG, xH, yH]) or Intersect(
                        [xA, yA, xB, yB],
                        [xH, yH, xE, yE]) or \
                            Intersect([xB, yB, xC, yC], [xE, yE, xF, yF]) or Intersect(
                        [xB, yB, xC, yC],
                        [xF, yF, xG, yG]) or \
                            Intersect([xB, yB, xC, yC], [xG, yG, xH, yH]) or Intersect(
                        [xB, yB, xC, yC],
                        [xH, yH, xE, yE]) or \
                            Intersect([xC, yC, xD, yD], [xE, yE, xF, yF]) or Intersect(
                        [xC, yC, xD, yD],
                        [xF, yF, xG, yG]) or \
                            Intersect([xC, yC, xD, yD], [xG, yG, xH, yH]) or Intersect(
                        [xC, yC, xD, yD],
                        [xH, yH, xE, yE]) or \
                            Intersect([xD, yD, xA, yA], [xE, yE, xF, yF]) or Intersect(
                        [xD, yD, xA, yA],
                        [xF, yF, xG, yG]) or \
                            Intersect([xD, yD, xA, yA], [xG, yG, xH, yH]) or Intersect(
                        [xD, yD, xA, yA],
                        [xH, yH, xE, yE]):
                        P3 = uncertainty
                        break

                res[j] += P1 * P2 * P3
        if np.any(res):
            adv_traj_id = np.argmax(res)
            score = res[adv_traj_id]
        else:
            adv_traj_id = np.argmin(min_dist)
            score = 0

        adv_path = trajs_OV[adv_traj_id]
        adv_yaw = get_polyline_yaw(adv_path).reshape(-1, 1)
        adv_vel = np.linalg.norm(get_polyline_vel(adv_path), axis=1).reshape(-1, 1)

        ego_traj = get_full_trajectory(ego_idx, features)[:, ::-1]
        adv_traj_original = get_full_trajectory(adversary_idx, features)[:,
                            ::-1]
        full_adv_traj = get_full_trajectory(adversary_idx, features,
                                            future=adv_path)[:,
                        ::-1]
        ego_traj = np.concatenate([
            ego_traj,
            np.rad2deg(get_polyline_yaw(ego_traj)).reshape(-1, 1)
        ], axis=1)
        full_adv_traj = np.concatenate([
            full_adv_traj,
            np.rad2deg(get_polyline_yaw(full_adv_traj)).reshape(-1, 1)
        ], axis=1)
        adv_traj_original = np.concatenate([
            adv_traj_original,
            np.rad2deg(get_polyline_yaw(adv_traj_original)).reshape(-1, 1)
        ], axis=1)


        visualize_traj(
            ego_traj[4:-4, 0],
            ego_traj[4:-4, 1],
            ego_traj[4:-4, 2],
            width,
            length,
            (0, 5, 0)
        )
        visualize_traj(
            adv_traj_original[4:-4, 0],
            adv_traj_original[4:-4, 1],
            adv_traj_original[4:-4, 2],
            adv_width,
            adv_length,
            (0, 0, 5)
        )
        visualize_traj(
            full_adv_traj[4:-4, 0],
            full_adv_traj[4:-4, 1],
            full_adv_traj[4:-4, 2],
            adv_width,
            adv_length,
            (5, 0, 0)
        )


        adv_traj = np.concatenate([adv_path[:, ::-1], adv_vel, adv_yaw], axis=1)
        return adv_traj_id, adv_traj, score

    def _extrapolate_ego_trajectory(self) -> list:
        route = self.current_scenario.route
        map = CarlaDataProvider.get_map()
        vel = self.actors[self.agents[0]].get_velocity()
        speed = np.linalg.norm([vel.x, vel.y])
        current_tf = self.actors[self.agents[0]].get_transform()
        current_loc = current_tf.location
        length = self.actors[self.agents[0]].bounding_box.extent.x
        point_forward = carla.Location(
            x=current_loc.x + length * np.cos(np.radians(current_tf.rotation.yaw)),
            y=current_loc.y + length * np.sin(np.radians(current_tf.rotation.yaw)),
            z=current_loc.z
        )
        world = CarlaDataProvider.get_world()
        spectator = world.get_spectator()
        spectator_tf = carla.Transform(
            location=carla.Location(x=current_loc.x, y=current_loc.y, z=current_loc.z + 20),
            rotation=carla.Rotation(pitch=-90, yaw=current_tf.rotation.yaw, roll=0)
        )
        spectator.set_transform(spectator_tf)

        world.debug.draw_point(point_forward, size=0.5, color=carla.Color(255, 0, 0), life_time=10000)

        route_pos = np.array([[tf.location.x, tf.location.y] for tf, _ in route])
        pos = np.array([point_forward.x, point_forward.y])
        dists = np.linalg.norm(route_pos - pos, axis=1)
        current_wp_idx = np.argmin(dists)
        dist = speed * 0.1
        trajectory = []

        current_loc = self.actors[self.agents[0]].get_location()
        for t in range(80):
            next_wp, _ = route[min(current_wp_idx, len(route) - 1)]
            i = current_wp_idx
            while i < len(route) and current_loc.distance(next_wp.location) < dist:
                i += 1
                next_wp, _ = route[i]

            distance = current_loc.distance(next_wp.location)
            r = dist / distance
            next_loc = carla.Location(
                x=current_loc.x + r * (next_wp.location.x - current_loc.x),
                y=current_loc.y + r * (next_wp.location.y - current_loc.y),
                z=current_loc.z
            )
            trajectory.append(map.get_waypoint(next_loc))
            current_loc = next_loc
            current_wp_idx = i

            bbox = self.actors[self.agents[0]].bounding_box
            bbox.location = trajectory[-1].transform.location
            bbox.rotation = trajectory[-1].transform.rotation
            world.debug.draw_box(bbox, trajectory[-1].transform.rotation, color=carla.Color(0, 255, 0), life_time=10000)
        world.tick()
        return trajectory





    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[AgentID, ObsType], dict[AgentID, dict]]:
        options = options or {}
        obs, info = self.env.reset(seed=seed, options=options)
        self._intersections = self._find_intersections_along_route(self.current_scenario.route)
        self._feature_encoder = MapFeatureEncoder(map_dir="../scenarios/maps")

        return obs, info

