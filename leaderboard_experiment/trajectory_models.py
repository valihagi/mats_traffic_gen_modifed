import numpy as np

from trafficgen.trafficgen.traffic_generator.traffic_generator import TrafficGen



class TrajectoryPredictor:

    def __init__(self, traffic_gen: TrafficGen):
        self._traffic_gen = traffic_gen

    def _preprocess_features(self, road_graph: np.ndarray):
        pass

    def predict(self, road_graph: dict, actors: dict, traffic_lights: dict):


        start = time.time()
        actors, actor_info = self._actor_encoder.encode(self.client)
        print(f"actor encoding time: {time.time() - start}")

        start = time.time()
        traffic_lights = self._traffic_light_encoder.encode(self.client)
        print(f"traffic light encoding time: {time.time() - start}")

        start = time.time()

        # Add ids to lanes and edges
        lane_ids = list(sorted(centerlines.keys()))
        lanes = []
        for i, lane_id in enumerate(lane_ids):
            lane = centerlines[lane_id]
            lane = np.concatenate([lane, np.full((len(lane), 1), i)], axis=1)
            lanes.append(lane)

        lanes = np.concatenate(lanes)
        edges = np.concatenate(
            [np.concatenate([edge, np.full((len(edge), 1), i + len(lanes))], axis=1) for i, edge in
             enumerate(edges)])
        crosswalks = np.concatenate([np.concatenate(
            [crosswalk, np.full((len(crosswalk), 1), i + len(lanes) + len(edges))], axis=1) for
                                     i, crosswalk in enumerate(crosswalks)])

        # Associate traffic lights with lane ids
        traffic_light_features = []
        for lane_id in traffic_lights:
            if lane_id in centerlines:
                idx = lane_ids.index(lane_id)
                feat = np.concatenate([[idx], traffic_lights[lane_id]], axis=0)
                traffic_light_features.append(feat)
        traffic_lights = np.array(traffic_light_features)

        # Add ego to the front of the actor list
        ego_id = self.actors[self.agents[0]].id
        ego_loc = self.actors[self.agents[0]].get_location()
        ego_loc = np.array([ego_loc.x, -ego_loc.y])

        actors = np.array(actors)
        actors = np.concatenate([
            [actors[actor_info.index(ego_id)]],
            actors[:actor_info.index(ego_id)],
            actors[actor_info.index(ego_id) + 1:]
        ])

        # lanes = lanes[np.linalg.norm(lanes[:, :2] - ego_loc, axis=1) < self._max_radius]
        # edges = edges[np.linalg.norm(edges[:, :2] - ego_loc, axis=1) < self._max_radius]
        # crosswalks = crosswalks[np.linalg.norm(crosswalks[:, :2] - ego_loc , axis=1) < self._max_radius]
        # traffic_lights = traffic_lights[np.linalg.norm(traffic_lights[:, 1:3] - ego_loc, axis=1) < self._max_radius]
        self._visualize(lanes, edges, crosswalks, traffic_lights)

        # Transform coordinates
        actors = np.repeat(actors[np.newaxis], 20, axis=0)
        actors[..., 1] = -actors[..., 1]
        actors[..., 4] = -actors[..., 4]

        roadgraph = np.concatenate([lanes, edges, crosswalks], axis=0)
        roadgraph[..., 1] = -roadgraph[..., 1]

        traffic_lights = np.repeat(traffic_lights[np.newaxis], 20, axis=0)
        traffic_lights[..., 2] = -traffic_lights[..., 2]

        scene = {
            "all_agent": actors,  # necessary due to a bug in trafficgen
            "lane": roadgraph[
                np.linalg.norm(roadgraph[:, :2] - ego_loc, axis=1) < self._max_radius],
            "traffic_light": traffic_lights,
            "unsampled_lane": roadgraph[:]
        }
        print(f"formatting time: {time.time() - start}")

        start = time.time()
        processed_scene = process_data_to_internal_format(scene)[0]
        print(f"processing time: {time.time() - start}")
        processed_scene = optree.tree_map(lambda x: np.expand_dims(x, axis=0), processed_scene)

        with torch.no_grad():
            processed_scene["agent_mask"][..., :18] = 1
            data = deepcopy(processed_scene)
            start = time.time()
            if True:
                model_output = self._trafficgen.place_vehicles_for_single_scenario(
                    batch=data,
                    index=0,
                    vis=True,
                    vis_dir="logs/output/vis/scene_initialized",
                    context_num=0
                )
                print(f"trafficgen init time: {time.time() - start}")
                model_output = optree.tree_map(
                    lambda x: x.cpu().numpy() if isinstance(x, torch.Tensor) else x, model_output)
                agent, agent_mask = self.from_list_to_array(model_output['agent'])

                # agents = []
                # world = CarlaDataProvider.get_world()
                # ego_loc = self.actors[self.agents[0]].get_location()
                # ego_yaw = np.deg2rad(self.actors[self.agents[0]].get_transform().rotation.yaw)
                # for actor in world.get_actors().filter("vehicle.*"):
                #     loc = actor.get_location()
                #     yaw = actor.get_transform().rotation.yaw
                #     yaw = np.deg2rad(yaw)
                #     width, length = actor.bounding_box.extent.y * 2, actor.bounding_box.extent.x * 2
                #     x, y = loc.x - ego_loc.x, loc.y - ego_loc.y
                #     x, y = rotate(x, y, -ego_yaw)
                #     yaw = yaw - ego_yaw
                #     if yaw < -np.pi:
                #         yaw += 2 * np.pi
                #     elif yaw > np.pi:
                #         yaw -= 2 * np.pi
                #     #if yaw <= 0:
                #     #    yaw = -yaw
                #     #else:
                #     #    yaw = 2*np.pi - yaw
                #     agents.append([x, y, 0.0, 0.0, yaw, length, width])
                # agent = np.array(agents)
                # agent_mask = np.zeros((max_agents,), dtype=bool)
                # agent_mask[:len(agents)] = True

            output = {}
            output['context_num'] = 0
            output['all_agent'] = agent
            output['agent_mask'] = agent_mask
            output['lane'] = data['other']['unsampled_lane'][0]
            output['unsampled_lane'] = data['other']['unsampled_lane'][0]
            output['traf'] = np.repeat(data['other']['traf'][0], 190, axis=0)
            output['gt_agent'] = data['other']['gt_agent'][0]
            output['gt_agent_mask'] = data['other']['gt_agent_mask'][0]

            start = time.time()
            pred_i = self._trafficgen.inference_control(output, ego_gt=False)
            print(f"trafficgen act time: {time.time() - start}")