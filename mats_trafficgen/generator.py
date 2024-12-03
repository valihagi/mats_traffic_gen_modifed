import carla

from trafficgen.trafficgen.act.model.tg_act import actuator
from trafficgen.trafficgen.init.model.tg_init import initializer


class LevelGenerator:
    def __init__(
            self,
            client: carla.Client,
            init_model_path: str,
            trajectory_model_path: str,
            map_resolution: float = 5.0,
            device: str = "cpu",
    ):
        self._client = client
        self._init_model = initializer.load_from_checkpoint(
            checkpoint_path=init_model_path,
            map_location=device
        )
        self._trajectory_model = actuator.load_from_checkpoint(
            checkpoint_path=trajectory_model_path,
            map_location=device
        )
        world = self._client.get_world()
        self._map_resolution = map_resolution
        town, maps = self._init_maps(world)
        self._map_cache = {
            town: maps
        }
        self._current_maps = maps

    def _place_vehicles(self):
        pass

    def generate(self, world: carla.World):
        return self.level

