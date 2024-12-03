import carla
import numpy as np
import scenic
from mats_gym.scenarios import ScenicScenarioConfiguration
from mats_gym.scenarios.actor_configuration import ActorConfiguration
from scenic.core.scenarios import Scene
from srunner.scenarioconfigs.scenario_configuration import ScenarioConfiguration, ActorConfigurationData


class ScenarioConfig:
    trajectories: dict[str, np.ndarray]
    actors: list[ActorConfigurationData]
    town: str


class LevelGenerator:

    def __init__(self, scenario: str, sut_name: str = "agent_0", seed: int = 0):
        self._seed = seed
        self._timestep = 0.1
        self._max_time_steps = 1000
        self._traffic_manager_port = 8000
        self._sut_name = sut_name
        self._scenario = scenic.scenarioFromFile(scenario, model="scenic.simulators.carla.model")

    def __call__(self) -> ScenarioConfiguration:
        return self._build_config(self._scenario)

    def _build_config(self, scene: Scene) -> ScenarioConfiguration:
        ego_vehicles = []
        for object in scene.objects:
            if object.isVehicle:
                actor_config = ActorConfiguration(
                    model=object.blueprint,
                    rolename=object.rolename,
                    transform=None,
                    route=object.route if hasattr(object, "route") else None,
                )
                ego_vehicles.append(actor_config)

        config = ScenicScenarioConfiguration(
            scene=scene,
            town=scene.params["carla_map"],
            ego_vehicles=ego_vehicles,
            seed=self._seed,
            timestep=self._timestep,
            max_time_steps=self._max_time_steps,
            traffic_manager_port=self._traffic_manager_port,
        )
        return config