import os
import pickle
import random

from metadrive.scenario import ScenarioDescription


class RandomScenarioGenerator:
    def __init__(self, dataset_path: str):
        self._dataset_path = dataset_path
        self._scenario_files = os.listdir(dataset_path)

    def sample(self, num_scenarios: int = 1) -> list[ScenarioDescription] | ScenarioDescription:
        files = random.choices(self._scenario_files, k=num_scenarios)
        self._scenario_files = list(set(self._scenario_files) - set(files))
        descriptions = []
        for file in files:
            with open(os.path.join(self._dataset_path, file), "r") as f:
                desc = pickle.load(f)
                descriptions.append(desc)
        return descriptions[0] if num_scenarios == 1 else descriptions