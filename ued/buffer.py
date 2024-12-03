import os
import pickle

import numpy as np
from metadrive.scenario import ScenarioDescription


class PrioritizedLevelReplayBuffer:

    def __init__(
            self,
            directory: str,
            max_size: int,
            replay_rate: float,
            p: float,
            temperature: float,
            update_sampler: bool = False,
            load_existing: bool = False,
    ) -> None:
        self._directory = directory
        os.makedirs(self._directory, exist_ok=True)
        self._max_size = max_size
        self._levels = []
        self._scores = []
        self._staleness = []
        self._p = p
        self._replay_rate = replay_rate
        self._temperature = temperature
        self._update_sampler = update_sampler

        if load_existing:
            files = os.listdir(self._directory)
            for file in files:
                with open(os.path.join(self._directory, file), "rb") as f:
                    scenario = pickle.load(f)
                    self.add(scenario, 0)

    def get_buffer_stats(self) -> dict:
        return {
            "size": len(self._levels),
            "mean_p": np.mean(self._p),
            "std_p": np.std(self._p),
            "temperature": self._temperature,
            "mean_score": np.mean(self._scores),
            "std_score": np.std(self._scores),
            "mean_staleness": np.mean(self._staleness),
            "std_staleness": np.std(self._staleness),
        }

    def get_level_stats(self) -> dict:
        score_probs = self._compute_score_prob()
        staleness_probs = self._compute_staleness_prob()
        return {
            level.id: {
                "score": self._scores[i],
                "staleness": self._staleness[i],
                "score_prob": score_probs[i],
                "staleness_prob": staleness_probs[i]
            }
            for i, level in enumerate(self._levels)
        }

    def checkpoint(self) -> dict:
        return {
            "levels": self._levels,
            "scores": self._scores,
            "staleness": self._staleness,
        }

    def add(self, scenario: ScenarioDescription, score: float = 0) -> int:
        if len(self._levels) >= self._max_size:
            min_score_idx = np.argmin(self._scores)
            path = os.path.join(self._directory, f"{min_score_idx}.pkl")
            self._levels[min_score_idx] = path
            self._scores[min_score_idx] = score
            self._staleness[min_score_idx] = 0
            os.remove(path)
            with open(path, "wb") as f:
                pickle.dump(scenario, f)
            return min_score_idx
        else:
            path = os.path.join(self._directory, f"{len(self._levels)}.pkl")
            self._scores.append(score)
            self._staleness.append(0)
            self._levels.append(path)
            new_idx = len(self._levels) - 1
            with open(path, "wb") as f:
                pickle.dump(scenario, f)
            return new_idx

    def sample(self) -> tuple[int, ScenarioDescription]:
        # Compute probabilities
        p_score = self._compute_score_prob()
        c_score = self._compute_staleness_prob()
        probs = (1 - self._p) * p_score + self._p * c_score
        i = np.random.choice(len(self._levels), p=probs)

        # Update staleness
        staleness = np.array(self._staleness)
        staleness[i] = 0
        self._staleness = staleness.tolist()
        return i, self._levels[i]

    def update(self, index: int, score: float) -> None:
        self._scores[index] = score

    def __len__(self):
        return len(self._levels)

    def _compute_score_prob(self) -> np.ndarray:
        scores = np.array(self._scores)
        ranks = np.argsort(scores) + 1
        inv_ranks = 1 / ranks
        probs = np.power(inv_ranks, 1 / self._temperature)
        probs /= probs.sum()
        return probs

    def _compute_staleness_prob(self) -> np.ndarray:
        staleness = np.array(self._staleness)
        if staleness.sum() == 0:
            return np.ones(len(self._levels)) / len(self._levels)
        probs = staleness / staleness.sum()
        return probs