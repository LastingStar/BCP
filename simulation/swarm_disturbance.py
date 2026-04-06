from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from configs.config import SimulationConfig


@dataclass
class GustEvent:
    start_time_s: float
    end_time_s: float
    vector_xy: np.ndarray


class SwarmDisturbanceManager:
    """Runtime gust/noise helper kept outside the core wind model."""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.rng = np.random.default_rng(int(config.wind_seed) + 4096)
        self.active_event: Optional[GustEvent] = None
        self.history: List[GustEvent] = []

    def reset(self, seed: Optional[int] = None) -> None:
        gust_seed = int(self.config.wind_seed if seed is None else seed) + 4096
        self.rng = np.random.default_rng(gust_seed)
        self.active_event = None
        self.history = []

    def begin_step(self, current_time_s: float, step_dt: float) -> Tuple[np.ndarray, Optional[GustEvent], bool]:
        if not self.config.enable_random_gusts or self.config.gust_max_speed_mps <= 0.0:
            return np.zeros(2, dtype=float), None, False

        if self.active_event is not None and current_time_s < self.active_event.end_time_s:
            return self.active_event.vector_xy.copy(), self.active_event, False

        self.active_event = None
        if self.rng.random() >= self.config.gust_trigger_prob:
            return np.zeros(2, dtype=float), None, False

        vector_xy = self._sample_gust_vector()
        event = GustEvent(
            start_time_s=current_time_s,
            end_time_s=current_time_s + max(step_dt, self.config.gust_duration_s),
            vector_xy=vector_xy,
        )
        self.active_event = event
        self.history.append(event)
        return vector_xy.copy(), event, True

    def apply_obs_noise(self, obs: np.ndarray) -> np.ndarray:
        if self.config.gust_obs_noise_std <= 0.0:
            return obs
        noise = self.rng.normal(0.0, self.config.gust_obs_noise_std, size=obs.shape)
        return (np.asarray(obs, dtype=np.float32) + noise.astype(np.float32)).astype(np.float32)

    def _sample_gust_vector(self) -> np.ndarray:
        magnitude = self.rng.uniform(self.config.gust_min_speed_mps, self.config.gust_max_speed_mps)
        angle = self.rng.uniform(0.0, 2.0 * np.pi)
        return np.array([np.cos(angle), np.sin(angle)], dtype=float) * magnitude
