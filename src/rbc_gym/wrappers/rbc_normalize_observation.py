from typing import Any

import gymnasium as gym
import numpy as np
from rbc_gym.envs import RayleighBenardConvection2DEnv


class RBCNormalizeObservation(gym.ObservationWrapper):
    """Normalize the observation to image range [0, maxval] with clipping"""

    def __init__(
        self,
        env: RayleighBenardConvection2DEnv,
        heater_limit: int,
        maxval: int = 1,
    ):
        gym.ObservationWrapper.__init__(self, env)
        self.maxval = maxval
        shape = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=self.maxval, shape=shape)
        self.heater_limit = heater_limit

        # fixed min and max values for each channel
        minT = 1
        maxT = 2 + self.heater_limit
        self.min_vals = [minT, -1.3, -1.3]
        self.max_vals = [maxT, 1.3, 1.3]

    def observation(self, obs) -> Any:
        # Normalize each channel
        for c in range(obs.shape[0]):
            obs[c] = (
                self.maxval
                * (obs[c] - self.min_vals[c])
                / (self.max_vals[c] - self.min_vals[c])
            )
        return np.clip(obs, 0, self.maxval)
