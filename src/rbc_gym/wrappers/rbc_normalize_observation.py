from typing import Any

import gymnasium as gym
import numpy as np
from rbc_gym.envs import RayleighBenardConvection2DEnv, RayleighBenardConvection3DEnv


class RBCNormalizeObservation(gym.ObservationWrapper):
    """Normalize the observation to image range [0, maxval] with clipping"""

    def __init__(
        self,
        env: RayleighBenardConvection2DEnv | RayleighBenardConvection3DEnv,
        heater_limit: float,
        maxval: int = 1,
        U_limit = 1.3,
        excursion_eps: float = 0.3,  # how many percent of the range to allow excursions outside the min/max values
        clip_obs: bool = False,
    ):
        gym.ObservationWrapper.__init__(self, env)
        self.maxval = maxval
        shape = env.observation_space.shape
        self.excursion_eps = excursion_eps
        self.U_limit = U_limit
        self.observation_space = gym.spaces.Box(
            low=-self.maxval * (1 + excursion_eps),
            high=self.maxval * (1 + excursion_eps),
            shape=shape,
            dtype=np.float32
        )
        self.heater_limit = heater_limit
        self.clip_obs = clip_obs

        # fixed min and max values for each channel
        T = env.unwrapped.temperature_difference
        minT = T[0]
        maxT = T[1] + self.heater_limit
        # TODO investigate empirically if the value below is reasonable for all Ra values, otherwise make a curve fitting based.
        self.min_vals = [minT, -U_limit, -U_limit, -U_limit]
        self.max_vals = [maxT, U_limit, U_limit, U_limit]

    def observation(self, obs) -> Any:
        # Normalize each channel
        for c in range(obs.shape[0]):
            obs[c] = self.maxval * (2 * (obs[c] - self.min_vals[c]) / (self.max_vals[c] - self.min_vals[c]) - 1)
        # at this point obs is in the range [-maxval, maxval], or outside of it if the observation is outside the min/max range
        if self.clip_obs:
            obs = np.clip(obs, -self.maxval, self.maxval)
        if np.any(np.abs(obs) > (1 + self.excursion_eps) * self.maxval):
            max_obs = np.max(np.abs(obs))
            print(f"Warning: observation exceeds maxval {self.maxval}, namely: {max_obs} is the max observed value.")
        return obs