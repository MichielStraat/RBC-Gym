import logging
import gymnasium as gym
from rbc_gym.envs import RayleighBenardConvection2DEnv


class RBCNormalizeReward(gym.RewardWrapper):
    """Normalize the reward to ~[0, 1]"""
    def __init__(self, env: RayleighBenardConvection2DEnv):
        super().__init__(env)

        self.logger = logging.getLogger(__name__)
        self.ra = env.unwrapped.ra
        

    def reward(self, reward):
        return (
            reward + self.__reward_scale()
        ) / self.__reward_scale()
        

    def __reward_scale(self) -> float:
        # Determined by baseline runs -> highest usual Ra
        if self.ra == 5_000:
            return 3.0
        elif self.ra == 10_000:
            return 4.0
        elif self.ra == 50_000:
            return 6.0
        elif self.ra == 100_000:
            return 7.0
        elif self.ra == 500_000:
            return 10.6
        elif self.ra == 1_000_000:
            return 13.0
        elif self.ra == 5_000_000:
            return 20.0
        else:
            self.logger.warning(f"Reward scaling not implemented for Ra={self.ra}. Reward is not normalized.")
            return 1
