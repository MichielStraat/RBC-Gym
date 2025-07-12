import logging
import gymnasium as gym
from rbc_gym.envs import RayleighBenardConvection2DEnv, RayleighBenardConvection3DEnv


class RBCNormalizeReward(gym.RewardWrapper):
    """Normalize the reward to ~[0, 1]"""

    def __init__(self, env):
        """
        The maximum Nusselt number depends on the Rayleigh number with a power law: Nu ~ sRa^a

        Default values:  
        3D: s=0.22, a=0.27  
        2D: s=0.1 , a=0.4
        """
        super().__init__(env)

        ra = env.unwrapped.ra
        if isinstance(env.unwrapped, RayleighBenardConvection2DEnv):
            s = 0.1
            a = 0.4
        elif isinstance(env.unwrapped, RayleighBenardConvection3DEnv):
            s = 0.22
            a = 0.27

        self.scale = s * (ra**a)
        

    def reward(self, reward):
        # NOTE The Nusselt number is a value between [1 and self.scale]
        return (reward + self.scale) / (self.scale - 1)


