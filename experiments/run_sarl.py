import rbc_gym  # noqa: F401
from rbc_gym.wrappers import (
    RBCNormalizeObservation,
    RBCNormalizeReward,
    RBCRewardShaping,
)
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation, FrameStackObservation
from rbc_gym.models.CustomNetwork import CustomActorCriticPolicy

import multiprocessing as mp

# Reinforcement learning related imports
from rbc_gym.models.CNN import FluidCNNExtractor
from stable_baselines3.ppo import PPO
from rbc_gym.callbacks.callbacks import RenderCallback

def main() -> None:
    env = gym.make(
        "rbc_gym/RayleighBenardConvection3D-v0",
        checkpoint="data/checkpoints/train/3D_ckpt_ra2500.h5",
        render_mode="rgb_array",
        heater_duration=0.375,
        heater_limit=0.9,
        rayleigh_number=2500,
        episode_length=10,
    )
    # Environment wrappers
    env = RBCNormalizeObservation(env, heater_limit=env.unwrapped.heater_limit, maxval=1)
    # env = RBCNormalizeReward(env) # TODO implement reward normalization for 3D for other Ra values, but probably not important right now if you don't use reward shaing
    # env = RBCRewardShaping(env, shaping_weight=0.1)   # TODO implement correctly for 3D, don't use for now
    # env = FlattenObservation(env) # NOTE should not be used with CNN feature extractor
    # env = FrameStackObservation(env, 4) # TODO include later

    policy_kwargs = dict(
        features_extractor_class=FluidCNNExtractor,
        features_extractor_kwargs=dict(features_dim=8*4*8*8),
        share_features_extractor=True,
    )

    model = PPO(
        CustomActorCriticPolicy,
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        # device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # print(model.policy)

    callback = RenderCallback(check_freq=1)

    model.learn(
        total_timesteps=10,
        progress_bar=False,
        callback=callback
    )

if __name__ == "__main__":
    # On macOS and Windows the default “spawn” start‑method requires this guard.
    # mp.set_start_method("spawn", force=True)
    main()
