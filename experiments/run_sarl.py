import rbc_gym  # noqa: F401
from rbc_gym.wrappers import (
    RBCNormalizeObservation,
    RBCNormalizeReward,
    RBCRewardShaping,
)

import torch
import multiprocessing as mp

# Reinforcement learning related imports
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation, FrameStackObservation
from rbc_gym.models.CustomNetwork import CustomActorCriticPolicy
from rbc_gym.models.CNN import FluidCNNExtractor
from stable_baselines3.ppo import PPO
from rbc_gym.callbacks.callbacks import RenderCallback
import logging

# Set up logging
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

def main() -> None:

    # env parameters

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

    # PPO/RL parameters

    n_steps = 4  # Number of steps per environment in a rollout (default is 2048 in on_policy_algorithm)
    batch_size = 4  # Minibatch size for each gradient update after a rollout
    n_epochs = 10  # Number of epochs to update the policy for each rollout
    n_envs = 1  # Number of parallel environments TODO UPDATE!!!!
    rollout_buffer_size = n_steps * n_envs  # Size of the rollout buffer, i.e., number of steps that are collected before updating the policy
    stat_window_size = 50  # Size of the window for computing statistics (e.g., mean, std) of the reward
    ent_coef = 0.01  # Entropy coefficient for the loss function

    model = PPO(
        CustomActorCriticPolicy,
        env,
        policy_kwargs=policy_kwargs,
        ent_coef=ent_coef,
        stats_window_size=stat_window_size,
        n_steps=n_steps,  # Number of steps per environment per update given no truncation of the episode? TODO what should this be?
        batch_size=batch_size,  # NOTE TODO We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.
        n_epochs=n_epochs,  # Number of epochs to update the policy for each rollout
        verbose=1,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Summary of all the different "step" attributes:
    # model.n_steps: Number of steps per environment per update (default 2048)
    # `RolloutBuffer` is of size `n_steps * n_envs
    # model.batch_size: Minibatch size for each gradient update after a rollout (default 64)
    # model.n_epochs: Number of epochs to update the policy for each rollout (default 10)
    # BTW: f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
    # total_timesteps, given to model.learn():
        # we already have defined the size of the rolloutbuffer, we already have the batch size. So how many times do we want to collect rollouts?
        # the model.num_timesteps increases in increments of n_envs for each step in the rollout collection.
        # As the check self.num_timsteps < total_timesteps is only done before each rollout collection,
        # total_timesteps is a lower bound for the number of timesteps collected, and can be on the order of 
        # a full rollout collection n_steps * n_envs larger.


    logging.info(
        f"model.n_steps {model.n_steps}, meaning so many steps are taken during rollout collection "
        f"amounting to n_steps * n_envs = {rollout_buffer_size} timesteps in total per rollout."
    )

    # callback = RenderCallback(check_freq=1)

    total_timesteps = rollout_buffer_size * 1  # Total number of timesteps collected in the training process
    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=False,
        # callback=callback
    )

if __name__ == "__main__":
    # On macOS and Windows the default “spawn” start‑method requires this guard.
    # mp.set_start_method("spawn", force=True)
    main()
