import argparse
import logging
import multiprocessing as mp
import yaml
import os
from datetime import datetime
import glob

import rbc_gym  # noqa: F401
from rbc_gym.wrappers import (
    RBCNormalizeObservation,
    RBCNormalizeReward,
    RBCRewardShaping,
)

import torch
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation, FrameStackObservation
from rbc_gym.models.CustomNetwork import CustomActorCriticPolicy
from rbc_gym.models.CNN import FluidCNNExtractor

from stable_baselines3.ppo import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from rbc_gym.callbacks.callbacks import NusseltCallback
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import CheckpointCallback

# Setup logging
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    datestring = datetime.now().strftime('%Y%m%d_%H%M%S')
    parser.add_argument("--output_dir", type=str, default=f"results/run_local_{datestring}", help="Output directory")
    parser.add_argument("--resume_training", action='store_true', help="Resume training from the last checkpoint")
    return parser.parse_args(), datestring


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    args, datestring = parse_args()

    # ----------------------------------------
    # Load config or defaults
    # ----------------------------------------
    if args.config and os.path.isfile(args.config):
        config = load_config(args.config)
        logging.info(f"Loaded config from {args.config}: {config}")
    else:
        # construct a config dictionary with default values
        config = {
            'rl_n_steps': 4,
            'rl_n_envs': 1,
            'rl_batch_size': 4,
            'rl_n_epochs': 10,
            'rl_ent_coef': 0.01,
            'rl_stat_window_size': 50,
            'rl_nr_iterations': 1,
            'rbc_heater_duration': 0.375,
            'rbc_heater_limit': 0.9,
            'rbc_rayleigh_number': 2500,
            'rbc_episode_length': 10
        }
        logging.info(f"No config file provided or file does not exist. Using a default config in script.")

    # Extract parameters from config
    rl_n_steps = config.get('rl_n_steps', 4)
    rl_n_envs = config.get('rl_n_envs', 1)
    rl_batch_size = config.get('rl_batch_size', 4)
    rl_n_epochs = config.get('rl_n_epochs', 10)
    rl_ent_coef = config.get('rl_ent_coef', 0.01)
    rl_stat_window_size = config.get('rl_stat_window_size', 50)
    rl_nr_iterations = config.get('rl_nr_iterations', 1)
    rbc_heater_duration = config.get('rbc_heater_duration', 0.375)
    rbc_heater_limit = config.get('rbc_heater_limit', 0.9)
    rbc_rayleigh_number = config.get('rbc_rayleigh_number', 2500)
    rbc_episode_length = config.get('rbc_episode_length', 10)

    # derived parameters
    rollout_buffer_size = rl_n_steps * rl_n_envs  # Size of the rollout buffer, i.e., number of steps that are collected before updating the policy.
    assert( rollout_buffer_size % rl_batch_size == 0), "It's recommended that rollout_buffer_size be divisible by batch_size"

    os.makedirs(args.output_dir, exist_ok=True)

    # ----------------------------------------
    # Initialize environment
    # ----------------------------------------
    # env = gym.make(
    #     "rbc_gym/RayleighBenardConvection3D-v0",
    #     checkpoint="data/checkpoints/train/3D_ckpt_ra2500.h5",
    #     render_mode="rgb_array",
    #     heater_duration=rbc_heater_duration,
    #     heater_limit=rbc_heater_limit,
    #     rayleigh_number=rbc_rayleigh_number,
    #     episode_length=rbc_episode_length,
    # )

    def wrap_environment(env):
        """Function to wrap the environment with necessary wrappers."""
        env = RBCNormalizeObservation(env, heater_limit=rbc_heater_limit, maxval=1)
        # env = RBCNormalizeReward(env)
        # env = RBCRewardShaping(env, shaping_weight=0.1)
        # env = FlattenObservation(env)
        # env = FrameStackObservation(env, 4)
        return env
    
    vec_env = make_vec_env(
        "rbc_gym/RayleighBenardConvection3D-v0",
        n_envs=rl_n_envs,
        vec_env_cls=SubprocVecEnv,  # Use SubprocVecEnv for parallel environments
        monitor_dir=os.path.join(args.output_dir, 'envs_monitor', 'train'),
        wrapper_class=wrap_environment,
        env_kwargs=dict(
            checkpoint=f"data/checkpoints/train/3D_ckpt_ra{rbc_rayleigh_number}.h5",
            checkpoint_idx=1,  # NOTE temporary use fixed checkpoint for train and val
            render_mode="rgb_array",
            heater_duration=rbc_heater_duration,
            heater_limit=rbc_heater_limit,
            rayleigh_number=rbc_rayleigh_number,
            episode_length=rbc_episode_length,
        ),
    )

    vec_env_eval = make_vec_env(
        "rbc_gym/RayleighBenardConvection3D-v0",
        n_envs=rl_n_envs,
        vec_env_cls=SubprocVecEnv,  # Use SubprocVecEnv for parallel environments
        monitor_dir=os.path.join(args.output_dir, 'envs_monitor', 'eval'),
        wrapper_class=wrap_environment,
        env_kwargs=dict(
            checkpoint=f"data/checkpoints/train/3D_ckpt_ra{rbc_rayleigh_number}.h5",
            checkpoint_idx=1,  # temporary use fixed checkpoint for train and val  
            render_mode="rgb_array",
            heater_duration=rbc_heater_duration,
            heater_limit=rbc_heater_limit,
            rayleigh_number=rbc_rayleigh_number,
            episode_length=rbc_episode_length,
        ),
    )


    policy_kwargs = dict(
        features_extractor_class=FluidCNNExtractor,
        features_extractor_kwargs=dict(features_dim=8 * 4 * 8 * 8),
        share_features_extractor=True,
    )

    model = PPO(
        CustomActorCriticPolicy,
        vec_env,
        policy_kwargs=policy_kwargs,
        ent_coef=rl_ent_coef,
        stats_window_size=rl_stat_window_size,
        n_steps=rl_n_steps,
        batch_size=rl_batch_size,
        n_epochs=rl_n_epochs,
        verbose=1,
        tensorboard_log=args.output_dir,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    logging.info(
        f"model.n_steps: {model.n_steps} and model.n_envs: {model.n_envs} → Rollout buffer: {rollout_buffer_size} timesteps per rollout"
    )

    # ----------------------------------------
    # Initialize W&B
    # ----------------------------------------
    job_id = os.environ.get(
        "SLURM_JOB_ID",
        "local_" + datestring
    )
    wandb.init(
        project="rbc-3D-rl",
        name= f"run_{job_id}",
        config=config,
        sync_tensorboard=True,
        dir=args.output_dir
    )
    wandb.define_metric(
        "rollout/nusselt_mean",
        summary="min",
        step_metric="global_step",
    )
    wandb.define_metric("*", step_metric="global_step")

    # Define callsbacks
    # TODO implement
    wandb_callback = WandbCallback(
        model_save_path=os.path.join(args.output_dir, "models"),
        verbose=2
    )
    eval_callback = EvalCallback(
        vec_env_eval,
        n_eval_episodes=1,
        eval_freq=rl_n_steps,  # Evaluate every rollout  # TODO is this in timesteps?
        deterministic=True,
        log_path=os.path.join(args.output_dir, 'eval'),
        best_model_save_path=os.path.join(args.output_dir, 'models'),
        render=False,  # Render during evaluation
        verbose=2,
        # callback_on_new_best=wandb_callback,  # Log to W&B when a new best model is found,
    )
    nusselt_callback = NusseltCallback()

    model_checkpoint_callback = CheckpointCallback(
        save_freq=4 * rl_n_steps, # every 4 episodes
        save_path=os.path.join(args.output_dir, "models", "checkpoints"),
        name_prefix="rl_model",
        save_replay_buffer=True, # important for offline RL, probably does nothing for online RL because the replay buffer is empty in that case
        save_vecnormalize=True,
    )

    total_timesteps = rollout_buffer_size * rl_nr_iterations
    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=False,
        callback=[wandb_callback, eval_callback, nusselt_callback, model_checkpoint_callback],
    )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
