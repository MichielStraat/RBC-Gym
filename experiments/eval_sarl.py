import rbc_gym  # noqa: F401
import vtk
import gymnasium as gym
from tqdm import tqdm
import argparse
from datetime import datetime
import yaml
import os

from stable_baselines3 import PPO
from rbc_gym.wrappers import RBCNormalizeObservation

# TODO perhaps log the best model to weights and biases

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resultdir", type=str, default=None, help="Directory of the results of the training run")
    return parser.parse_args()


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)

def main():
    args = parse_args()
    if args.resultdir and os.path.isfile(os.path.join(args.resultdir, "config.yaml")):
        config = load_config(os.path.join(args.resultdir, "config.yaml"))
        print(f"Loaded config from {args.resultdir}/config.yaml: {config}")
    else:
        raise ValueError("No valid result directory provided or config.yaml not found.")


    # Extract parameters from the config
    rbc_rayleigh_number = config["rayleigh_number"]
    rbc_heater_duration = config["heater_duration"]
    rbc_heater_limit = config["heater_limit"]
    rbc_episode_length = config["episode_length"]

    env = gym.make(
        "rbc_gym/RayleighBenardConvection3D-v0",
        render_mode="human",
        rayleigh_number=rbc_rayleigh_number,
        checkpoint=f"data/checkpoints/train/3D_ckpt_ra{rbc_rayleigh_number}.h5",
        checkpoint_idx=1,  # NOTE temporary use fixed checkpoint
        heater_limit=rbc_heater_limit,
        heater_duration=rbc_heater_duration,
        episode_length=rbc_episode_length
    )

    env = RBCNormalizeObservation(env, heater_limit=rbc_heater_limit, maxval=1)

     # Load the trained model
    model_path = os.path.join(args.resultdir, "models", "best_model.zip")
    if not os.path.isfile(model_path):
        raise ValueError(f"Model checkpoint not found: {model_path}")
    
    model = PPO.load(model_path)

    obs, info = env.reset()
    with tqdm(range(env.unwrapped.episode_length)) as pbar:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)

            pbar.update(info["t"] - pbar.n)
            pbar.set_postfix({"reward": reward, "nusselt": info["nusselt"], "t": info["t"]})
            env.render()
            if truncated:
                break

    env.close()

if __name__ == "__main__":
    main()
