import rbc_gym  # noqa: F401
import wandb
import numpy as np
import gymnasium as gym
from tqdm import tqdm


wandb.init(project="rbc-3D")
wandb.define_metric(
    "nusselt",
    summary="mean",
    step_metric="t",
)

env = gym.make(
    "rbc_gym/RayleighBenardConvection3D-v0",
    render_mode="rgb_array",
    rayleigh_number=2500,
    heater_duration=0.25,
    episode_length=10,
    checkpoint="data/checkpoints/train/3D_ckpt_ra2500.h5",
)

obs, info = env.reset()
frames = [env.render()]
with tqdm(range(env.unwrapped.episode_length)) as pbar:
    while True:
        action = env.action_space.sample() * 0
        observation, reward, terminated, truncated, info = env.step(action)

        pbar.update(info["t"] - pbar.n)
        pbar.set_postfix({"reward": reward, "t": info["t"]})

        frames.append(env.render())
        wandb.log(
            {
                "reward": reward,
                "t": info["t"],
                "nusselt": info["nusselt"],
                "state": wandb.Image(frames[-1]),
            }
        )

        if truncated:
            break

# --- Save collected frames as video -------------------------------
frames_array = np.stack(frames).astype(np.uint8)
frames_array = np.moveaxis(frames_array, -1, 1)
wandb.log(
    {
        "video": wandb.Video(
            frames_array,
            fps=10,
            format="mp4",
            caption=f"Rayleigh-Benard Convection 3D - Ra={env.unwrapped.ra}",
        )
    }
)

# --- Close the environment -------------------------------
env.close()
