import rbc_gym  # noqa: F401
import numpy as np
import gymnasium as gym
from tqdm import tqdm
import imageio.v3 as iio
import matplotlib.pyplot as plt

from rbc_gym.wrappers import RBCNormalizeObservation, RBCNormalizeReward

env = gym.make(
    "rbc_gym/RayleighBenardConvection3D-v0",
    render_mode="rgb_array",
    rayleigh_number=2500,
    heater_duration=0.25,
    episode_length=10,
    checkpoint="data/checkpoints/train/3D_ckpt_ra2500.h5",
)
# wrapper to clip the obs space to [0, 1]
env = RBCNormalizeObservation(env, heater_limit=env.unwrapped.heater_limit, maxval=1)
# wrapper to normalize the reward to [0, 1]
env = RBCNormalizeReward(env)

obs, info = env.reset()
frames = [env.render()]
nusselts = []
time = []

with tqdm(range(env.unwrapped.episode_length)) as pbar:
    while True:
        action = env.action_space.sample() * 0
        observation, reward, terminated, truncated, info = env.step(action)

        pbar.update(info["t"] - pbar.n)
        pbar.set_postfix({"reward": reward, "t": info["t"]})

        frames.append(env.render())

        nusselts.append(info["nusselt"])
        time.append(info["t"])

        if truncated:
            break

# --- Save collected frames as an MP4 -------------------------------
if frames:
    iio.imwrite("logs/rbc_sim.mp4", frames, fps=10, codec="h264")

# --- Plot nusselt number -------------------------------
plt.figure(figsize=(10, 5))
plt.plot(time, nusselts, label="Nusselt Number")
plt.xlabel("Time")
plt.ylabel("Nusselt Number")
plt.title("Nusselt Number over Time")
plt.legend()
plt.grid()
plt.savefig("logs/nusselt_number.png")

print(f"Mean nusselt number for last 200 timesteps: {np.mean(nusselts[-100:])}")

# --- Close the environment -------------------------------
env.close()
