import rbc_gym  # noqa: F401
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation, FrameStackObservation
from tqdm import tqdm
import logging
import numpy as np
import pickle as pkl


logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

env = gym.make(
    "rbc_gym/RayleighBenardConvection3D-v0",
    render_mode="human",
    heater_duration=0.375,
    rayleigh_number=10000,
    episode_length=5,
)

nusselt_step = np.zeros(env.unwrapped.episode_length)
uv_max_step = np.zeros(env.unwrapped.episode_length)
uw_max_step = np.zeros(env.unwrapped.episode_length)
vz_max_step = np.zeros(env.unwrapped.episode_length)

obs, info = env.reset()
print(f"Observation shape: {obs.shape}")
t_ff = 4  # free-fall time units, used for scaling in the environment
nr_steps = int(env.unwrapped.episode_length // env.unwrapped.heater_duration)
with tqdm(range(nr_steps)) as pbar:
    for step in pbar:
        action = env.action_space.sample() * 0
        observation, reward, terminated, truncated, info = env.step(action)
        env.render()
        pbar.set_postfix(
            {
                "reward": reward,
                "nusselt": info["nusselt"],
                "time": info["t"],
            }
        )

        nusselt_step[step] = info["nusselt"]
        uv_max_step[step] = np.max(np.abs(observation[1]))
        uw_max_step[step] = np.max(np.abs(observation[2]))
        vz_max_step[step] = np.max(np.abs(observation[3]))

        print(f"Step {step}: Nusselt={info['nusselt']}, "
              f"UV Max={uv_max_step[step]}, "
              f"UW Max={uw_max_step[step]}, "
              f"VZ Max={vz_max_step[step]}")

        print(f"min and max temperature: {np.min(observation[0])} and {np.max(observation[0])}")
        
        if truncated:
            logging.info("Episode truncated after %d steps", step + 1)
            break

# save a pkl with the data
data = {
    "nusselt_step": nusselt_step,
    "uv_max_step": uv_max_step,
    "uw_max_step": uw_max_step,
    "vz_max_step": vz_max_step,
}
with open("flowstats_ra.pkl", "wb") as f:
    pkl.dump(data, f)
logging.info("Flow statistics saved to flowstats_ra10000.pkl")