import rbc_gym  # noqa: F401
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation, FrameStackObservation
from tqdm import tqdm
import logging
import numpy as np
import pickle as pkl
import os

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)


  # write the data to file
pkl_path = "experiments/flowstats/flowstats_ra.pkl"
# 1️⃣ Check if the pickle exists, load if so:
if os.path.exists(pkl_path):
    with open(pkl_path, "rb") as f:
        flowstats = pkl.load(f)
    logging.info(f"Loaded existing flowstats from {pkl_path}")
else:
    flowstats = {}
    logging.info(f"No existing flowstats found → creating new file {pkl_path}")

rayleigh_numbers = [1000000]
for ra in rayleigh_numbers:
    env = gym.make(
        "rbc_gym/RayleighBenardConvection3D-v0",
        render_mode="human",
        heater_duration=0.25,
        rayleigh_number=ra,
        episode_length=100,
    )

    t_ff = 4  # free-fall time units, used for scaling in the environment
    nr_steps = int(np.ceil(env.unwrapped.episode_length / (env.unwrapped.heater_duration * t_ff)))
    print(f'Running Ra={ra}. Number of simulation steps to be executed: {nr_steps}')

    # stats per step
    nusselt_step = np.zeros(nr_steps)
    uv_max_step = np.zeros(nr_steps)
    uw_max_step = np.zeros(nr_steps)
    uz_max_step = np.zeros(nr_steps)

    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    
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
            uz_max_step[step] = np.max(np.abs(observation[3]))

            # print(f"Step {step}: Nusselt={info['nusselt']}, "
            #     f"UV Max={uv_max_step[step]}, "
            #     f"UW Max={uw_max_step[step]}, "
            #     f"VZ Max={uz_max_step[step]}")

            # print(f"min and max temperature: {np.min(observation[0])} and {np.max(observation[0])}")
            
            if truncated:
                logging.info("Episode truncated after %d steps", step + 1)
                break


        # 2️⃣ Add or overwrite the Ra entry:
        flowstats[str(ra)] = {
            "nusselt_step": nusselt_step,
            "uv_max_step": uv_max_step,
            "uw_max_step": uw_max_step,
            "uz_max_step": uz_max_step,
        }

        # 3️⃣ Save back to the same file:
        with open(pkl_path, "wb") as f:
            pkl.dump(flowstats, f)

        logging.info(f"Flow statistics for Ra={ra} saved to {pkl_path}")