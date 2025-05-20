import rbc_gym  # noqa: F401
import logging
import gymnasium as gym
from tqdm import tqdm


logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

env = gym.make(
    "rbc_gym/RayleighBenardConvection2D-v0",
    checkpoint="data/checkpoints/checkpoints10000.h5",
    render_mode="human",
)

obs, info = env.reset()
for step in tqdm(range(env.unwrapped.episode_length)):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()
    if truncated:
        break
