import rbc_gym  # noqa: F401
from rbc_gym.wrappers import RBCNormalizeObservation
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation, FrameStackObservation
from tqdm import tqdm

env = gym.make("rbc_gym/RayleighBenardConvection2D-v0", render_mode="human")
# wrapper to clip the obs space to [0, 1]
env = RBCNormalizeObservation(env, heater_limit=env.unwrapped.heater_limit, maxval=1)
# flatten the observation to a 1D vector
env = FlattenObservation(env)
# stack the last 4 frames
env = FrameStackObservation(env, 4)

obs, info = env.reset()
print(f"Observation shape: {obs.shape}")
with tqdm(range(env.unwrapped.episode_length)) as pbar:
    for step in pbar:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        env.render()
        if truncated:
            break
