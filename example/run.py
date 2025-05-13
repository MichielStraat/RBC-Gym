import rbc_gym
import gymnasium as gym
from tqdm import tqdm

env = gym.make("rbc_gym/RayleighBenardConvection2D-v0", render_mode="human")

print(env.observation_space.shape)
for i in range(50):
    obs, info = env.reset(seed=i)

    if not env.observation_space.contains(obs):
        print(f"T, max={obs[0].max()}, min={obs[0].min()}")
        print("Observation space is invalid")
exit(0)



obs, info = env.reset()
for step in tqdm(range(env.episode_length)):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()
    if truncated:
        break
