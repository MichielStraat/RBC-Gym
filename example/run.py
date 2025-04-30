from rbc_gym import RayleighBenard2DEnv
from tqdm import tqdm

env = RayleighBenard2DEnv(
    rayleigh_number=10_000,
    episode_length=300,
    render_mode="human",
)

env.reset()
done = False
for step in tqdm(range(env.episode_length)):
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    env.render()
    if done:
        break
