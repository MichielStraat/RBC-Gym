import rbc_gym  # noqa: F401
import gymnasium as gym
from tqdm import tqdm
# import imageio.v3 as iio

env = gym.make(
    "rbc_gym/RayleighBenardConvection3D-v0",
    render_mode="human",
    episode_length=30,
)

obs, info = env.reset()
print("Observation shape:", obs.shape)
frames = []

with tqdm(range(env.unwrapped.episode_length)) as pbar:
    while True:
        action = env.action_space.sample() * 0
        observation, reward, terminated, truncated, info = env.step(action)

        pbar.update(info["t"] - pbar.n)
        pbar.set_postfix({"reward": reward, "t": info["t"]})

        # frames.append(env.render())
        env.render()

        if truncated:
            break

# --- Save collected frames as an MP4 -------------------------------
# if frames:
#     print("Writing video to rbc_sim.mp4 ...")
#     iio.imwrite("rbc_sim.mp4", frames, fps=5, codec="h264")
