import rbc_gym  # noqa: F401

from tqdm import tqdm
import gymnasium as gym

import multiprocessing as mp


def main() -> None:
    """Run a small vectorized rollout to sanity-check the environment."""
    env = gym.make_vec(
        "rbc_gym/RayleighBenardConvection2D-v0",
        num_envs=6,
        vectorization_mode="async",
        vector_kwargs={
            "copy": True,
            "daemon": True,
        },
        render_mode="human",
    )

    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    for _ in tqdm(range(100)):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        env.render()
        if truncated.any():
            break

    env.close()


if __name__ == "__main__":
    # On macOS and Windows the default “spawn” start‑method requires this guard.
    mp.set_start_method("spawn", force=True)
    main()
