from gymnasium.envs.registration import register

register(
    id="rbc_gym/RayleighBenardConvection2D-v0",
    entry_point="rbc_gym.envs:RayleighBenardConvection2DEnv",
    kwargs={
        "rayleigh_number": 10_000,
        "episode_length": 300,
        "observation_shape": (8, 48),
        "state_shape": (64, 96),
        "heater_segments": 12,
        "render_mode": None,
        "heater_limit": 0.75,
        "heater_duration": 1.5,
        "checkpoint_dir": "",
        "use_gpu": False,
    },
)
