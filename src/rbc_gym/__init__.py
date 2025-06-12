from gymnasium.envs.registration import register
import numpy as np

register(
    id="rbc_gym/RayleighBenardConvection2D-v0",
    entry_point="rbc_gym.envs:RayleighBenardConvection2DEnv",
    kwargs={
        "rayleigh_number": 10_000,
        "episode_length": 300,
        "observation_shape": (8, 48),
        "state_shape": (64, 96),
        "heater_segments": 12,
        "heater_limit": 0.75,
        "heater_duration": 1.5,
        "checkpoint": None,
        "use_gpu": False,
        "render_mode": None,
    },
)

register(
    id="rbc_gym/RayleighBenardConvection3D-v0",
    entry_point="rbc_gym.envs:RayleighBenardConvection3DEnv",
    kwargs={
        "rayleigh_number": 500,
        "prandtl_number": 0.7,
        "domain": [2, 4 * np.pi, 4 * np.pi],
        "state_shape": (16, 32, 32),
        "temperature_difference": [1, 2],
        "heater_segments": 8,
        "heater_limit": 0.9,
        "heater_duration": 0.125,
        "episode_length": 300,
        "checkpoint": None,
        "use_gpu": False,
        "render_mode": None,
    },
)
