import logging
import os
from typing import Any, Dict, Optional, Tuple
import gymnasium as gym
import juliacall
import numpy as np

from rbc_gym.utils import RBCField, colormap

#TODO test vecenv
#TODO test GPU
#TODO timing analysis
#TODO checkpoints

class RayleighBenard2DEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10,
    }

    def __init__(
        self,
        rayleigh_number: Optional[int] = 10_000,
        episode_length: Optional[int] = 300,
        observation_shape: Optional[tuple] = (8, 48),
        heater_segments: Optional[int] = 12,
        render_mode: Optional[str] = None,
        heater_limit: Optional[float] = 0.75,
    ) -> None:
        """
        Initialize the Rayleigh-Benard environment with the given configuration Dictionary.
        """
        super().__init__()
        self.closed = False

        # Environment configuration
        self.ra = rayleigh_number
        self.episode_length = episode_length
        self.observation_shape = observation_shape
        self.heater_segments = heater_segments
        self.heater_limit = heater_limit

        # Print environment configuration
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Using Rayleigh number Ra={self.ra}")
        self.logger.info(f"Using episode length {self.episode_length} timesteps")

        # The agent takes actions between [-1, 1] on the bottom segments
        self.action_space = gym.spaces.Box(
            -1, 1, shape=(self.heater_segments,), dtype=np.float32
        )

        # Observation Space
        lows = np.stack(
            [
                np.ones(self.observation_shape) * (-np.inf),
                np.ones(self.observation_shape) * (-np.inf),
                np.ones(self.observation_shape) * 1,
            ],
            dtype=np.float32,
            axis=0,
        )
        highs = np.stack(
            [
                np.ones(self.observation_shape) * np.inf,
                np.ones(self.observation_shape) * np.inf,
                np.ones(self.observation_shape) * 2 + self.heater_limit,
            ],
            dtype=np.float32,
            axis=0,
        )
        self.observation_space = gym.spaces.Box(
            lows,
            highs,
            shape=(
                3,
                self.observation_shape[0],
                self.observation_shape[1],
            ),
            dtype=np.float32,
        )

        # Julia simulation
        self.sim = juliacall.newmodule("RBCGymAPI")
        package_dir = os.path.dirname(os.path.abspath(__file__))
        julia_file_path = os.path.join(package_dir, "sim", "rbc_sim2D_api.jl")
        self.sim.include(julia_file_path)

        # Rendering
        self.render_mode = render_mode
        self.screen_width = 768
        self.screen_height = 512
        self.screen = None
        self.clock = None

    def reset(
        self,
        seed: int | None = None,
        options: Dict[str, Any] | None = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        super().reset(seed=seed)

        # initialize julia simulation
        self.sim.initialize_simulation(Ra=self.ra, use_gpu=False)  # TODO test GPU

        # Reset action
        self.last_action = self.action_space.sample() * 0

        return self.__get_obs(), self.__get_info()

    def step(self, action: Any = None) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        done = False
        # Simulation Step
        success = self.sim.step_simulation(np.array(action))
        if not success:
            raise RuntimeError("Error in simulation step, probably NaN values")

        # Get current info
        self.last_obs = self.__get_obs()
        self.last_reward = self.__get_reward()
        self.last_info = self.__get_info()

        # Check for truncation
        if self.last_info["t"] >= self.episode_length:
            done = True

        return self.last_obs, self.last_reward, done, self.last_info

    def __get_state(self) -> Any:
        return np.array(self.sim.get_state())

    def __get_obs(self) -> Any:
        return np.array(self.sim.get_observation())

    def __get_reward(self) -> float:
        nu = self.sim.get_nusselt(state=False)
        return -nu

    def __get_info(self) -> dict[str, Any]:
        t, step = self.sim.get_info()
        return {
            "t": t,
            "step": step,
        }

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
            )
            return

        try:
            import pygame
        except ImportError:
            raise gym.error.DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
                pygame.display.set_caption("Rayleigh Benard Convection")
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        # Data
        data = self.__get_state()[RBCField.T]
        data = np.fliplr(data)
        data = colormap(data, vmin=1, vmax=2 + self.heater_limit)

        if self.render_mode == "human":
            canvas = pygame.Surface((96, 64))
            pygame.surfarray.blit_array(canvas, data)

            # scale canvas
            canvas = pygame.transform.scale(
                canvas, (self.screen_width, self.screen_height)
            )
            self.screen.blit(canvas, (0, 0))

            # Show screen
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
            return None

        elif self.render_mode == "rgb_array":
            return data.transpose(1, 0, 2)

        else:
            raise ValueError(f"Unknown render mode: {self.render_mode}")
