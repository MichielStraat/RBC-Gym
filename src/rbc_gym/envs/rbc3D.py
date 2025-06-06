import logging
import warnings
import os
from pathlib import Path
from enum import IntEnum
from typing import Any, Dict, Optional, Tuple
import matplotlib

import gymnasium as gym
import juliacall
import juliapkg
import numpy as np

# Optional dependency for 3‑D rendering
try:
    import pyvista as pv

except ImportError:  # pragma: no cover
    pv = None


class RBC3DField(IntEnum):
    T = 0
    U = 1
    V = 2
    W = 3


def colormap(value, vmin=1, vmax=2, colormap="turbo"):
    cmap = matplotlib.colormaps[colormap]
    value = (value - vmin) / (vmax - vmin)
    return cmap(value, bytes=True)[:, :, :3]


class RayleighBenardConvection3DEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10,
    }

    def __init__(
        self,
        rayleigh_number: Optional[int] = 10_000,
        episode_length: Optional[int] = 300,
        state_shape: Optional[list] = [32, 48, 48],
        heater_segments: Optional[int] = 12,
        heater_limit: Optional[float] = 0.75,
        heater_duration: Optional[float] = 0.125,
        use_gpu: Optional[bool] = False,
        checkpoint: Optional[str] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        """
        Initialize the Rayleigh-Benard environment with the given configuration Dictionary.
        """
        super().__init__()
        self.closed = False
        self.use_gpu = use_gpu
        self.checkpoint = checkpoint

        # Environment configuration
        self.ra = rayleigh_number
        self.episode_length = episode_length
        self.state_shape = state_shape
        self.heater_segments = heater_segments
        self.heater_limit = heater_limit
        self.heater_duration = heater_duration

        # Print environment configuration
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Using Rayleigh number Ra={self.ra}")
        self.logger.info(f"Using episode length {self.episode_length} timesteps")

        # The agent takes actions between [-1, 1] on the bottom segments
        self.action_space = gym.spaces.Box(
            -1, 1, shape=(self.heater_segments, self.heater_segments), dtype=np.float32
        )

        # Observation Space
        lows = np.stack(
            [
                np.full(self.state_shape, 1),
                np.full(self.state_shape, -np.inf),
                np.full(self.state_shape, -np.inf),
            ],
            dtype=np.float32,
            axis=0,
        )
        highs = np.stack(
            [
                np.full(self.state_shape, 2 + self.heater_limit),
                np.full(self.state_shape, np.inf),
                np.full(self.state_shape, np.inf),
            ],
            dtype=np.float32,
            axis=0,
        )
        self.observation_space = gym.spaces.Box(
            lows,
            highs,
            shape=(
                3,
                self.state_shape[0],
                self.state_shape[1],
                self.state_shape[2],
            ),
            dtype=np.float32,
        )

        # Julia simulation
        juliapkg.resolve()
        self.sim = juliacall.newmodule("RBCGymAPI")
        package_dir = os.path.dirname(os.path.abspath(__file__))
        julia_file_path = os.path.join(package_dir, "..", "sim", "rbc_sim3D_api.jl")
        self.sim.include(julia_file_path)

        # Rendering
        self.render_mode = render_mode
        self._plotter = None
        self._grid = None
        self._volume = None

    def reset(
        self,
        seed: int | None = None,
        options: Dict[str, Any] | None = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        super().reset(seed=seed)

        # initialize julia simulation
        self.sim.initialize_simulation(
            Ra=self.ra,
            grid=self.state_shape[::-1],  # julia uses column-major order
            heaters=self.heater_segments,
            heater_limit=self.heater_limit,
            dt=self.heater_duration,
            seed=self.np_random_seed,
            checkpoint_path=None,
            use_gpu=self.use_gpu,
        )

        # Reset action
        self.last_action = self.action_space.sample() * 0

        return self.__get_obs(), self.__get_info()

    def step(self, action: Any = None) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        terminated = False  # is always false; no terminal state
        truncated = False
        # zero action if none
        if action is None:
            action = np.zeros(self.action_space.shape, dtype=np.float32)
            warnings.warn("No action provided, using zero action")

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
            truncated = True

        return self.last_obs, self.last_reward, terminated, truncated, self.last_info

    def __get_obs(self) -> Any:
        obs = np.array(self.sim.get_state(), dtype=np.float32)
        obs = obs.transpose(0, 3, 2, 1)  # julia uses column-major order
        obs = np.flip(obs, axis=2)
        return obs

    def __get_reward(self) -> float:
        nu = self.sim.get_nusselt()
        return -nu

    def __get_info(self) -> dict[str, Any]:
        t, step = self.sim.get_info()
        nu = self.sim.get_nusselt()
        return {
            "t": t,
            "step": step,
            "nusselt": nu,
        }

    def render(
        self, dx: float = 1.0, dy: float = 1.0, dz: float = 1.0, stride: int = 1
    ):
        """
        Visualise the current 3‑D temperature field using PyVista.
        Parameters
        ----------
        dx, dy, dz : float, optional
            Physical grid spacing in the x-, y-, and z-directions. These
            scale the axes in the scene. Defaults to 1.0 for each.
        stride : int, optional
            Sub-sampling factor along each axis (1 = full resolution, 2 = every
            other voxel, …). Rendering ≈ N³/stride³ voxels, so large stride
            values speed up volume display.
        """
        if pv is None:
            raise ModuleNotFoundError(
                "PyVista is required for volume rendering. "
                "Install it with `pip install pyvista`."
            )

        # Get observation
        obs = self.__get_obs()

        # Temperature is channel 0
        T = obs[RBC3DField.T]
        # print(f"T variance {T.var():.3f}, mean {T.mean():.3f}, max {T.max():.3f}")
        stride = max(1, int(stride))
        if stride > 1:
            T = T[::stride, ::stride, ::stride]
            dx *= stride
            dy *= stride
            dz *= stride
        nz, ny, nx = T.shape

        # Build a rectilinear grid (VTK’s preferred format for cell‑centred data)
        x = np.arange(nx) * dx
        y = np.arange(ny) * dy
        z = np.arange(nz) * dz

        # update the RectilinearGrid
        grid = pv.RectilinearGrid(x, y, z)

        # VTK is column‑major like Julia, so flatten in Fortran order
        grid["T"] = T.ravel(order="C")

        if self.render_mode == "human":
            p = pv.Plotter()
            p.add_volume(
                grid,
                scalars="T",
                cmap="turbo",
                clim=(0.0, 1.0),
                opacity="sigmoid_1",
                shade=True,
            )
            p.add_axes()
            p.show(auto_close=True, interactive=False)

        elif self.render_mode == "rgb_array":
            # Off‑screen render to an image array for Gym‑style RGB output
            p = pv.Plotter(off_screen=True, window_size=(800, 800))
            p.add_volume(
                grid,
                scalars="T",
                cmap="turbo",
                clim=(0.0, 1.0),
                opacity="sigmoid_1",
                shade=True,
            )
            p.add_axes()
            img = p.screenshot(return_img=True)
            p.close()
            # img is RGBA uint8; convert to RGB for Gym
            return img[:, :, :3]

        else:
            raise ValueError(
                f"Unknown render_mode '{self.render_mode}'. "
                "Expected 'human' or 'rgb_array'."
            )

    def plot_T(self):
        """
        Plot the temperature field using Matplotlib.
        """
        import matplotlib.pyplot as plt

        obs = self.__get_obs()
        T = obs[RBC3DField.T]
        plt.imshow(T[:, :, T.shape[2] // 2], cmap="turbo")
        plt.colorbar(label="Temperature")
        plt.title("Temperature Field at Mid-Plane")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()
