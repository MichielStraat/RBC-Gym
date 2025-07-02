from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import LogEveryNTimesteps
from typing import TYPE_CHECKING, Any, Callable, Optional, Union
import wandb

import os


class RenderCallback(BaseCallback):
    """
    Callback for rendering the environment state

    :param check_freq: How often to render
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, check_freq: int = 1, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq


    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # self.training_env.render()
            print('Rendering environment at step:', self.n_calls)
        return True
    

# class LogEveryNTimestepsWandb(LogEveryNTimesteps):
#     """
#     Log data every ``n_steps`` timesteps

#     :param n_steps: Number of timesteps between two trigger.
#     """

#     def __init__(self, n_steps: int):
#         super().__init__(n_steps=n_steps)


#     def _log_data(self, _locals: dict[str, Any], _globals: dict[str, Any]) -> bool:
#         super()._log_data(_locals, _globals)
#         # additionally log to wandb
#         return True
    

# class NusseltCallback(BaseCallback):
#     def __init__(
#         self,
#         freq: int = 1,
#         verbose: int = 0,
#     ):
#         super().__init__(verbose)

#     def _on_step(self) -> bool:
#         infos = self.locals.get("infos")
#         for info in infos:
#             self.logger.record_mean("rollout/nusselt_obs_mean", info["nusselt_obs"])
#             self.logger.record_mean("rollout/nusselt_mean", info["nusselt"])
#             self.logger.record_mean("rollout/cell_dist_mean", info["cell_dist"])
#         return True
    
# class LogNusseltNumberCallback(CallbackBase):
#     def __init__(
#         self,
#         interval: Optional[int] = 1,
#         nr_episodes: int = 1,
#     ):
#         super().__init__(interval=interval)
#         for idx in range(nr_episodes):
#             wandb.define_metric(f"ep{idx}/time")
#             wandb.define_metric(
#                 f"ep{idx}/nusselt_state", step_metric=f"ep{idx}/time", summary="mean"
#             )
#             wandb.define_metric(
#                 f"ep{idx}/nusselt_obs", step_metric=f"ep{idx}/time", summary="mean"
#             )

#     def __call__(self, env, obs, reward, info, episode_idx):
#         if super().__call__(env, obs, reward, info):
#             # state = env.simulation.state
#             # nusselt = env.simulation.compute_nusselt(state)
#             wandb.log(
#                 {
#                     f"ep{episode_idx}/time": info["t"],
#                     f"ep{episode_idx}/nusselt_state": info["nusselt"],
#                     f"ep{episode_idx}/nusselt_obs": info["nusselt_obs"],
#                 }
#             )