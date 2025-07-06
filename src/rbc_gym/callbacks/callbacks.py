from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import LogEveryNTimesteps
from typing import TYPE_CHECKING, Any, Callable, Optional, Union
import os
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecEnv
import numpy as np


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
    

class NusseltCallback(BaseCallback):
    def __init__(
        self,
        freq: int = 1,
        verbose: int = 0,
    ):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        for info in infos:
            self.logger.record_mean("rollout/nusselt_mean", info["nusselt"])
            # TODO: log any other metric, such as:
            # self.logger.record_mean("rollout/cell_dist_mean", info["cell_dist"])
        return True
    
    

class EvaluationCallback(BaseCallback):
    def __init__(
        self,
        env: VecEnv,
        save_model: bool = False,
        save_path: Optional[str] = None,
        freq: int = 1,
        verbose: int = 0,
    ):
        super().__init__(verbose)

        self.freq = freq
        self.env = env
        self.save_model = save_model
        self.save_path = save_path
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        return True

    def _on_rollout_start(self) -> None:
        if self.n_calls % self.freq == 0:
            self.logger.info(f"Evaluating model at {self.num_timesteps} timesteps")
            nr_envs = self.env.num_envs
            rewards_list = []

            # env loop
            obs = self.env.reset()
            done = np.zeros(nr_envs)
            while not done.any():
                # env step
                actions, _ = self.model.predict(obs, deterministic=True)
                _, rewards, done, infos = self.env.step(actions)
                for id in range(nr_envs):
                    rewards_list.append(rewards[id])
                    self.logger.record_mean("eval/reward", rewards[id])
                    self.logger.record_mean("eval/nusselt", infos[id]["nusselt"])
                    self.logger.record_mean("eval/nusselt_obs", infos[id]["nusselt_obs"])

            # check for new best model
            mean_reward = np.mean(rewards_list)
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.logger.info(f"New best model with mean reward {mean_reward}")
                if self.save_model:
                    self.model.save(os.path.join(self.save_path, "best_model"))
  