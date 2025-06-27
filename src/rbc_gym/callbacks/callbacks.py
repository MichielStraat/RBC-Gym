from stable_baselines3.common.callbacks import BaseCallback
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
            self.training_env.render()            
        return True