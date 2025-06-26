import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces


class FluidCNNExtractor(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor for fluid dynamics environments,
    for use in reinforcement learning tasks with Stable Baselines3.
    It processes 3D observations and outputs a feature vector.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 32):
        super(FluidCNNExtractor, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv3d(n_input_channels, 8, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(8, 8, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(8, 8, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Flatten(),
        )

    # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
            torch.as_tensor(observation_space.sample()[None]).float()
        ).shape[1]

        self.cnn.add_module("linear1", nn.Linear(n_flatten, features_dim))
        self.cnn.add_module("act1", nn.GELU())
        

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.cnn(observation)