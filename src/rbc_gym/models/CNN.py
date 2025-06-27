import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces


class PeriodicPad3D(nn.Module):
    def __init__(self, pad_d=0, pad_h=1, pad_w=1):
        """
        pad_d: padding in depth (zero padding, one-sided)
        pad_h: padding in height (periodic)
        pad_w: padding in width (periodic)
        """
        super().__init__()
        self.pad_d = pad_d
        self.pad_h = pad_h
        self.pad_w = pad_w

    def forward(self, x):
        # x: (B, C, D, H, W)
        if self.pad_h > 0:
            x = torch.cat([x[:, :, -self.pad_h:, :, :], x, x[:, :, :self.pad_h, :, :]], dim=2)
        if self.pad_w > 0:
            x = torch.cat([x[:, :, :, -self.pad_w:, :], x, x[:, :, :, :self.pad_w, :]], dim=3)
        if self.pad_d > 0:
            # For depth padding, we use zero padding
            x = F.pad(x, (self.pad_d, self.pad_d), mode='constant', value=0)
        return x


class FluidCNNExtractor(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor for fluid dynamics environments,
    for use in reinforcement learning tasks with Stable Baselines3.
    It processes 3D observations and outputs a feature vector.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 32, flatten: bool = False):
        super(FluidCNNExtractor, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            PeriodicPad3D(pad_d=1, pad_h=1, pad_w=1),
            # (4, 32, 32 16)
            nn.Conv3d(n_input_channels, 8, kernel_size=3, padding=0),
            nn.GELU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            # (8, 16, 16, 8)
            PeriodicPad3D(pad_d=1, pad_h=1, pad_w=1),
            nn.Conv3d(8, 8, kernel_size=3, padding=0),
            nn.GELU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            # (8, 8, 8, 4), meaning 8 channels, 8x8 spatial dimensions, and 4 height slices = 2048 total elements, reduction of 65536 / 2048 = 32x
        )

        if flatten:
            self.cnn.add_module("flatten", nn.Flatten())
        # Compute shape by doing one forward pass
            with torch.no_grad():
                n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

            self.cnn.add_module("linear1", nn.Linear(n_flatten, features_dim))
            self.cnn.add_module("act1", nn.GELU())
        

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.cnn(observation)