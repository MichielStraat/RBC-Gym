from typing import Callable, Dict, List, Optional, Tuple, Type, Union

from gymnasium import spaces
from torch import nn
import torch.nn.functional as F

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

import torch
from rbc_gym.models.CNN import PeriodicPad3D


class CustomNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        # --------- Actor (policy) ------------
        self.policy_net = nn.Sequential(
            PeriodicPad3D(pad_d=1, pad_h=1, pad_w=1),
            nn.Conv3d(8, 4, kernel_size=3, stride=1, padding=0),
            nn.GELU(),
            
            PeriodicPad3D(pad_d=1, pad_h=1, pad_w=1),
            nn.Conv3d(4, 1, kernel_size=3, stride=1, padding=0),
            nn.GELU(),

            # NEW: learnable layer to collapse depth
            nn.Conv3d(1, 1, kernel_size=(1, 1, 4), stride=1) # (1,8,8,4) → (1,8,8,1)
        )

        # --------- Critic (value) ------------
        self.value_net = nn.Sequential(
            PeriodicPad3D(pad_d=1, pad_h=1, pad_w=1),
            nn.Conv3d(8, 4, kernel_size=3, stride=1, padding=0),
            nn.GELU(),

            PeriodicPad3D(pad_d=1, pad_h=1, pad_w=1),
            nn.Conv3d(4, 1, kernel_size=3, stride=1, padding=0),
            nn.GELU(),
        )

        self.value_fc = nn.Linear(8 * 8 * 4, 1)

    def forward_actor(self, x: torch.Tensor) -> torch.Tensor:
        # Input: (B, 8, 8, 8, 4) → PyTorch expects (B, C, D, H, W)
        x = self.policy_net(x)
        # Output shape: (B, 1, 8, 8, 1) → (B, C=1, D=8, H=8, W=1)
        return x.squeeze(1).squeeze(-1)  # → (B, 8, 8)

    def forward_critic(self, x: torch.Tensor) -> torch.Tensor:
        # Input: (B, 8, 8, 8, 4) → PyTorch expects (B, C, D, H, W)
        
        x = self.value_net(x)
        # Output shape: (B, 1, 8, 8, 4) → (B, C=1, D=8, H=8, W=4)
        
        x = x.squeeze(1) # → (B, 8, 8, 4)
        x = x.view(x.size(0), -1)  # Flatten to (B, 8*8*4)
        return self.value_fc(x)       # → (B, 1)

    def forward(self, x: torch.Tensor):
        return self.forward_actor(x), self.forward_critic(x)



class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        # Disable orthogonal initialization
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )


    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork()


if __name__ == "__main__":
    # Test output shapes
    dummy_input = torch.randn(2, 8, 8, 8, 4)  # (B, C=8, H=8, W=8, D=4)
    model = CustomNetwork()
    policy_out, value_out = model(dummy_input)
    policy_out.shape, value_out.shape

    print("Policy output shape:", policy_out.shape)  # Expected: (2, 8, 8)
    print("Value output shape:", value_out.shape)    # Expected: (2, 1)

    assert isinstance(policy_out, torch.Tensor)
    assert policy_out.shape == (2, 8, 8)
    assert value_out.shape == (2, 1)