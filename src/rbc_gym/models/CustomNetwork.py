from typing import Callable, Dict, List, Optional, Tuple, Type, Union

from gymnasium import spaces
from torch import nn
import torch.nn.functional as F

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

import torch
from rbc_gym.models.CNN import PeriodicPad3D


class CustomNetwork(nn.Module):
    def __init__(
            self,
            feature_dim: int = 8 * 4 * 8 * 8,  # 8 channels, and 4 height slices and  8x8 spatial dimensions, 
            last_layer_dim_pi: int = 8 * 8,  # Output of actor (policy) network
            last_layer_dim_vf: int = 8 * 8,  # Output of critic (value) network
    ):
        super().__init__()
        # IMPORTANT:
        # Save output dimensions, used to create the distributions # TODO MS: I think this is a sort of hidden layer output still
        self.feature_dim = feature_dim
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf


        # --------- Actor (policy) ------------
        self.policy_net = nn.Sequential(
            PeriodicPad3D(pad_d=1, pad_h=1, pad_w=1),
            nn.Conv3d(8, 4, kernel_size=3, stride=1, padding=0),
            nn.GELU(),
            
            PeriodicPad3D(pad_d=1, pad_h=1, pad_w=1),
            nn.Conv3d(4, 1, kernel_size=3, stride=1, padding=0),
            nn.GELU(),

            # NEW: learnable layer to collapse depth
            nn.Conv3d(1, 1, kernel_size=(4, 1, 1), stride=1), # (1,4,8,8) → (1,1,8,8)
            # TODO May want to add another learnable linear layer here with non-linear activation
            nn.Flatten(),  # Flatten to (B, 8*8)
        )

        # --------- Critic (value) ------------
        self.value_net = nn.Sequential(
            # input: (B, 8, 4, 8, 8)
            PeriodicPad3D(pad_d=1, pad_h=1, pad_w=1),
            nn.Conv3d(8, 4, kernel_size=3, stride=1, padding=0),
            nn.GELU(),

            PeriodicPad3D(pad_d=1, pad_h=1, pad_w=1),
            nn.Conv3d(4, 2, kernel_size=3, stride=1, padding=0),
            nn.GELU(),
            nn.MaxPool3d(kernel_size=2, stride=2),  
            # Output shape: (B, 2, 2, 4, 4)

            # TODO May want to add another learnable linear layer here with non-linear activation
            nn.Flatten(),  # Flatten to (B, 2*2*4*4) = (B, 64)
        )

    def forward_actor(self, x: torch.Tensor) -> torch.Tensor:
        # The input is flattened, so B*8*4*8*8, so reshape to (B, 8, 4, 8, 8)
        x = x.view(x.size(0), 8, 4, 8, 8)
        # Forward pass through the actor network
        return self.policy_net(x)

    def forward_critic(self, x: torch.Tensor) -> torch.Tensor:
        # The input is flattened, so 8*4*8*8
        x = x.view(x.size(0), 8, 4, 8, 8)
        # Forward pass through the critic network
        return self.value_net(x)
  
    def forward(self, x: torch.Tensor):
        # reshape input to (B, 8, 4, 8, 8) if needed
        if x.dim() == 2:  # If input is (B, C=
            x = x.view(x.size(0), 8, 4, 8, 8)
        elif x.dim() != 5:
            raise ValueError(f"Input tensor must be 2D (for flattened input features) or 5D (for input features with spatial structure), got {x.dim()}D tensor.")
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
        self.mlp_extractor = CustomNetwork(self.features_dim)


if __name__ == "__main__":
    # Test the Pytorch view function
    dummy_input = torch.randn(4, 5)
    print(dummy_input)
    dummy_input = torch.flatten(dummy_input)
    print(dummy_input)
    dummy_input = dummy_input.view(4, 5)  # Reshape to (2, 10) 
    print(dummy_input)

    # Test output shapes
    dummy_input = torch.randn(2, 8 * 4 * 8 * 8)  # (B, C=8, H=8, W=8, D=4)
    model = CustomNetwork()
    policy_out, value_out = model(dummy_input)
    policy_out.shape, value_out.shape

    print("Policy output shape:", policy_out.shape)  # Expected: (2, 8, 8)
    print("Value output shape:", value_out.shape)    # Expected: (2, 1)

    assert isinstance(policy_out, torch.Tensor)
    assert policy_out.shape == (2, 64)
    assert value_out.shape == (2, 64)