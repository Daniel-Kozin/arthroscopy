"""
Decoder: maps a latent vector to a stiffness profile.

Input:  (batch, hidden_dim)   — from ScanEncoder
Output: (batch, n_zones)      — predicted Young's modulus per zone (normalised [0,1])

Architecture: MLP with residual connections.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from training.configs import DecoderConfig


class StiffnessDecoder(nn.Module):
    """
    MLP decoder: latent → stiffness profile.

    Output is passed through Sigmoid to ensure values are in [0, 1],
    matching the normalised label from ArthroscopyDataset.
    """

    def __init__(self, config: DecoderConfig, input_dim: int | None = None):
        """
        Args:
            config: DecoderConfig
            input_dim: encoder output dim; defaults to config.hidden_dim
        """
        super().__init__()
        in_dim = input_dim if input_dim is not None else config.hidden_dim

        layers: list[nn.Module] = []
        dims = [in_dim] + [config.hidden_dim] * (config.num_layers - 1) + [config.output_dim]
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1])]
            if i < len(dims) - 2:
                layers += [nn.LayerNorm(dims[i + 1]), nn.ReLU()]

        self.net = nn.Sequential(*layers)
        self.out_act = nn.Sigmoid()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (batch, hidden_dim)
        return self.out_act(self.net(z))  # (batch, n_zones)
