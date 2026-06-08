"""
Full encoder-decoder model for tissue stiffness prediction.

Input:  sensor scan  (batch, n_positions, frames_per_poke, 3)
Output: stiffness    (batch, n_zones)  in [0, 1]  (normalised Young's moduli)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from sim.configs import ModelConfig
from model.encoder import ScanEncoder
from model.decoder import StiffnessDecoder


class ArthroscopyModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.encoder = ScanEncoder(config.encoder)
        self.decoder = StiffnessDecoder(
            config.decoder,
            input_dim=config.encoder.hidden_dim,
        )

    def forward(self, sensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sensor: (batch, n_positions, frames_per_poke, 3)
        Returns:
            stiffness: (batch, n_zones)
        """
        z = self.encoder(sensor)       # (batch, hidden_dim)
        return self.decoder(z)          # (batch, n_zones)

    def encode(self, sensor: torch.Tensor) -> torch.Tensor:
        """Expose latent for visualisation / downstream tasks."""
        return self.encoder(sensor)

    @classmethod
    def from_config(cls, config: ModelConfig) -> "ArthroscopyModel":
        return cls(config)
