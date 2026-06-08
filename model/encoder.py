"""
Encoder: maps a scan of force/torque readings to a latent vector.

Input shape:  (batch, n_positions, frames_per_poke, input_dim=3)
Output shape: (batch, n_positions, hidden_dim)

Architecture: per-poke 1D-CNN (shared weights across positions) → per-position embedding.
The scan-level representation is the sequence of per-position embeddings, which
the decoder can then process.

Why 1D-CNN: force signals are short time series (10-30 frames), CNN is efficient
and invariant to exact timing.  Can be swapped for LSTM/Transformer later.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from sim.configs import EncoderConfig


class PokeEncoder(nn.Module):
    """
    Encodes a single poke (time series of sensor readings) into a fixed-size vector.

    Input:  (batch, frames, input_dim)
    Output: (batch, hidden_dim)
    """

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config

        layers: list[nn.Module] = []
        in_ch = config.input_dim
        for i in range(config.num_layers):
            out_ch = config.hidden_dim
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=config.kernel_size, padding=config.kernel_size // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.Dropout(config.dropout),
            ]
            in_ch = out_ch

        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)  # temporal pooling → (batch, hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, frames, input_dim)
        x = x.permute(0, 2, 1)          # → (batch, input_dim, frames)
        x = self.conv(x)                 # → (batch, hidden_dim, frames)
        x = self.pool(x).squeeze(-1)     # → (batch, hidden_dim)
        return x


class ScanEncoder(nn.Module):
    """
    Encodes a full probe scan (sequence of pokes) into a fixed-size latent vector.

    Input:  (batch, n_positions, frames, input_dim)
    Output: (batch, hidden_dim)

    Uses PokeEncoder with shared weights across positions, then pools across positions.
    """

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.poke_encoder = PokeEncoder(config)
        self.hidden_dim = config.hidden_dim

        # Optional: aggregate across positions with attention instead of mean
        self.position_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_positions, frames, input_dim)
        B, P, F, D = x.shape

        # Encode each poke independently (shared weights)
        x_flat = x.view(B * P, F, D)
        poke_emb = self.poke_encoder(x_flat)            # (B*P, hidden_dim)
        poke_emb = poke_emb.view(B, P, self.hidden_dim) # (B, P, hidden_dim)

        # Pool across positions
        z = self.position_pool(
            poke_emb.permute(0, 2, 1)   # (B, hidden_dim, P)
        ).squeeze(-1)                    # (B, hidden_dim)

        return z

    def encode_positions(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns per-position embeddings (no pooling).
        Useful for sequence-to-sequence tasks (future: position-wise stiffness pred).

        Input:  (batch, n_positions, frames, input_dim)
        Output: (batch, n_positions, hidden_dim)
        """
        B, P, F, D = x.shape
        x_flat = x.view(B * P, F, D)
        poke_emb = self.poke_encoder(x_flat)
        return poke_emb.view(B, P, self.hidden_dim)
