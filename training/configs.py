"""
Configuration dataclasses for the model and training loop.

All configs are designed for use with pyrallis (YAML <-> dataclass).

See also:
  - sim.configs  -> simulation physics (ExperimentConfig, TissueConfig, ...)
  - data.configs -> dataset generation (DatasetConfig)
"""
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@dataclass
class EncoderConfig:
    """1D-CNN encoder for per-poke force/torque sequences."""
    input_dim: int = 3          # (Fx, Fy, Mz) per frame
    hidden_dim: int = 64
    num_layers: int = 3
    kernel_size: int = 3
    dropout: float = 0.1


@dataclass
class DecoderConfig:
    """MLP decoder: latent → stiffness profile."""
    hidden_dim: int = 128
    num_layers: int = 2
    output_dim: int = 5         # = n_zones; overridden at runtime from TissueConfig


@dataclass
class ModelConfig:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    data_path: str = "./data/dataset.h5"
    output_dir: str = "./runs/exp_001"
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    lr_scheduler: str = "cosine"   # "cosine" | "step" | "none"
    val_split: float = 0.15
    test_split: float = 0.10
    seed: int = 42
    num_workers: int = 4
    model: ModelConfig = field(default_factory=ModelConfig)
