"""
Configuration dataclasses for dataset generation.

All configs are designed for use with pyrallis (YAML <-> dataclass).

See also:
  - sim.configs      -> simulation physics (ExperimentConfig, TissueConfig, ...)
  - training.configs -> model + training (ModelConfig, TrainingConfig)
"""
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

@dataclass
class DatasetConfig:
    """Controls parallel dataset generation."""
    data_folder: str = "./data/dataset"
    num_experiments: int = 1000
    start_index: int = 0
    num_processes: int = 8
    num_scan_positions: int = 10      # probe poke locations evenly spaced across width
    frames_per_poke: int = 10         # frames per individual poke trajectory
    penetration: float = 0.95         # how deep the probe goes (fraction of tissue height)
    experiment_config_path: str = "configs/default.yaml"
