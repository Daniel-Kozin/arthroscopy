"""
Configuration dataclasses for the arthroscopy POC simulation.

All configs are designed for use with pyrallis (YAML ↔ dataclass).
"""
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Trajectory
# ---------------------------------------------------------------------------

@dataclass
class TrajectoriesConfig:
    """Defines a sequence of trajectory segments for the probe."""
    type: List[str]    # e.g. ["TwoPointTrajectory", "TwoPointTrajectory"]
    frames: List[int]  # frames per segment
    params: List[dict] # kwargs passed to each Trajectory subclass


# ---------------------------------------------------------------------------
# Simulation engine
# ---------------------------------------------------------------------------

@dataclass
class SimulationConfig:
    collision_spring_constant: float = 0.02   # penalty stiffness for probe–tissue contact
    steps: int = 500                           # optimizer steps per frame
    dt: float = 0.1                            # time step (used by trajectory)
    learning_rate: float = 0.001
    adam_beta_1: float = 0.2
    adam_beta_2: float = 0.999
    warmup: bool = True                        # 5 warmup steps at 2× lr
    probe_force_noise_std: float = 1e-4        # Gaussian noise added to force readings
    observation_noise: float = 0.0             # (reserved) noise on state obs
    frames: int = 0                            # overwritten per trajectory segment
    save_folder: str = "./data/sim_out"
    save_vectors: bool = True
    save_images: bool = False
    save_video: bool = False
    opaque_model: bool = False


# ---------------------------------------------------------------------------
# Tissue (the arthroscopy phantom rectangle)
# ---------------------------------------------------------------------------

@dataclass
class TissueConfig:
    """Rectangle tissue phantom with N stiffness zones."""
    width: float = 2.0                    # long dimension (x-axis)
    height: float = 0.4                   # short dimension (y-axis)
    grid_size: float = 0.08               # mesh spacing
    n_zones: int = 5                      # number of stiffness zones along width
    poisson_ratio: float = 0.45           # background Poisson's ratio
    poisson_ratio_var: float = 0.005      # per-element variation
    young_modulus_min: float = 0.002      # minimum zone Young's modulus
    young_modulus_max: float = 0.02       # maximum zone Young's modulus
    young_modulus_var: float = 0.0002     # per-element within-zone variation


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------

@dataclass
class ProbeConfig:
    """Circular tip probe (simplified from the arthroscopic instrument)."""
    num_points: int = 8        # vertices on the circular tip
    radius: float = 0.05       # probe tip radius
    trajectories: TrajectoriesConfig = field(
        default_factory=lambda: TrajectoriesConfig(type=[], frames=[], params=[])
    )


# ---------------------------------------------------------------------------
# Top-level experiment config
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    tissue: TissueConfig = field(default_factory=TissueConfig)
    probe: ProbeConfig = field(default_factory=ProbeConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)


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
