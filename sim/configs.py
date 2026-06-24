"""
Configuration dataclasses for the arthroscopy simulation.

All configs are designed for use with pyrallis (YAML <-> dataclass).

This module covers the *simulation* side only: probe trajectories, the FEM
tissue phantom, the physics engine, and the top-level experiment config that
ties them together.

Related config modules:
  - data.configs     -> dataset generation (DatasetConfig)
  - training.configs -> model + training (ModelConfig, TrainingConfig)
"""
import math
from dataclasses import dataclass, field
from typing import List


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

    # ------------------------------------------------------------------
    # Sensor / contact model
    #
    # These live here (rather than on ProbeConfig) because they are
    # consumed directly by SoftObjectSimulation.get_sensor_reading(), which
    # only has access to SimulationConfig — not the probe's own config.
    # ------------------------------------------------------------------
    friction_coeff: float = 0.0                # Coulomb friction coefficient mu.
                                                # 0.0 = frictionless (legacy behaviour, Fx == 0).
    shaft_length: float = 0.0                  # Distance L from the probe-tip centre to the
                                                # F/T sensor, measured along the probe's vertical
                                                # shaft axis (+y). 0.0 = sensor co-located with the
                                                # tip (legacy behaviour).
    probe_angle_deg: float = 0.0               # Tilt of the poke approach direction from
                                                # vertical (degrees, +x is positive). Used to
                                                # decompose the contact normal reaction into
                                                # (Fx, Fy). 0.0 = straight-down poke (legacy
                                                # behaviour, Fx from tilt == 0).


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
    hover_height: float = 0.15      # y-distance above the tissue surface where
                                     # a poke starts and ends
    tip_penetration_depth: float = 0.09 # y-distance the probe TIP (its lowest
                                     # point, i.e. centre - radius) goes below
                                     # the (rest) tissue surface at the bottom
                                     # of a poke

    # ------------------------------------------------------------------
    # Poke tilt angle (degrees from vertical, +x direction is positive)
    # ------------------------------------------------------------------
    # Fixed tilt angle in degrees. NaN (the default) means "sample uniformly
    # in [angle_min_deg, angle_max_deg] per poke using the experiment's seeded
    # RNG". (pyrallis does not support Optional[float] cleanly, hence the NaN
    # sentinel — check with math.isnan().)
    probe_angle_deg: float = math.nan
    angle_min_deg: float = -45.0
    angle_max_deg: float = 45.0

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
