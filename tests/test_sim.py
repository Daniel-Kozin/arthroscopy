"""
Smoke tests for the simulation stack.
Run with: pytest tests/
"""
import numpy as np
import pytest
import torch

from sim.configs import TissueConfig, SimulationConfig, ExperimentConfig
from sim.tissue import build_tissue, scan_probe_positions, build_rectangle_mesh
from sim.shapes import create_circular_probe, SoftShapeFiniteElement
from sim.trajectory import TwoPointTrajectory, StaticTrajectory
from sim.simulation import SoftObjectSimulation


# ---------------------------------------------------------------------------
# Tissue
# ---------------------------------------------------------------------------

def test_build_rectangle_mesh():
    boundary, interior = build_rectangle_mesh(width=1.0, height=0.4, grid_size=0.2)
    assert len(boundary) > 0
    assert len(interior) > 0
    # All boundary points should be on the perimeter
    for x, y in boundary:
        on_edge = np.isclose(x, 0.0) or np.isclose(x, 1.0) or np.isclose(y, 0.0) or np.isclose(y, 0.4)
        assert on_edge, f"Point ({x},{y}) claimed as boundary but is not on perimeter"


def test_build_tissue_label_shape():
    config = TissueConfig(width=1.0, height=0.3, grid_size=0.15, n_zones=4)
    tissue, zone_moduli = build_tissue(config, rng=np.random.default_rng(42))
    assert len(zone_moduli) == 4
    assert all(config.young_modulus_min <= e <= config.young_modulus_max for e in zone_moduli)


def test_tissue_fem_shape():
    config = TissueConfig(width=1.0, height=0.3, grid_size=0.15, n_zones=3)
    tissue, _ = build_tissue(config)
    assert isinstance(tissue, SoftShapeFiniteElement)
    assert tissue.young_modulus.shape[0] == len(tissue.triangles)


def test_scan_positions():
    config = TissueConfig(width=2.0, n_zones=5)
    positions = scan_probe_positions(config, n_positions=10)
    assert len(positions) == 10
    assert positions[0] >= 0
    assert positions[-1] <= config.width


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------

def test_create_probe():
    probe = create_circular_probe(radius=0.05, num_points=8, x_center=0.5, y_center=0.5)
    assert probe.vertices.shape == (8, 2)


# ---------------------------------------------------------------------------
# Simulation (very short — just verifies it runs without crash)
# ---------------------------------------------------------------------------

def test_simulation_one_step():
    tissue_config = TissueConfig(width=1.0, height=0.3, grid_size=0.2, n_zones=2)
    sim_config = SimulationConfig(
        steps=5,   # minimal steps for test speed
        frames=2,
        save_vectors=True,
        save_images=False,
        warmup=False,
    )

    tissue, zone_moduli = build_tissue(tissue_config, rng=np.random.default_rng(0))

    traj = TwoPointTrajectory(
        T=sim_config.dt * sim_config.frames,
        x0=0.5, y0=0.5,
        x1=0.5, y1=0.25,
    )
    probe = create_circular_probe(
        radius=0.04, num_points=6, x_center=0.5, y_center=0.5, trajectory=traj
    )

    sim = SoftObjectSimulation([tissue, probe], sim_config)
    readings = sim.run()

    assert len(readings) == sim_config.frames
    assert readings[0].shape == (3,)  # (Fx, Fy, Mz)


def test_sensor_reading_shape():
    tissue_config = TissueConfig(width=1.0, height=0.3, grid_size=0.2, n_zones=2)
    sim_config = SimulationConfig(steps=3, frames=1, warmup=False)

    tissue, _ = build_tissue(tissue_config)
    probe = create_circular_probe(radius=0.04, num_points=6, x_center=0.5, y_center=0.5)

    sim = SoftObjectSimulation([tissue, probe], sim_config)
    reading = sim.get_aggregated_sensor()
    assert reading.shape == (3,)
