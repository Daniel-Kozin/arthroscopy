"""
Smoke tests for the simulation stack.
Run with: pytest tests/
"""
import math

import numpy as np
import pytest
import torch

from sim.configs import TissueConfig, SimulationConfig, ExperimentConfig
from sim.tissue import build_tissue, scan_probe_positions, build_rectangle_mesh
from sim.shapes import create_circular_probe, SoftShapeFiniteElement
from sim.trajectory import TwoPointTrajectory, StaticTrajectory, poke_trajectory_points
from sim.simulation import SoftObjectSimulation, BadAngleError


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


# ---------------------------------------------------------------------------
# Sensor wrench transform: friction + remote (shaft-mounted) F/T sensor
#
# These tests evaluate get_sensor_reading() on the *rest* configuration
# (no minimize_energy call): the tissue top edge is perfectly flat at
# y = height, so penetration depends only on each probe vertex's y, giving
# an exactly mirror-symmetric contact pattern about x_center = 0.5. This
# makes the expected results exact (to float precision), not approximate.
# ---------------------------------------------------------------------------

def _contact_sim(friction_coeff: float = 0.0, shaft_length: float = 0.0,
                  probe_angle_deg: float = 0.0) -> SoftObjectSimulation:
    """Probe resting with its lower half penetrating a flat tissue surface."""
    tissue_config = TissueConfig(width=1.0, height=0.3, grid_size=0.1, n_zones=1)
    sim_config = SimulationConfig(
        steps=1,
        frames=1,
        warmup=False,
        collision_spring_constant=0.05,
        probe_force_noise_std=0.0,
        friction_coeff=friction_coeff,
        shaft_length=shaft_length,
        probe_angle_deg=probe_angle_deg,
    )
    tissue, _ = build_tissue(tissue_config, rng=np.random.default_rng(0))
    probe = create_circular_probe(
        radius=0.04, num_points=8, x_center=0.5, y_center=tissue_config.height - 0.02,
    )
    return SoftObjectSimulation([tissue, probe], sim_config)


def test_sensor_force_invariant_to_shaft_length():
    """Net (Fx, Fy) is a rigid-body force balance: it must not depend on
    where along the shaft the sensor sits."""
    sim = _contact_sim(friction_coeff=0.2, shaft_length=0.0)
    sim.shapes[1].last_velocity = (0.2, 0.0)  # sliding -> Fx != 0

    fx0, fy0, _ = sim.get_aggregated_sensor()

    sim.config.shaft_length = 0.5
    fx1, fy1, _ = sim.get_aggregated_sensor()

    assert fx0 == pytest.approx(fx1)
    assert fy0 == pytest.approx(fy1)


def test_sensor_symmetric_centered_poke_zero_moment():
    """Uniform tissue, centred poke, no friction -> Mz == 0 for any shaft length."""
    sim = _contact_sim(friction_coeff=0.0, shaft_length=0.0)

    _, fy0, mz0 = sim.get_aggregated_sensor()
    assert fy0 > 0  # sanity: probe is actually in contact

    sim.config.shaft_length = 0.3
    _, _, mz1 = sim.get_aggregated_sensor()

    assert mz0 == pytest.approx(0.0, abs=1e-6)
    assert mz1 == pytest.approx(0.0, abs=1e-6)


def test_sensor_moment_scales_with_shaft_length():
    """With friction (Fx != 0), relocating the sensor by L shifts Mz linearly
    by -L * Fx (forces are fixed, only the moment arm changes)."""
    sim = _contact_sim(friction_coeff=0.3, shaft_length=0.0)
    sim.shapes[1].last_velocity = (0.2, 0.0)

    fx, _, mz0 = sim.get_aggregated_sensor()
    assert fx != pytest.approx(0.0)  # friction must be active

    sim.config.shaft_length = 0.4
    _, _, mz1 = sim.get_aggregated_sensor()
    sim.config.shaft_length = 0.8
    _, _, mz2 = sim.get_aggregated_sensor()

    # Forces are constant w.r.t. L, so Mz(L) is exactly linear in L.
    assert (mz1 - mz0) == pytest.approx(0.5 * (mz2 - mz0), rel=1e-5)
    assert mz1 != pytest.approx(mz0)


def test_sensor_legacy_formula_at_zero_shaft_length():
    """shaft_length=0, friction_coeff=0 reproduces the tip-centred moment
    formula Mz = (r x F)_z = dx*fy - dy*fx about the probe centre, exactly."""
    sim = _contact_sim(friction_coeff=0.0, shaft_length=0.0)

    probe_verts = sim.shapes[1].vertices[sim.shapes[1].boundary_idx].detach().numpy()
    probe_center = probe_verts.mean(axis=0)

    readings = sim.get_sensor_reading()
    for v, (fx, fy, mz) in zip(probe_verts, readings):
        dx, dy = v[0] - probe_center[0], v[1] - probe_center[1]
        assert mz == pytest.approx(dx * fy - dy * fx, abs=1e-6)


def test_sensor_moment_sign_convention():
    """Pin the sign convention: a vertex to the right of the probe centre
    (dx > 0) with a purely upward contact force (fy > 0, fx == 0) must give
    Mz = dx*fy > 0 (standard right-hand-rule / counterclockwise), not < 0."""
    sim = _contact_sim(friction_coeff=0.0, shaft_length=0.0)

    probe_verts = sim.shapes[1].vertices[sim.shapes[1].boundary_idx].detach().numpy()
    probe_center = probe_verts.mean(axis=0)

    readings = sim.get_sensor_reading()
    found_right_side_contact = False
    for v, (fx, fy, mz) in zip(probe_verts, readings):
        dx = v[0] - probe_center[0]
        if fy > 0 and fx == 0.0 and dx > 1e-6:
            found_right_side_contact = True
            assert mz == pytest.approx(dx * fy, abs=1e-6)
            assert mz > 0
    assert found_right_side_contact  # sanity: this scenario actually occurs


# ---------------------------------------------------------------------------
# Tilted poke: contact-normal decomposition into (Fx, Fy)
#
#   Fn = 2k * penetration (vertical penetration, unchanged)
#   Fy = Fn * cos(theta)
#   Fx = Fn * sin(theta)
# ---------------------------------------------------------------------------

def test_tilt_zero_angle_is_legacy_behaviour():
    """theta=0 must reproduce the pre-tilt model exactly: Fx == 0, Fy > 0
    for a vertex in contact."""
    sim = _contact_sim(probe_angle_deg=0.0)

    _, fy, _ = sim.get_aggregated_sensor()
    assert fy > 0

    for fx, fy_i, _ in sim.get_sensor_reading():
        if fy_i > 0:
            assert fx == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("theta_deg", [15.0, 30.0])
def test_tilt_positive_angle_gives_fx_with_expected_ratio(theta_deg):
    """For theta > 0, contact vertices get Fx < 0, Fy > 0 (reaction opposes
    the probe's direction of travel, Newton's 3rd law), with
    Fx / Fy == -tan(theta) exactly (Fn cancels in the ratio)."""
    sim = _contact_sim(probe_angle_deg=theta_deg)

    found_contact = False
    for fx, fy, _ in sim.get_sensor_reading():
        if fy > 0:
            found_contact = True
            assert fx < 0
            assert fx / fy == pytest.approx(-math.tan(math.radians(theta_deg)), rel=1e-5)
    assert found_contact


def test_tilt_sign_flips_with_negative_angle():
    """Flipping the sign of theta flips the sign of Fx, leaving Fy unchanged."""
    sim_pos = _contact_sim(probe_angle_deg=20.0)
    sim_neg = _contact_sim(probe_angle_deg=-20.0)

    fx_pos, fy_pos, _ = sim_pos.get_aggregated_sensor()
    fx_neg, fy_neg, _ = sim_neg.get_aggregated_sensor()

    assert fx_pos < 0
    assert fx_neg > 0
    assert fx_neg == pytest.approx(-fx_pos, rel=1e-5)
    assert fy_neg == pytest.approx(fy_pos, rel=1e-5)


# ---------------------------------------------------------------------------
# Tilted trajectory geometry
# ---------------------------------------------------------------------------

def test_poke_trajectory_points_zero_angle_unchanged():
    """angle_deg=0 reproduces the original straight-down trajectory exactly."""
    pts = poke_trajectory_points(1.0, tissue_top_y=0.4, hover_height=0.15, penetration_depth=0.04, angle_deg=0.0)
    assert pts[0] == (1.0, 0.55)
    assert pts[1][0] == pytest.approx(1.0)
    assert pts[1][1] == pytest.approx(0.36)


def test_poke_trajectory_points_tilted():
    """A tilted poke keeps the vertical penetration depth but shifts the
    end point in x by (hover_height + penetration_depth) * tan(theta)."""
    hover, pen, theta_deg = 0.15, 0.04, 30.0
    start, end = poke_trajectory_points(1.0, tissue_top_y=0.4, hover_height=hover,
                                          penetration_depth=pen, angle_deg=theta_deg)
    assert start == (1.0, 0.55)
    assert end[1] == pytest.approx(0.4 - pen)
    expected_shift = (hover + pen) * math.tan(math.radians(theta_deg))
    assert end[0] == pytest.approx(1.0 + expected_shift)


# ---------------------------------------------------------------------------
# Bad-angle detection
# ---------------------------------------------------------------------------

def test_bad_angle_raises(tmp_path):
    """A fixed, extreme tilt angle carries the probe off the tissue for any
    scan position, so no frame ever makes contact -> BadAngleError."""
    from data.configs import DatasetConfig
    from sim.configs import ExperimentConfig, TissueConfig, ProbeConfig
    from sim.generate_dataset import run_single_experiment

    exp_config = ExperimentConfig(
        tissue=TissueConfig(width=1.0, height=0.3, grid_size=0.1, n_zones=1),
        probe=ProbeConfig(
            num_points=8, radius=0.04, hover_height=0.1, tip_penetration_depth=0.05,
            probe_angle_deg=89.0,  # tan(89 deg) ~ 57 -> end point shifted far off tissue
        ),
    )
    exp_config.simulation.steps = 2
    exp_config.simulation.warmup = False

    dataset_config = DatasetConfig(
        data_folder=str(tmp_path), num_experiments=1, num_processes=1,
        num_scan_positions=1, frames_per_poke=2,
    )

    with pytest.raises(BadAngleError):
        run_single_experiment(0, str(tmp_path), dataset_config, exp_config, np.random.default_rng(0))
