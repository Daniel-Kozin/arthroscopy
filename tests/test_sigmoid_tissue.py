"""
Tests for the sigmoid stiffness field and the grid-aligned zone fix.

Covers:
  - batched _strain_energy() == original per-triangle loop
  - discrete zones are perfect rectangles (no triangle straddles a boundary)
  - sigmoid field: one E per mesh column, monotone, k->inf recovers a 2-zone step
  - x0 snapping to mesh grid lines
"""
import dataclasses

import numpy as np
import pytest
import torch

from sim.configs import TissueConfig
from sim.tissue import (
    build_tissue,
    build_tissue_sigmoid,
    grid_counts,
    sample_sigmoid_params,
    sample_step_params,
    sigmoid_column_stiffness,
    zone_boundaries_snapped,
    SigmoidFieldParams,
)


def _tissue_config(**kwargs) -> TissueConfig:
    defaults = dict(width=2.0, height=0.4, grid_size=0.08)
    defaults.update(kwargs)
    return TissueConfig(**defaults)


# ---------------------------------------------------------------------------
# Vectorized strain energy
# ---------------------------------------------------------------------------

def test_strain_energy_matches_reference_loop():
    config = _tissue_config(grid_size=0.1, n_zones=3)
    tissue, _ = build_tissue(config, rng=np.random.default_rng(0))

    # Deform the mesh so the energy is non-trivial.
    torch.manual_seed(0)
    with torch.no_grad():
        tissue.vertices += torch.randn_like(tissue.vertices) * 0.01

    fast = tissue._strain_energy()
    reference = tissue._strain_energy_reference()
    assert fast.item() == pytest.approx(reference.item(), rel=1e-4)
    assert fast.item() > 0


def test_strain_energy_zero_at_rest():
    config = _tissue_config(grid_size=0.1, n_zones=2)
    tissue, _ = build_tissue(config, rng=np.random.default_rng(0))
    assert tissue._strain_energy().item() == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Discrete zones: perfect rectangles
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_zones", [2, 3, 5, 7])
def test_zones_are_perfect_rectangles(n_zones):
    """No triangle may straddle a zone boundary: every triangle's full x-extent
    must lie inside the zone its centroid selects. n_zones=2 previously failed
    (boundary at x=1.0 is not a grid line for grid_size=0.08)."""
    config = _tissue_config(n_zones=n_zones)
    moduli = list(np.linspace(0.002, 0.02, n_zones))
    tissue, _ = build_tissue(config, zone_young_moduli=moduli)

    bounds = np.array(zone_boundaries_snapped(config))
    verts = tissue.original_positions.numpy()
    eps = 1e-5

    for tri, e in zip(tissue.triangles, tissue.young_modulus.numpy()):
        xs = verts[tri, 0]
        zone = int(np.searchsorted(bounds, xs.mean(), side="right") - 1)
        zone = min(zone, n_zones - 1)
        assert e == pytest.approx(moduli[zone], abs=1e-7)
        assert xs.min() >= bounds[zone] - eps
        assert xs.max() <= bounds[zone + 1] + eps


def test_zone_boundaries_snapped_to_grid_lines():
    config = _tissue_config(n_zones=2)
    nx, _ = grid_counts(config)
    cell = config.width / nx
    bounds = zone_boundaries_snapped(config)
    assert bounds[0] == 0.0 and bounds[-1] == config.width
    for b in bounds:
        assert (b / cell) == pytest.approx(round(b / cell), abs=1e-9)
    # Snap moves a boundary by at most half a cell.
    ideal = [i * config.width / config.n_zones for i in range(config.n_zones + 1)]
    for b, ib in zip(bounds, ideal):
        assert abs(b - ib) <= cell / 2 + 1e-9


def test_too_many_zones_raises():
    config = _tissue_config(n_zones=100)  # only 25 mesh columns at grid 0.08
    with pytest.raises(ValueError):
        zone_boundaries_snapped(config)


# ---------------------------------------------------------------------------
# Sigmoid field
# ---------------------------------------------------------------------------

def test_sigmoid_columns_uniform_and_monotone():
    """Each mesh column holds exactly one E value (thin perfect rectangle),
    and column E values increase monotonically when e_left < e_right."""
    config = _tissue_config(n_columns=64)
    params = SigmoidFieldParams(e_left=0.002, e_right=0.02, x0=1.0, k=10.0)
    tissue, _, column_E = build_tissue_sigmoid(config, params=params)

    nx, _ = grid_counts(config)
    cell = config.width / nx
    verts = tissue.original_positions.numpy()

    col_values: dict[int, set] = {}
    for tri, e in zip(tissue.triangles, tissue.young_modulus.numpy()):
        col = int(verts[tri, 0].mean() / cell)
        col_values.setdefault(col, set()).add(round(float(e), 10))

    assert len(col_values) == nx
    for col, values in col_values.items():
        assert len(values) == 1, f"column {col} has mixed E values: {values}"
        assert values.pop() == pytest.approx(column_E[col], abs=1e-7)

    assert np.all(np.diff(column_E) > 0)


def test_sigmoid_large_k_is_two_zone_step():
    """k -> infinity must reproduce the 2-rectangle case exactly: only two
    distinct E values, separated exactly at x0."""
    config = _tissue_config(n_columns=64)
    params = SigmoidFieldParams(e_left=0.002, e_right=0.02, x0=1.0, k=1e6)
    tissue, _, column_E = build_tissue_sigmoid(config, params=params)

    assert set(np.round(column_E, 10)) == {0.002, 0.02}
    verts = tissue.original_positions.numpy()
    for tri, e in zip(tissue.triangles, tissue.young_modulus.numpy()):
        expected = 0.002 if verts[tri, 0].mean() < params.x0 else 0.02
        assert e == pytest.approx(expected, abs=1e-7)


def test_step_mode_k_inf_exact_two_rectangles():
    """k = math.inf (explicit step mode) must evaluate without NaN and give
    exactly two E values split at x0."""
    import math

    config = _tissue_config(n_columns=64)
    params = SigmoidFieldParams(e_left=0.002, e_right=0.02, x0=1.0, k=math.inf)

    E = params.young_modulus(np.array([0.0, 0.999, 1.0, 1.001, 2.0]))
    assert not np.any(np.isnan(E))
    assert list(E) == [0.002, 0.002, 0.02, 0.02, 0.02]

    _, column_E = sigmoid_column_stiffness(config, params)
    assert set(np.round(column_E, 10)) == {0.002, 0.02}

    sampled = sample_step_params(config, np.random.default_rng(0))
    assert math.isinf(sampled.k) and sampled.is_step


def test_sigmoid_no_overflow_far_from_x0():
    params = SigmoidFieldParams(e_left=0.002, e_right=0.02, x0=1.0, k=500.0)
    with np.errstate(over="raise"):
        E = params.young_modulus(np.linspace(0.0, 2.0, 100))
    assert E[0] == pytest.approx(0.002)
    assert E[-1] == pytest.approx(0.02)


def test_sampled_x0_snapped_to_grid():
    config = _tissue_config(n_columns=128)
    nx, _ = grid_counts(config)
    cell = config.width / nx
    for seed in range(20):
        params = sample_sigmoid_params(config, np.random.default_rng(seed))
        assert (params.x0 / cell) == pytest.approx(round(params.x0 / cell), abs=1e-9)
        assert config.sigmoid.x0_margin - cell / 2 <= params.x0
        assert params.x0 <= config.width - config.sigmoid.x0_margin + cell / 2
        assert config.sigmoid.k_min <= params.k <= config.sigmoid.k_max


def test_sigmoid_params_reproducible():
    config = _tissue_config(n_columns=128)
    p1 = sample_sigmoid_params(config, np.random.default_rng(7))
    p2 = sample_sigmoid_params(config, np.random.default_rng(7))
    assert dataclasses.asdict(p1) == dataclasses.asdict(p2)
