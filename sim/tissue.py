"""
Arthroscopy tissue phantom: a rectangle with a per-column stiffness profile.

The rectangle lies in [0, width] x [0, height].
Bottom edge is fixed (Dirichlet BC).
The probe approaches from above (positive y direction).

Two stiffness layouts are supported:

1. Discrete zones (legacy) — n_zones equal-width zones, each with an
   independently sampled Young's modulus (`build_tissue`). Zone boundaries
   are SNAPPED to the nearest mesh grid line so every zone is a perfect
   rectangle of whole mesh columns (previously a boundary could cut through
   a cell and produce a jagged triangle edge).

2. Sigmoid gradient (`build_tissue_sigmoid`) — Young's modulus follows

       E(x) = E_left + (E_right - E_left) * sigmoid(k * (x - x0))

   evaluated per mesh column (one value per column -> n_columns thin
   rectangles). k controls the transition sharpness: k -> infinity is a
   two-rectangle step at x0; small k is a smooth gradient. The label is
   (E_left, E_right, x0, k).
"""
import math
from dataclasses import dataclass, replace

import numpy as np
from scipy.special import expit

from sim.configs import TissueConfig
from sim.shapes import SoftShapeFiniteElement
from sim.trajectory import StaticTrajectory


# ---------------------------------------------------------------------------
# Mesh construction
# ---------------------------------------------------------------------------

def grid_counts(config: TissueConfig) -> tuple[int, int]:
    """
    Number of mesh cells (nx, ny) along x and y.

    If config.n_columns > 0, the mesh has exactly that many columns and the
    row count is chosen to keep cells approximately square. Otherwise both
    counts derive from grid_size (legacy behaviour).
    """
    if config.n_columns > 0:
        nx = config.n_columns
        cell = config.width / nx
        ny = max(1, round(config.height / cell))
    else:
        nx = max(1, round(config.width / config.grid_size))
        ny = max(1, round(config.height / config.grid_size))
    return nx, ny


def build_rectangle_mesh_cells(
    width: float,
    height: float,
    nx: int,
    ny: int,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """
    Generate a uniform (nx × ny cells) grid of vertices for a rectangle.
    Grid lines land exactly on 0 and width/height by construction.

    Returns:
        boundary_vertices: list of (x, y) on the perimeter
        interior_vertices: list of (x, y) strictly inside
    """
    xs = np.linspace(0.0, width, nx + 1)
    ys = np.linspace(0.0, height, ny + 1)

    boundary = []
    interior = []
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            point = (float(x), float(y))
            if i in (0, nx) or j in (0, ny):
                boundary.append(point)
            else:
                interior.append(point)
    return boundary, interior


def build_rectangle_mesh(
    width: float,
    height: float,
    grid_size: float,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Legacy grid_size-based entry point (kept for existing callers)."""
    nx = max(1, round(width / grid_size))
    ny = max(1, round(height / grid_size))
    return build_rectangle_mesh_cells(width, height, nx, ny)


def snap_to_grid(x: float, width: float, nx: int) -> float:
    """Snap an x-coordinate to the nearest mesh grid line (of nx+1 lines)."""
    cell = width / nx
    return float(np.clip(round(x / cell) * cell, 0.0, width))


def _build_fem_shape(
    config: TissueConfig,
    default_young_modulus: float,
) -> SoftShapeFiniteElement:
    """Mesh the rectangle and wrap it in a SoftShapeFiniteElement."""
    nx, ny = grid_counts(config)
    boundary_verts, interior_verts = build_rectangle_mesh_cells(
        config.width, config.height, nx, ny
    )
    all_verts = np.array(boundary_verts + interior_verts, dtype=np.float64)

    # Fixed vertices: bottom edge (y ≈ 0)
    fixed_idx = [i for i, v in enumerate(all_verts) if np.isclose(v[1], 0.0)]

    # Boundary indices: all perimeter vertices (used for collision detection)
    n_boundary = len(boundary_verts)
    boundary_idx = list(range(n_boundary))

    return SoftShapeFiniteElement(
        vertices=all_verts,
        fixed_vertices_idx=fixed_idx,
        trajectory=StaticTrajectory(T=1.0),
        boundary_idx=boundary_idx,
        default_young_modulus=default_young_modulus,
        default_poisson_ratio=config.poisson_ratio,
        young_modulus_var=config.young_modulus_var,
        poisson_ratio_var=config.poisson_ratio_var,
    )


# ---------------------------------------------------------------------------
# Layout 1: discrete zones (legacy, now grid-aligned)
# ---------------------------------------------------------------------------

def sample_zone_stiffness(config: TissueConfig, rng: np.random.Generator | None = None) -> list[float]:
    """
    Sample a Young's modulus for each zone uniformly in [E_min, E_max].

    Returns list of length n_zones.
    """
    if rng is None:
        rng = np.random.default_rng()
    return list(
        rng.uniform(config.young_modulus_min, config.young_modulus_max, size=config.n_zones)
    )


def zone_boundaries_snapped(config: TissueConfig) -> list[float]:
    """
    Equal-split zone boundaries snapped to the nearest mesh grid line, so
    every zone is a whole number of mesh columns (a perfect rectangle).
    The max shift per boundary is half a cell (= width / (2*nx)).
    """
    nx, _ = grid_counts(config)
    zone_width = config.width / config.n_zones
    snapped = [snap_to_grid(i * zone_width, config.width, nx) for i in range(config.n_zones + 1)]
    if any(b1 - b0 <= 0 for b0, b1 in zip(snapped[:-1], snapped[1:])):
        raise ValueError(
            f"n_zones={config.n_zones} too high for mesh resolution nx={nx}: "
            "two zone boundaries snapped to the same grid line. Increase "
            "n_columns / decrease grid_size."
        )
    return snapped


def build_tissue(
    config: TissueConfig,
    zone_young_moduli: list[float] | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[SoftShapeFiniteElement, list[float]]:
    """
    Build a rectangular FEM tissue phantom with n_zones discrete zones.

    Args:
        config: TissueConfig
        zone_young_moduli: optional pre-specified stiffness per zone.
            If None, sampled randomly.
        rng: optional random generator for reproducibility.

    Returns:
        tissue: SoftShapeFiniteElement
        zone_young_moduli: the stiffness values used (ground truth label)
    """
    if zone_young_moduli is None:
        zone_young_moduli = sample_zone_stiffness(config, rng)

    tissue = _build_fem_shape(config, default_young_modulus=float(np.mean(zone_young_moduli)))
    tissue.set_zone_stiffness(zone_boundaries_snapped(config), zone_young_moduli)
    return tissue, zone_young_moduli


# ---------------------------------------------------------------------------
# Layout 2: sigmoid stiffness gradient
# ---------------------------------------------------------------------------

@dataclass
class SigmoidFieldParams:
    """Ground-truth parameters of one sigmoid stiffness field (the label)."""
    e_left: float    # E as x -> 0
    e_right: float   # E as x -> width
    x0: float        # transition centre (snapped to a mesh grid line)
    k: float         # transition steepness; k = math.inf is a hard 2-zone step

    @property
    def is_step(self) -> bool:
        return math.isinf(self.k)

    def young_modulus(self, x) -> np.ndarray:
        """Evaluate E(x). Works on scalars or numpy arrays; overflow-safe.
        k = inf is the exact step limit (expit would give NaN at x == x0)."""
        x = np.asarray(x, dtype=np.float64)
        if self.is_step:
            return np.where(x >= self.x0, self.e_right, self.e_left)
        return self.e_left + (self.e_right - self.e_left) * expit(
            self.k * (x - self.x0)
        )


def sample_sigmoid_params(
    config: TissueConfig,
    rng: np.random.Generator | None = None,
) -> SigmoidFieldParams:
    """
    Sample (E_left, E_right, x0, k):
      - E_left, E_right ~ U[E_min, E_max] (independent)
      - x0 ~ U[margin, width - margin], snapped to the nearest grid line
      - k ~ log-uniform in [k_min, k_max] so gradients and near-steps are
        equally represented
    """
    if rng is None:
        rng = np.random.default_rng()
    scfg = config.sigmoid
    e_left, e_right = rng.uniform(
        config.young_modulus_min, config.young_modulus_max, size=2
    )
    nx, _ = grid_counts(config)
    x0 = snap_to_grid(
        float(rng.uniform(scfg.x0_margin, config.width - scfg.x0_margin)),
        config.width, nx,
    )
    k = float(np.exp(rng.uniform(math.log(scfg.k_min), math.log(scfg.k_max))))
    return SigmoidFieldParams(float(e_left), float(e_right), x0, k)


def sample_step_params(
    config: TissueConfig,
    rng: np.random.Generator | None = None,
) -> SigmoidFieldParams:
    """Sample a hard two-rectangle step field: same (E_left, E_right, x0)
    distribution as the sigmoid case, with k = inf (the step limit)."""
    return replace(sample_sigmoid_params(config, rng), k=math.inf)


def sigmoid_column_stiffness(
    config: TissueConfig,
    params: SigmoidFieldParams,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluate the sigmoid field at every mesh-column centre.

    Returns:
        column_boundaries: (nx+1,) x-coordinates of the column edges
        column_E: (nx,) Young's modulus per column
    """
    nx, _ = grid_counts(config)
    column_boundaries = np.linspace(0.0, config.width, nx + 1)
    centers = 0.5 * (column_boundaries[:-1] + column_boundaries[1:])
    return column_boundaries, params.young_modulus(centers)


def build_tissue_sigmoid(
    config: TissueConfig,
    params: SigmoidFieldParams | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[SoftShapeFiniteElement, SigmoidFieldParams, np.ndarray]:
    """
    Build a rectangular FEM tissue phantom whose stiffness follows the
    sigmoid profile E(x) = E_left + (E_right - E_left) * sigmoid(k*(x - x0)),
    discretized to one E value per mesh column (each column is a perfect
    thin rectangle).

    Returns:
        tissue: SoftShapeFiniteElement
        params: the SigmoidFieldParams used (ground-truth label)
        column_E: (nx,) per-column Young's modulus actually assigned
    """
    if params is None:
        params = sample_sigmoid_params(config, rng)

    column_boundaries, column_E = sigmoid_column_stiffness(config, params)
    tissue = _build_fem_shape(config, default_young_modulus=float(column_E.mean()))
    tissue.set_zone_stiffness(list(column_boundaries), list(column_E))
    return tissue, params, column_E


# ---------------------------------------------------------------------------
# Probe scan positions
# ---------------------------------------------------------------------------

def scan_probe_positions(
    config: TissueConfig,
    n_positions: int,
    margin: float = 0.1,
) -> list[float]:
    """
    Return n_positions evenly-spaced x-coordinates across the tissue width,
    avoiding the edges by `margin`.
    """
    return list(np.linspace(margin, config.width - margin, n_positions))
