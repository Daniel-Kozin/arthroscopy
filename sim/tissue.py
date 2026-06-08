"""
Arthroscopy tissue phantom: a rectangle with K stiffness zones.

The rectangle lies in [0, width] x [0, height].
Bottom edge is fixed (Dirichlet BC).
The probe approaches from above (positive y direction).

Zone layout (n_zones=5, width=2.0):
  zone 0: x in [0.0, 0.4)
  zone 1: x in [0.4, 0.8)
  ...
  zone 4: x in [1.6, 2.0]

Each zone gets an independently sampled Young's modulus. This is the label
the model must predict from probe force readings.
"""
import numpy as np

from sim.configs import TissueConfig
from sim.shapes import SoftShapeFiniteElement
from sim.trajectory import StaticTrajectory


def build_rectangle_mesh(
    width: float,
    height: float,
    grid_size: float,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """
    Generate a uniform grid of vertices inside and on the boundary of a rectangle.

    Returns:
        boundary_vertices: list of (x, y) on the perimeter
        interior_vertices: list of (x, y) strictly inside
    """
    xs = np.arange(0.0, width + grid_size * 0.5, grid_size)
    ys = np.arange(0.0, height + grid_size * 0.5, grid_size)

    boundary = []
    interior = []

    for x in xs:
        for y in ys:
            x = float(np.clip(x, 0.0, width))
            y = float(np.clip(y, 0.0, height))
            on_boundary = (
                np.isclose(x, 0.0)
                or np.isclose(x, width)
                or np.isclose(y, 0.0)
                or np.isclose(y, height)
            )
            if on_boundary:
                if (x, y) not in boundary:
                    boundary.append((x, y))
            else:
                interior.append((x, y))

    return boundary, interior


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


def build_tissue(
    config: TissueConfig,
    zone_young_moduli: list[float] | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[SoftShapeFiniteElement, list[float]]:
    """
    Build a rectangular FEM tissue phantom.

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

    boundary_verts, interior_verts = build_rectangle_mesh(config.width, config.height, config.grid_size)
    all_verts = np.array(boundary_verts + interior_verts, dtype=np.float64)

    # Fixed vertices: bottom edge (y ≈ 0)
    fixed_idx = [i for i, v in enumerate(all_verts) if np.isclose(v[1], 0.0)]

    # Boundary indices: all perimeter vertices (used for collision detection)
    boundary_set = set(map(tuple, [v for v in all_verts if
        np.isclose(v[0], 0.0) or np.isclose(v[0], config.width) or
        np.isclose(v[1], 0.0) or np.isclose(v[1], config.height)
    ]))
    boundary_idx = [i for i, v in enumerate(all_verts) if tuple(v) in boundary_set]

    static_traj = StaticTrajectory(T=1.0)

    tissue = SoftShapeFiniteElement(
        vertices=all_verts,
        fixed_vertices_idx=fixed_idx,
        trajectory=static_traj,
        boundary_idx=boundary_idx,
        default_young_modulus=float(np.mean(zone_young_moduli)),
        default_poisson_ratio=config.poisson_ratio,
        young_modulus_var=config.young_modulus_var,
        poisson_ratio_var=config.poisson_ratio_var,
    )

    # Assign per-zone stiffness
    zone_width = config.width / config.n_zones
    zone_boundaries = [i * zone_width for i in range(config.n_zones + 1)]
    tissue.set_zone_stiffness(zone_boundaries, zone_young_moduli)

    return tissue, zone_young_moduli


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
