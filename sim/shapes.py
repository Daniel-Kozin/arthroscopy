"""
Soft-body shape classes using PyTorch for differentiable FEM.
Ported from the breast-palpation project with minor clean-ups.

SoftShapeFiniteElement: linear elastic FEM with Delaunay triangulation.
SoftShapeSpringMass: spring-mass model (used for the probe).
"""
import numpy as np
import torch
from scipy.spatial import Delaunay

from sim.trajectory import Trajectory, StaticTrajectory


class SoftShape:
    def __init__(
        self,
        vertices: np.ndarray,
        fixed_vertices_idx: list[int],
        trajectory: Trajectory,
        boundary_idx: list[int],
    ):
        self.vertices = torch.tensor(vertices, dtype=torch.float32, requires_grad=True)
        self.initial_vertices = self.vertices.data.clone()
        self.fixed_vertices_idx = fixed_vertices_idx
        self.boundary_idx = boundary_idx
        self.frozen_mask = torch.ones_like(self.vertices, requires_grad=False)
        self.frozen_mask[self.fixed_vertices_idx] = 0
        self.trajectory = trajectory
        # Velocity applied by the most recent update_positions() call. Used by
        # the sensor model (e.g. to pick a Coulomb-friction sliding direction).
        self.last_velocity = (0.0, 0.0)

    def reset_positions(self):
        with torch.no_grad():
            self.vertices.data = self.initial_vertices.clone()


class SoftShapeSpringMass(SoftShape):
    """Spring-mass shape — used for the probe (rigid body approximation)."""

    def __init__(self, vertices, fixed_vertices_idx, trajectory, links, link_weights, boundary_idx):
        super().__init__(vertices, fixed_vertices_idx, trajectory, boundary_idx)
        self.links = links
        self.link_weights = link_weights if link_weights is not None else [0.1] * len(links)
        self.edges = self._compute_edges()
        self.rest_lengths = self._compute_rest_lengths()

    def _compute_edges(self):
        bi = self.boundary_idx
        return [bi[i: i + 2] for i in range(len(bi) - 1)] + [[bi[-1], bi[0]]]

    def _compute_rest_lengths(self):
        return [
            torch.norm(self.vertices[v1] - self.vertices[v2], p=2).detach()
            for v1, v2 in self.links
        ]

    def update_positions(self, dt: float):
        vel = torch.tensor(self.trajectory.step(dt), dtype=torch.float32, requires_grad=False)
        self.last_velocity = (float(vel[0]), float(vel[1]))
        with torch.no_grad():
            self.vertices[self.fixed_vertices_idx] = (
                self.vertices[self.fixed_vertices_idx] + dt * vel
            )

    def internal_energy(self):
        energy = torch.tensor(0.0)
        for (v1, v2), rest_len, w in zip(self.links, self.rest_lengths, self.link_weights):
            cur_len = torch.norm(self.vertices[v1] - self.vertices[v2], p=2)
            energy = energy + 0.5 * w * (cur_len - rest_len) ** 2
        return energy


class SoftShapeFiniteElement(SoftShape):
    """
    Linear elastic FEM shape.

    Material properties are per-triangle.  Call ``set_zone_stiffness`` after
    construction to assign Young's modulus to rectangular zones.
    """

    def __init__(
        self,
        vertices: np.ndarray,
        fixed_vertices_idx: list[int],
        trajectory: Trajectory,
        boundary_idx: list[int],
        default_young_modulus: float = 0.01,
        default_poisson_ratio: float = 0.45,
        young_modulus_var: float = 0.0,
        poisson_ratio_var: float = 0.0,
    ):
        super().__init__(vertices, fixed_vertices_idx, trajectory, boundary_idx)

        tri = Delaunay(vertices)
        self.triangles = tri.simplices

        # Build edge list (no duplicates)
        self.links = []
        seen_links = set()
        for t in self.triangles:
            for i in range(3):
                v1, v2 = int(t[i]), int(t[(i + 1) % 3])
                key = (v1, v2) if v1 < v2 else (v2, v1)
                if key not in seen_links:
                    seen_links.add(key)
                    self.links.append((v1, v2))

        self.edges = self._compute_edges()
        self.original_positions = self.vertices.data.clone()

        n_tri = len(self.triangles)
        self.young_modulus = torch.full((n_tri,), default_young_modulus, dtype=torch.float32)
        self.young_modulus += torch.rand(n_tri) * 2 * young_modulus_var - young_modulus_var

        self.poisson_ratio = torch.full((n_tri,), default_poisson_ratio, dtype=torch.float32)
        self.poisson_ratio += torch.rand(n_tri) * 2 * poisson_ratio_var - poisson_ratio_var

        self._precompute_fem_matrices()

    def _precompute_fem_matrices(self):
        """
        Precompute per-triangle rest areas and strain-displacement matrices B
        (they depend only on the rest configuration) so _strain_energy() can
        run batched every optimizer step instead of looping over triangles.
        """
        self._tri_index = torch.tensor(np.ascontiguousarray(self.triangles), dtype=torch.long)
        v_ref = self.original_positions[self._tri_index]        # (T, 3, 2)
        x, y = v_ref[:, :, 0], v_ref[:, :, 1]                   # (T, 3)

        area = 0.5 * torch.abs(
            (x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0])
            - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0])
        )                                                        # (T,)

        n_tri = v_ref.shape[0]
        B = torch.zeros((n_tri, 3, 6))
        for i in range(3):
            j, k = (i + 1) % 3, (i + 2) % 3
            B[:, 0, 2 * i]     = y[:, j] - y[:, k]
            B[:, 1, 2 * i + 1] = x[:, k] - x[:, j]
            B[:, 2, 2 * i]     = B[:, 1, 2 * i + 1]
            B[:, 2, 2 * i + 1] = B[:, 0, 2 * i]
        B = B / (2 * area)[:, None, None]

        self._tri_areas = area
        self._B = B
        self._ref_flat = v_ref.reshape(n_tri, 6)

    def _compute_edges(self):
        bi = self.boundary_idx
        return [bi[i: i + 2] for i in range(len(bi) - 1)] + [[bi[-1], bi[0]]]

    # ------------------------------------------------------------------
    # Zone stiffness assignment
    # ------------------------------------------------------------------

    def set_zone_stiffness(
        self,
        zone_boundaries: list[float],
        zone_young_moduli: list[float],
    ):
        """
        Assign Young's modulus to triangles based on which x-zone their
        centroid falls in.

        Args:
            zone_boundaries: list of x-coordinates that separate zones.
                Length = n_zones + 1  (e.g. [0.0, 0.4, 0.8, 1.2, 1.6, 2.0])
            zone_young_moduli: list of E values, one per zone.
                Length = n_zones = len(zone_boundaries) - 1
        """
        assert len(zone_young_moduli) == len(zone_boundaries) - 1

        vertices_np = self.original_positions.numpy()
        centroid_x = vertices_np[self.triangles, 0].mean(axis=1)      # (T,)
        bounds = np.asarray(zone_boundaries, dtype=np.float64)
        zone_idx = np.searchsorted(bounds, centroid_x, side="right") - 1
        valid = (zone_idx >= 0) & (zone_idx < len(zone_young_moduli))

        moduli = np.asarray(zone_young_moduli, dtype=np.float32)
        tri_ids = np.nonzero(valid)[0]
        self.young_modulus[torch.from_numpy(tri_ids)] = torch.from_numpy(
            moduli[zone_idx[tri_ids]]
        )

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------

    def update_positions(self, dt: float):
        vel = torch.tensor(self.trajectory.step(dt), dtype=torch.float32, requires_grad=False)
        self.last_velocity = (float(vel[0]), float(vel[1]))
        with torch.no_grad():
            self.vertices[self.fixed_vertices_idx] = (
                self.vertices[self.fixed_vertices_idx] + dt * vel
            )

    def internal_energy(self):
        return self._strain_energy()

    def _strain_energy(self):
        """
        Linear elastic strain energy summed over all triangles (batched).

        Numerically equivalent to _strain_energy_reference() (see test suite)
        but runs as a handful of batched tensor ops instead of a Python loop —
        required for fine meshes (e.g. 128 columns ≈ 6.6k triangles).
        """
        disp = self.vertices[self._tri_index].reshape(-1, 6) - self._ref_flat   # (T, 6)
        strain = torch.einsum("tij,tj->ti", self._B, disp)                      # (T, 3)

        E, nu = self.young_modulus, self.poisson_ratio
        c = E / (1 - nu ** 2)
        stress_0 = c * (strain[:, 0] + nu * strain[:, 1])
        stress_1 = c * (nu * strain[:, 0] + strain[:, 1])
        stress_2 = c * ((1 - nu) / 2) * strain[:, 2]

        stress_dot_strain = (
            stress_0 * strain[:, 0] + stress_1 * strain[:, 1] + stress_2 * strain[:, 2]
        )
        return (0.5 * stress_dot_strain * 2 * self._tri_areas).sum()

    def _strain_energy_reference(self):
        """Original per-triangle loop implementation. Kept as the ground truth
        for testing the batched _strain_energy(); do not use in hot paths."""
        energy = torch.tensor(0.0)
        for tri, E, nu in zip(self.triangles, self.young_modulus, self.poisson_ratio):
            v_cur = self.vertices[tri]           # (3, 2)
            v_ref = self.original_positions[tri] # (3, 2)

            area = 0.5 * torch.abs(
                torch.linalg.det(
                    torch.stack([
                        torch.tensor([1.0, v_ref[0, 0], v_ref[0, 1]]),
                        torch.tensor([1.0, v_ref[1, 0], v_ref[1, 1]]),
                        torch.tensor([1.0, v_ref[2, 0], v_ref[2, 1]]),
                    ])
                )
            )

            # Strain-displacement matrix B  (3×6)
            B = torch.zeros((3, 6))
            for i in range(3):
                j = (i + 1) % 3
                k = (i + 2) % 3
                B[0, 2 * i]     = v_ref[j, 1] - v_ref[k, 1]
                B[1, 2 * i + 1] = v_ref[k, 0] - v_ref[j, 0]
                B[2, 2 * i]     = B[1, 2 * i + 1]
                B[2, 2 * i + 1] = B[0, 2 * i]
            B = B / (2 * area)

            # Constitutive matrix D
            D = (E / (1 - nu ** 2)) * torch.tensor([
                [1,  nu, 0],
                [nu, 1,  0],
                [0,  0,  (1 - nu) / 2],
            ])

            disp = v_cur.flatten() - v_ref.flatten()
            strain = B @ disp
            stress = D @ strain
            energy = energy + 0.5 * torch.dot(stress, strain) * 2 * area

        return energy


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def create_circular_probe(
    radius: float,
    num_points: int,
    x_center: float,
    y_center: float,
    trajectory: Trajectory | None = None,
) -> SoftShapeSpringMass:
    """Circular probe with all vertices fixed (rigid body)."""
    if trajectory is None:
        trajectory = StaticTrajectory(1.0, x_center, y_center)

    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    vertices = [(radius * np.cos(a) + x_center, radius * np.sin(a) + y_center) for a in angles]
    idx = list(range(num_points))
    return SoftShapeSpringMass(
        vertices=vertices,
        fixed_vertices_idx=idx,
        trajectory=trajectory,
        links=[],
        link_weights=None,
        boundary_idx=idx,
    )
