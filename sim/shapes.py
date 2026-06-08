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
        for t in self.triangles:
            for i in range(3):
                v1, v2 = int(t[i]), int(t[(i + 1) % 3])
                if (v1, v2) not in self.links and (v2, v1) not in self.links:
                    self.links.append((v1, v2))

        self.edges = self._compute_edges()
        self.original_positions = self.vertices.data.clone()

        n_tri = len(self.triangles)
        self.young_modulus = torch.full((n_tri,), default_young_modulus, dtype=torch.float32)
        self.young_modulus += torch.rand(n_tri) * 2 * young_modulus_var - young_modulus_var

        self.poisson_ratio = torch.full((n_tri,), default_poisson_ratio, dtype=torch.float32)
        self.poisson_ratio += torch.rand(n_tri) * 2 * poisson_ratio_var - poisson_ratio_var

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
        for i, tri in enumerate(self.triangles):
            centroid_x = float(vertices_np[tri, 0].mean())
            for z, (x_lo, x_hi) in enumerate(zip(zone_boundaries[:-1], zone_boundaries[1:])):
                if x_lo <= centroid_x < x_hi:
                    self.young_modulus[i] = zone_young_moduli[z]
                    break

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------

    def update_positions(self, dt: float):
        vel = torch.tensor(self.trajectory.step(dt), dtype=torch.float32, requires_grad=False)
        with torch.no_grad():
            self.vertices[self.fixed_vertices_idx] = (
                self.vertices[self.fixed_vertices_idx] + dt * vel
            )

    def internal_energy(self):
        return self._strain_energy()

    def _strain_energy(self):
        """Linear elastic strain energy summed over all triangles."""
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
