"""
Core soft-body simulation engine.
Ported from soft_object_sim.py with clean-ups and an extended sensor model.

Key change vs. breast project:
  - get_sensor_reading() returns (Fx, Fy, Mz) instead of just (Fx, Fy).
    Mz is the moment of the contact forces about the probe center.
"""
import math
import os
import pickle
from typing import Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from sim.shapes import SoftShape
from sim.configs import SimulationConfig


class BadAngleError(Exception):
    """Raised when a tilted poke never makes contact with the tissue."""


class SoftObjectSimulation:
    """
    Simulates interaction between a soft tissue (shapes[0]) and a probe (shapes[1]).

    The equilibrium at each frame is found by minimizing total strain + collision energy
    using Adam.
    """

    def __init__(self, shapes: list[SoftShape], config: SimulationConfig):
        assert len(shapes) == 2, "Expects exactly [tissue, probe]"
        self.shapes = shapes
        self.config = config
        self.dt = config.dt
        self.k_collision = config.collision_spring_constant

        # Precompute tissue geometry for collision detection.
        # We identify the top-edge vertices by their original y position so we can
        # track the *deformed* surface height during optimisation.
        orig = shapes[0].original_positions.numpy()
        y_max = float(orig[:, 1].max())
        self._top_edge_idx = np.where(np.isclose(orig[:, 1], y_max))[0]
        self._tissue_x_min = float(orig[:, 0].min())
        self._tissue_x_max = float(orig[:, 0].max())
        self._tissue_y_max = y_max

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _point_in_polygon(point, polygon) -> bool:
        """Ray-casting polygon containment test."""
        x, y = point
        inside = False
        n = len(polygon)
        px, py = polygon[0]
        for i in range(1, n + 1):
            qx, qy = polygon[i % n]
            if min(py, qy) < y <= max(py, qy) and x <= max(px, qx):
                if py != qy:
                    xints = (y - py) * (qx - px) / (qy - py) + px
                if px == qx or x <= xints:
                    inside = not inside
            px, py = qx, qy
        return inside

    @staticmethod
    def _dist_point_to_segment(point, seg):
        """
        Returns (distance, vector_from_point_to_closest_point_on_seg).
        """
        (x1, y1), (x2, y2) = seg
        px, py = point
        dx, dy = x2 - x1, y2 - y1
        if dx == dy == 0:
            d = torch.hypot(px - x1, py - y1)
            v = torch.nn.functional.normalize(torch.tensor([px - x1, py - y1], dtype=torch.float32), dim=0)
            return d, v
        t = ((px - x1) * dx + (py - y1) * dy) / (dx ** 2 + dy ** 2)
        t = torch.clamp(torch.tensor(t, dtype=torch.float32), 0.0, 1.0)
        cx, cy = x1 + t * dx, y1 + t * dy
        d = torch.hypot(px - cx, py - cy)
        v = torch.tensor([cx - px, cy - py], dtype=torch.float32)
        return d, v

    def _closest_edge(self, point, polygon):
        best_dist = float("inf")
        best_vec = None
        n = len(polygon)
        for i in range(n):
            seg = (polygon[i], polygon[(i + 1) % n])
            dist, vec = self._dist_point_to_segment(point, seg)
            if dist < best_dist:
                best_dist = dist
                best_vec = vec
        return best_dist, best_vec

    # ------------------------------------------------------------------
    # Sensor model
    # ------------------------------------------------------------------

    def get_sensor_reading(self) -> np.ndarray:
        """
        Returns a (N_probe_verts, 3) array of [Fx, Fy, Mz] per probe vertex.

        Force model: for each probe vertex that is below the deformed tissue
        surface, the tissue pushes the probe with a normal reaction
        Fn = 2k * penetration (consistent with the collision energy
        E = k * penetration²), where penetration is still measured vertically
        (surface_y - vertex_y).

        Tilted-normal decomposition (probe_angle_deg = theta)
        ─────────────────────────────────────────────────────
        When the probe pokes at a tilt theta from vertical (0 = straight
        down, +theta tilts the DESCENT direction toward +x, so the probe
        travels along (+sin(theta), -cos(theta))), we assume the soft
        tissue's effective contact normal tilts by theta as well, and the
        normal reaction (Newton's 3rd law) points opposite to the probe's
        direction of travel:
            fy = Fn * cos(theta)
            fx_normal = -Fn * sin(theta)
        At theta = 0 this reduces exactly to the legacy model (fx_normal = 0,
        fy = Fn).

        If `friction_coeff` (mu) > 0, a Coulomb tangential reaction
        Fx_friction = -mu * Fy * sign(vx_probe) opposes the probe's
        horizontal sliding motion (vx_probe == 0 -> Fx_friction = 0), and is
        added to fx_normal. (No tilt-dependent friction model — out of
        scope.)

        Sensor wrench transform (probe tip -> remote F/T sensor on the shaft)
        ─────────────────────────────────────────────────────────────────────
        The real F/T sensor sits on the handle, a distance `shaft_length` (L)
        up the probe shaft from the contact tip — not at the tip itself. For
        a rigid probe in quasi-static equilibrium this only changes where the
        moment is taken about; it does NOT change the net force:

          * Force is unchanged: ΣF=0 has no distance term, so
            F_sensor = F_contact = Σ f_n. We do NOT scale/attenuate forces
            by shaft length.
          * Moment depends on sensor location: Mz_sensor is computed directly
            about the sensor point s = probe_centre + (0, L), using the
            standard right-hand-rule convention Mz = (r x F)_z, i.e.
              Mz_sensor = Σ [ (p_n_x - s_x) * f_n_y - (p_n_y - s_y) * f_n_x ]
            (Equivalently Mz_sensor = Mz_about_tip + r × F.)
          * With L=0, s == probe_centre and this reduces exactly to the
            original tip-centred formula (legacy/backward-compatible).

        Mz interpretation: zero when contact is symmetric (uniform tissue,
        centred poke, no friction), non-zero when the probe straddles a
        stiffness zone boundary (asymmetric Fy) or when friction (Fx != 0)
        is combined with a nonzero shaft length L.

        Note: this intentionally does NOT model the cannula/portal reaction
        (a sideways constraint force on the shaft from the trocar), which
        would break F_sensor = F_contact. Out of scope for now.
        """
        top_edge = self.shapes[0].vertices[self._top_edge_idx].detach().numpy()
        probe_verts = self.shapes[1].vertices[self.shapes[1].boundary_idx].detach().numpy()
        probe_center = probe_verts.mean(axis=0)
        noise_std = self.config.probe_force_noise_std
        mu = self.config.friction_coeff
        theta = math.radians(self.config.probe_angle_deg)
        cos_theta, sin_theta = math.cos(theta), math.sin(theta)

        # Sensor point: offset from the probe centre along the shaft's
        # vertical axis. L=0 -> sensor co-located with the tip (legacy).
        sensor_x = probe_center[0]
        sensor_y = probe_center[1] + self.config.shaft_length

        # Coulomb friction direction: opposes the probe's horizontal sliding
        # velocity. No horizontal motion -> no tangential (static) force.
        probe_vx = self.shapes[1].last_velocity[0]
        slip_sign = np.sign(probe_vx)

        readings = []
        for v in probe_verts:
            vx, vy = float(v[0]), float(v[1])
            if self._tissue_x_min <= vx <= self._tissue_x_max:
                nearest = int(np.abs(top_edge[:, 0] - vx).argmin())
                surface_y = float(top_edge[nearest, 1])
                penetration = surface_y - vy
            else:
                penetration = -1.0

            if penetration > 0:
                fn = 2.0 * self.k_collision * penetration    # normal reaction magnitude
                fy = fn * cos_theta
                fx = -fn * sin_theta + (-mu * fy * slip_sign)  # tilt + Coulomb friction
            else:
                fx, fy = 0.0, 0.0

            fx += np.random.randn() * noise_std
            fy += np.random.randn() * noise_std

            # Moment about the SENSOR point (not the probe tip). Forces are
            # unchanged; only this lever arm shifts with shaft_length.
            dx = vx - sensor_x
            dy = vy - sensor_y
            # Standard 2D moment: Mz = (r x F)_z = dx*fy - dy*fx,
            # with (dx, dy) = vertex - sensor_point. (Previous code used the
            # negated convention fx*dy - fy*dx; flipped here so Mz matches a
            # real sensor and composes correctly with r x F in any future
            # wrench transform.)
            mz = float(dx * fy - dy * fx)
            readings.append([fx, fy, mz])

        return np.array(readings, dtype=np.float32)

    def get_aggregated_sensor(self) -> np.ndarray:
        """
        Aggregate per-vertex readings into a single (3,) vector [Fx_total, Fy_total, Mz_total].
        This mimics a single 6-DOF sensor at the probe handle.
        """
        readings = self.get_sensor_reading()
        return readings.sum(axis=0)  # (3,)

    def get_contact_point(self) -> np.ndarray | None:
        """
        Returns the centroid of probe vertices currently inside the tissue,
        or None when the probe is not in contact.

        Uses axis-aligned bounding box of the tissue rest configuration instead
        of the polygon test (boundary vertices are not stored in perimeter order,
        making the ray-casting test unreliable).
        """
        orig = self.shapes[0].original_positions.numpy()
        x_min, y_min = orig.min(axis=0)
        x_max, y_max = orig.max(axis=0)

        probe_verts = (
            self.shapes[1].vertices[self.shapes[1].boundary_idx].detach().numpy()
        )
        inside = [
            v for v in probe_verts
            if x_min <= v[0] <= x_max and y_min <= v[1] <= y_max
        ]
        if not inside:
            return None
        return np.mean(inside, axis=0)

    # ------------------------------------------------------------------
    # Energy minimization
    # ------------------------------------------------------------------

    def _minimize_energy(self, steps: int = 100):
        optimizer = torch.optim.Adam(
            [s.vertices for s in self.shapes],
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta_1, self.config.adam_beta_2),
        )

        def closure():
            optimizer.zero_grad()
            energy = sum(s.internal_energy() for s in self.shapes)

            # Collision penalty: for every probe vertex that sits below the
            # deformed tissue surface, add  k * penetration².
            #
            # top_edge_verts is re-sliced inside closure so a fresh computation
            # graph is built on every call (avoids retain_graph errors).
            # surface_y is a real tensor → gradient flows back to tissue. ✓
            # The probe vertex gradient is zeroed by frozen_mask below. ✓
            top_edge_verts = self.shapes[0].vertices[self._top_edge_idx]
            for v in self.shapes[1].vertices[self.shapes[1].boundary_idx]:
                vx = v[0].item()
                if not (self._tissue_x_min <= vx <= self._tissue_x_max):
                    continue
                nearest = int((top_edge_verts[:, 0].detach() - vx).abs().argmin())
                surface_y = top_edge_verts[nearest, 1]   # tensor — keeps grad
                penetration = surface_y - v[1]
                if penetration.item() > 0:
                    energy = energy + self.k_collision * penetration ** 2

            energy.backward()

            # Zero gradients for frozen DOF
            with torch.no_grad():
                for s in self.shapes:
                    if s.vertices.grad is not None:
                        s.vertices.grad[torch.isnan(s.vertices.grad)] = 0.0
                        s.vertices.grad *= s.frozen_mask
            return energy

        if self.config.warmup:
            optimizer.param_groups[0]["lr"] = self.config.learning_rate * 2.0
            for _ in range(5):
                optimizer.step(closure)
            optimizer.param_groups[0]["lr"] = self.config.learning_rate

        best = 1e10
        no_improve = 0
        threshold_pct = 0.001

        for _ in range(steps):
            energy = optimizer.step(closure)
            improvement = best - energy.item()
            if improvement > best * threshold_pct:
                best = energy.item()
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= 5:
                return
        print("Warning: optimizer did not converge.")

    def _closest_edge_tensor(self, point, polygon):
        """Version that keeps tensors for backprop through collision penalty."""
        best_dist = torch.tensor(float("inf"))
        best_vec = None
        n = len(polygon)
        for i in range(n):
            seg = (polygon[i], polygon[(i + 1) % n])
            dist, vec = self._dist_point_to_segment(
                (point[0].item(), point[1].item()),
                ((seg[0][0].item(), seg[0][1].item()), (seg[1][0].item(), seg[1][1].item())),
            )
            if dist < best_dist:
                best_dist = dist
                best_vec = vec
        return None, best_dist, best_vec

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def reset_positions(self):
        for s in self.shapes:
            s.reset_positions()

    def step(self):
        """Advance one frame: move probe, then minimize energy."""
        for s in self.shapes:
            s.update_positions(self.dt)
        self._minimize_energy(steps=self.config.steps)

        tissue_verts = self.shapes[0].vertices.detach().numpy()
        probe_verts = self.shapes[1].vertices.detach().numpy()
        return tissue_verts, probe_verts

    # ------------------------------------------------------------------
    # Full simulation run
    # ------------------------------------------------------------------

    def run(self, save_folder: Optional[str] = None) -> list[np.ndarray]:
        """
        Run all frames, collect sensor readings.

        Returns:
            List of aggregated sensor arrays, one per frame. Shape: (n_frames, 3).
        """
        if save_folder:
            os.makedirs(save_folder, exist_ok=True)

        sensor_log: list[np.ndarray] = []

        for frame in range(self.config.frames):
            self.step()

            if self.config.save_vectors:
                sensor_log.append(self.get_aggregated_sensor())

            if self.config.save_images:
                self._save_frame_image(save_folder, frame)

        if self.config.save_vectors and save_folder:
            with open(os.path.join(save_folder, "sensor_log.pkl"), "wb") as f:
                pickle.dump(sensor_log, f)

        return sensor_log

    # ------------------------------------------------------------------
    # Visualisation helpers
    # ------------------------------------------------------------------

    def _save_frame_image(self, folder: str, frame: int):
        fig, ax = plt.subplots(figsize=(8, 3))
        tissue = self.shapes[0]
        probe = self.shapes[1]

        # Draw tissue triangles coloured by stiffness
        if hasattr(tissue, "triangles") and hasattr(tissue, "young_modulus"):
            E = tissue.young_modulus.numpy()
            E_norm = (E - E.min()) / (E.max() - E.min() + 1e-8)
            for tri, e in zip(tissue.triangles, E_norm):
                x = tissue.vertices[tri][:, 0].detach().numpy()
                y = tissue.vertices[tri][:, 1].detach().numpy()
                ax.fill(x, y, color=plt.cm.viridis(e), alpha=0.7)

        # Draw probe
        pv = probe.vertices.detach().numpy()
        ax.plot(pv[:, 0], pv[:, 1], "ro", markersize=4)

        ax.set_xlim(-0.2, 2.2)
        ax.set_ylim(-0.3, 0.8)
        ax.set_aspect("equal")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(folder, f"frame_{frame:04d}.png"), dpi=150)
        plt.close(fig)

    def draw_tissue(self, save_path: Optional[str] = None):
        """Static visualisation of the tissue stiffness map."""
        tissue = self.shapes[0]
        fig, ax = plt.subplots(figsize=(8, 3))

        if hasattr(tissue, "triangles") and hasattr(tissue, "young_modulus"):
            E = tissue.young_modulus.numpy()
            E_norm = (E - E.min()) / (E.max() - E.min() + 1e-8)
            for tri, e in zip(tissue.triangles, E_norm):
                x = tissue.vertices[tri][:, 0].detach().numpy()
                y = tissue.vertices[tri][:, 1].detach().numpy()
                ax.fill(x, y, color=plt.cm.viridis(e), alpha=0.9, edgecolor="none")

        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
        sm.set_array(tissue.young_modulus.numpy())
        plt.colorbar(sm, ax=ax, label="Young's modulus (E)")
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title("Tissue stiffness map")
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150)
            plt.close(fig)
        else:
            plt.show()
            plt.close(fig)
