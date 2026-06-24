"""
visualize_sim.py — Arthroscopy probe simulation viewer.

Usage:
    python visualize_sim.py --x_pos 1.0 --seed 0 --n_frames 16

Physics recap:

  TISSUE MODEL
  ─────────────
  Rectangle phantom (2.0 × 0.4) split into N zones.
  Each zone has a different Young's modulus E (stiffness).
  Bottom edge is clamped. All other vertices are free.

  FEM EQUILIBRIUM PER FRAME
  ──────────────────────────
  At each frame the tissue reaches quasi-static equilibrium by minimising:
    U_total = U_elastic(E, ε)  +  k · Σ max(0, surface_y − probe_y)²
              ^^^^^^^^^^^^          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
              FEM strain energy     collision penalty (probe pressing in)

  Larger E → tissue resists deformation → surface barely moves → probe sinks
  deeper → larger collision penalty gradient → larger reaction force.

  SENSOR — Newton's 3rd law, tilted-normal decomposition
  ────────────────────────────────────────────────────────
    Fn_i = 2k · penetration_i        (normal reaction magnitude per vertex)
    Fy_i = Fn_i · cos(theta)         (theta = poke tilt from vertical)
    Fx_i = -Fn_i · sin(theta)        (reaction opposes the probe's travel direction)
    Fx   = Σ Fx_i, Fy = Σ Fy_i       (total force at sensor on handle)
    Mz   = Σ (dx_i·Fy_i − dy_i·Fx_i) (torque about probe centre, 2D only)

  Fy LARGE  → tissue is stiff (little deformation → deep penetration).
  Fx ≠ ZERO → probe is poking at a tilt (theta != 0).
  Mz ZERO   → symmetric contact, probe fully inside one zone.
  Mz ≠ ZERO → probe straddles two zones of different stiffness, or tilt +
              shaft offset introduces an asymmetric lever arm.

  Note: this is a 2D simulation, so there is exactly ONE torque component
  (Mz, about the out-of-plane z-axis). A 3D probe would additionally have
  Mx and My, which do not exist here.
"""
import argparse
import dataclasses
import math
import sys

import numpy as np
import pyrallis
import matplotlib

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from sim.configs import ExperimentConfig, TissueConfig, ProbeConfig, SimulationConfig
from sim.tissue import build_tissue
from sim.shapes import create_circular_probe
from sim.trajectory import TwoPointTrajectory, poke_trajectory_points
from sim.simulation import SoftObjectSimulation


# ──────────────────────────────────────────────────────────────────────────────
# Fixed physical constants
# ──────────────────────────────────────────────────────────────────────────────

# Probe / simulation defaults for run_poke() when called directly (without
# going through main()/the config file).
_DEFAULT_PROBE = ProbeConfig()
_DEFAULT_SIM   = SimulationConfig()

HANDLE_OFFSET   = 0.28   # handle centre is this far above probe centre
HANDLE_W        = 0.10
HANDLE_H        = 0.045

# Fixed y-axis ranges for the Fx/Fy/Mz time-series subplots (sim units).
# Kept constant across runs so different seeds/angles/positions are visually
# comparable. Chosen to comfortably bound typical values at the default
# config; retune if collision_spring_constant / young_modulus / noise change
# significantly.
FX_YLIM = (-0.0030, 0.0030)
FY_YLIM = (-0.0005, 0.005)
MZ_YLIM = (-0.0001, 0.0001)


# ──────────────────────────────────────────────────────────────────────────────
# Simulation
# ──────────────────────────────────────────────────────────────────────────────

def _make_sim(tissue, x_start, y_start, x_end, y_end, n_frames, probe_radius, angle_deg, sim_cfg: SimulationConfig):
    traj = TwoPointTrajectory(
        T=sim_cfg.dt * n_frames,
        x0=x_start, y0=y_start,
        x1=x_end, y1=y_end,
    )
    probe = create_circular_probe(
        radius=probe_radius, num_points=8,
        x_center=x_start, y_center=y_start,
        trajectory=traj,
    )
    scfg = dataclasses.replace(
        sim_cfg,
        frames=n_frames,
        probe_angle_deg=angle_deg,
        save_vectors=False,
        save_images=False,
        save_video=False,
    )
    return SoftObjectSimulation([tissue, probe], scfg)


def run_poke(
    x_pos: float,
    seed: int = 0,
    n_down: int = 8,
    n_up: int = 8,
    n_zones: int = 5,
    probe_radius: float = _DEFAULT_PROBE.radius,
    hover_height: float = _DEFAULT_PROBE.hover_height,
    tip_penetration_depth: float = _DEFAULT_PROBE.tip_penetration_depth,
    angle_deg: float = 0.0,
    sim_cfg: SimulationConfig = _DEFAULT_SIM,
):
    """
    Run a full poke (down + up) at x_pos.

    hover_height: how far above the tissue surface the probe starts/ends.
    tip_penetration_depth: how far below the tissue surface the probe TIP
        (its lowest point, i.e. centre - radius) goes at the bottom of the poke.
    angle_deg: tilt of the poke direction from vertical (degrees). 0 = straight
        down. Positive tilts the approach direction toward +x.
    sim_cfg: physics/noise parameters (collision stiffness, sensor noise,
        friction, etc.) — see sim.configs.SimulationConfig. Defaults to the
        dataclass defaults; main() passes the loaded experiment config's
        simulation section so the visualization matches the dataset-gen physics.

    Returns list of per-frame dicts, the tissue object, zone labels, and config.
    """
    np.random.seed(seed)          # seeds global RNG used by sensor noise
    rng  = np.random.default_rng(seed)
    tcfg = TissueConfig(n_zones=n_zones)
    tissue, zone_label_list = build_tissue(tcfg, rng=rng)
    zone_label = np.array(zone_label_list, dtype=np.float32)

    # center_penetration_depth converts the configured TIP penetration depth
    # to the probe CENTRE's vertical travel (poke_trajectory_points works in
    # terms of the centre).
    center_penetration_depth = tip_penetration_depth - probe_radius
    pts = poke_trajectory_points(
        x_pos, tcfg.height, hover_height, center_penetration_depth, angle_deg=angle_deg
    )
    (x_hover, y_hover), (x_bottom, y_bottom) = pts
    total = n_down + n_up

    sim_down = _make_sim(tissue, x_hover, y_hover, x_bottom, y_bottom, n_down, probe_radius, angle_deg, sim_cfg)
    frames = []
    for i in range(n_down):
        sys.stdout.write(f"\r  frame {i+1}/{total} (pressing down)")
        sys.stdout.flush()
        tv, pv = sim_down.step()
        frames.append({
            "tissue_verts": tv.copy(),
            "probe_verts":  pv.copy(),
            "sensor":       sim_down.get_aggregated_sensor().copy(),
            "contact_pt":   sim_down.get_contact_point(),
            "phase":        "down",
        })

    sim_up = _make_sim(tissue, x_bottom, y_bottom, x_hover, y_hover, n_up, probe_radius, angle_deg, sim_cfg)
    for i in range(n_up):
        sys.stdout.write(f"\r  frame {n_down+i+1}/{total} (retracting) ")
        sys.stdout.flush()
        tv, pv = sim_up.step()
        frames.append({
            "tissue_verts": tv.copy(),
            "probe_verts":  pv.copy(),
            "sensor":       sim_up.get_aggregated_sensor().copy(),
            "contact_pt":   sim_up.get_contact_point(),
            "phase":        "up",
        })

    print()
    return frames, tissue, zone_label, tcfg


# ──────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ──────────────────────────────────────────────────────────────────────────────

def _draw_tissue(ax, tissue, frame_data, e_min: float, e_max: float):
    """Fill triangles with viridis coloured by Young's modulus on a fixed scale."""
    E    = tissue.young_modulus.numpy()
    E_n  = (E - e_min) / (e_max - e_min + 1e-12)
    cmap = matplotlib.colormaps["viridis"]
    verts = frame_data["tissue_verts"]
    for tri, en in zip(tissue.triangles, E_n):
        ax.fill(verts[tri, 0], verts[tri, 1],
                fc=cmap(float(np.clip(en, 0, 1))), ec="none", alpha=0.85, zorder=2)


def _draw_probe(ax, frame_data, in_contact: bool, probe_radius: float):
    """Draw probe tip circle, shaft and handle with sensor marker."""
    pv  = frame_data["probe_verts"]
    pc  = pv.mean(axis=0)
    ang = np.linspace(0, 2 * np.pi, 64)

    tip_col = "#ff6b5b" if in_contact else "#a7b3bd"

    # Soft glow behind the tip — brighter / larger when in contact, to draw
    # the eye to the moment of contact without a jarring color jump.
    glow_scale = 1.9 if in_contact else 1.35
    ax.fill(pc[0] + (probe_radius * glow_scale) * np.cos(ang),
            pc[1] + (probe_radius * glow_scale) * np.sin(ang),
            fc=tip_col, ec="none", alpha=0.16, zorder=4)

    ax.fill(pc[0] + probe_radius * np.cos(ang),
            pc[1] + probe_radius * np.sin(ang),
            fc=tip_col, ec="#2c2c3a", lw=1.1, alpha=0.92, zorder=5)
    # Small highlight to give the tip a touch of 3D shine.
    ax.fill(pc[0] + probe_radius * 0.35 * np.cos(ang) - probe_radius * 0.25,
            pc[1] + probe_radius * 0.35 * np.sin(ang) + probe_radius * 0.3,
            fc="#ffffff", ec="none", alpha=0.25, zorder=6)

    handle_cx = pc[0]
    handle_cy = pc[1] + HANDLE_OFFSET
    ax.plot([pc[0], handle_cx],
            [pc[1] + probe_radius, handle_cy - HANDLE_H / 2],
            color="#7f8fa6", lw=2.5, solid_capstyle="round", zorder=4)
    ax.add_patch(mpatches.FancyBboxPatch(
        (handle_cx - HANDLE_W / 2, handle_cy - HANDLE_H / 2),
        HANDLE_W, HANDLE_H,
        boxstyle="round,pad=0.0,rounding_size=0.012",
        fc="#dfe6e9", ec="#7f8c8d", lw=1, zorder=4,
    ))
    ax.plot(handle_cx, handle_cy, "D", color="#3498db", markersize=9,
            mec="#1b4f72", mew=0.8, zorder=6)
    ax.text(handle_cx, handle_cy + HANDLE_H / 2 + 0.025,
            "sensor", fontsize=7, color="#aed6f1", ha="center", va="bottom", zorder=7,
            bbox=dict(boxstyle="round,pad=0.15", fc="#16213e", ec="none", alpha=0.6))

    return handle_cx, handle_cy


def _draw_force_arrow(ax, handle_cx, handle_cy, fx: float, fy: float, max_force: float):
    """Draw arrows at the sensor proportional to Fy (vertical) and Fx (horizontal)."""
    if max_force < 1e-9:
        return
    label_bbox = dict(boxstyle="round,pad=0.15", fc="#16213e", ec="none", alpha=0.65)

    if fy > 1e-9:
        arrow_len = 0.18 * fy / max_force
        ax.annotate(
            "",
            xy=(handle_cx, handle_cy + 0.02 + arrow_len),
            xytext=(handle_cx, handle_cy + 0.02),
            arrowprops=dict(arrowstyle="-|>", color="#2ecc71", lw=2.4),
            zorder=7,
        )
        ax.text(handle_cx + 0.04, handle_cy + 0.02 + arrow_len * 0.55,
                f"Fy={fy:.4f}", fontsize=7.5, color="#2ecc71", va="center", zorder=7,
                bbox=label_bbox)

    if abs(fx) > 1e-9:
        arrow_len = 0.18 * abs(fx) / max_force
        x_sign = 1.0 if fx > 0 else -1.0
        ax.annotate(
            "",
            xy=(handle_cx + x_sign * arrow_len, handle_cy),
            xytext=(handle_cx, handle_cy),
            arrowprops=dict(arrowstyle="-|>", color="#f0932b", lw=2.4),
            zorder=7,
        )
        ax.text(handle_cx + x_sign * arrow_len * 0.55, handle_cy + 0.035,
                f"Fx={fx:.4f}", fontsize=7.5, color="#f0932b", ha="center", zorder=7,
                bbox=label_bbox)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Arthroscopy sim viewer")
    parser.add_argument("--x_pos",    type=float, default=1.0,
                        help="Probe x position (0 – 2.0)")
    parser.add_argument("--seed",     type=int,   default=0,
                        help="RNG seed for tissue stiffness zones")
    parser.add_argument("--frames", type=int,   default=16,
                        help="Total frames, split evenly between pressing and retracting")
    parser.add_argument("--n_zones", type=int,  default=5,
                        help="Number of tissue stiffness zones (rectangles) along the width")
    parser.add_argument("--config", default="configs/default.yaml",
                        help="Experiment config (probe geometry: radius, hover_height, "
                             "tip_penetration_depth, angle range) — see configs/default.yaml")
    parser.add_argument("--angle_deg", type=float, default=None,
                        help="Tilt of the poke direction from vertical, in degrees "
                             "(0 = straight down, positive tilts toward +x). If not "
                             "given, taken from the config's probe_angle_deg, or "
                             "sampled from [angle_min_deg, angle_max_deg] (same as "
                             "generate_dataset.py) if that is also unset.")
    parser.add_argument("--save",   action="store_true",
                        help="Save animation as GIF to output/sim/")
    args = parser.parse_args()

    with open(args.config) as f:
        exp_cfg = pyrallis.load(ExperimentConfig, f)
    probe_radius = exp_cfg.probe.radius
    hover_height = exp_cfg.probe.hover_height
    tip_penetration_depth = exp_cfg.probe.tip_penetration_depth

    x_pos = float(np.clip(args.x_pos, 0.05, 1.95))
    n_down = args.frames // 2
    n_up   = args.frames - n_down
    total_frames = args.frames

    # Decide the poke angle the same way generate_dataset.py does: explicit
    # CLI override > fixed exp_cfg.probe.probe_angle_deg > random sample from
    # [angle_min_deg, angle_max_deg] using the seeded RNG.
    if args.angle_deg is not None:
        angle_deg = args.angle_deg
    elif not math.isnan(exp_cfg.probe.probe_angle_deg):
        angle_deg = exp_cfg.probe.probe_angle_deg
    else:
        angle_deg = float(np.random.default_rng(args.seed).uniform(
            exp_cfg.probe.angle_min_deg, exp_cfg.probe.angle_max_deg
        ))

    print(f"\nRunning poke at x={x_pos:.2f}  seed={args.seed}  frames={total_frames}  angle={angle_deg:.1f}°")
    print("Sensor: Fn = 2k × penetration, Fy = Fn·cos(theta), Fx = -Fn·sin(theta)")

    frames, tissue, zone_label, tcfg = run_poke(
        x_pos, seed=args.seed, n_down=n_down, n_up=n_up, n_zones=args.n_zones,
        probe_radius=probe_radius, hover_height=hover_height,
        tip_penetration_depth=tip_penetration_depth,
        angle_deg=angle_deg,
        sim_cfg=exp_cfg.simulation,
    )

    fx_series = np.array([f["sensor"][0] for f in frames])
    fy_series = np.array([f["sensor"][1] for f in frames])
    mz_series = np.array([f["sensor"][2] for f in frames])
    t_axis    = np.arange(total_frames)
    max_force = max(abs(fx_series).max(), abs(fy_series).max(), 1e-9)

    # Where the probe actually ends up touching the tissue (bottom of the
    # poke), not where it starts — that's what determines the stiffness zone
    # it's sensing.
    center_penetration_depth = tip_penetration_depth - probe_radius
    _, (x_touch, _) = poke_trajectory_points(
        x_pos, tcfg.height, hover_height, center_penetration_depth, angle_deg=angle_deg
    )

    n_zones    = len(zone_label)
    zone_width = tcfg.width / n_zones
    probe_zone = int(np.clip(x_touch / zone_width, 0, n_zones - 1))

    orig_verts   = tissue.original_positions.numpy()
    x_min, y_min = orig_verts.min(axis=0)
    x_max, y_max = orig_verts.max(axis=0)

    # Scene x-range: tissue extent plus wherever the (possibly tilted) probe
    # actually travels, with a small margin — so a large tilt angle near a
    # tissue edge doesn't push the probe outside the visible plot.
    all_probe_x = np.concatenate([f["probe_verts"][:, 0] for f in frames])
    scene_x_min = min(x_min, all_probe_x.min()) - 0.08
    scene_x_max = max(x_max, all_probe_x.max()) + 0.08

    # Absolute E scale — same seed always maps to same color
    e_min_scale = tcfg.young_modulus_min
    e_max_scale = tcfg.young_modulus_max

    zone_boundaries = [zone_width * (i + 1) for i in range(n_zones - 1)]
    near_boundary   = any(abs(x_touch - b) < probe_radius * 2 for b in zone_boundaries)

    print(f"Zone E values: {[f'{e:.4f}' for e in zone_label]}")
    print(f"Probe zone {probe_zone}  E={zone_label[probe_zone]:.4f}"
          f"  {'← near zone boundary' if near_boundary else ''}")
    print(f"Max Fx: {abs(fx_series).max():.5f}  |  Max Fy: {abs(fy_series).max():.5f}  |  Max |Mz|: {abs(mz_series).max():.6f}")
    print("\nShowing animation. Close the window to exit.\n")

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor("#1a1a2e")
    gs = GridSpec(3, 2, figure=fig,
                  width_ratios=[2.2, 1.0],
                  height_ratios=[2.0, 1.0, 1.0],
                  hspace=0.6, wspace=0.38)

    ax_scene = fig.add_subplot(gs[:2, 0])
    ax_zones = fig.add_subplot(gs[2, 0])
    ax_fx    = fig.add_subplot(gs[0, 1])
    ax_fy    = fig.add_subplot(gs[1, 1])
    ax_mz    = fig.add_subplot(gs[2, 1])

    for ax in (ax_scene, ax_zones, ax_fx, ax_fy, ax_mz):
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="#ccc")
        ax.xaxis.label.set_color("#ccc")
        ax.yaxis.label.set_color("#ccc")
        ax.title.set_color("#eee")
        for spine in ax.spines.values():
            spine.set_edgecolor("#555")

    def _setup_series_axis(ax, series, color, ylabel, title, ylim):
        """Shared styling for the Fx/Fy/Mz time-series subplots: phase
        shading, zero line, the data curve, a fixed y-range (so different
        runs are visually comparable), and a "current frame" indicator
        line. Returns that indicator line."""
        ax.grid(True, color="#3a3a55", lw=0.6, alpha=0.5, zorder=0)
        ax.axhline(0, color="#888", lw=0.8, ls="--", zorder=1)
        ax.axvspan(0,      n_down,       alpha=0.12, color="#e74c3c", zorder=0)
        ax.axvspan(n_down, total_frames, alpha=0.12, color="#3498db", zorder=0)
        ax.plot(t_axis, series, color=color, lw=2.4, zorder=3,
                solid_capstyle="round")
        ax.fill_between(t_axis, series, 0, color=color, alpha=0.12, zorder=2)
        ax.set_xlabel("Frame")
        ax.set_ylabel(ylabel)
        ax.set_title(title, color="#eee", fontsize=9, fontweight="bold")
        ax.set_xlim(-0.5, total_frames - 0.5)
        ax.set_ylim(*ylim)

        return ax.axvline(0, color="#fff", lw=1.5, alpha=0.7, zorder=4)

    # ── Fx graph ──────────────────────────────────────────────────────────────
    fx_vline = _setup_series_axis(ax_fx, fx_series, "#e67e22", "Fx  [sim N]", "Fx", FX_YLIM)

    # Peak marker (largest-magnitude Fx, which may be negative).
    peak_fx_f = int(np.argmax(np.abs(fx_series)))
    ax_fx.plot(peak_fx_f, fx_series[peak_fx_f], "o", color="#f1c40f", ms=5, zorder=5)
    ax_fx.annotate(
        f"peak {fx_series[peak_fx_f]:.4f}",
        xy=(peak_fx_f, fx_series[peak_fx_f]),
        xytext=(0.5, 0.95), textcoords="axes fraction",
        fontsize=7.5, color="#f1c40f", ha="center", va="top",
        arrowprops=dict(arrowstyle="->", color="#f1c40f", lw=1),
    )

    # ── Fy graph ──────────────────────────────────────────────────────────────
    fy_vline = _setup_series_axis(ax_fy, fy_series, "#27ae60", "Fy  [sim N]", "Fy", FY_YLIM)

    # Peak marker, placed inside the y-margin so it never collides with the title.
    peak_f = int(np.argmax(fy_series))
    ax_fy.plot(peak_f, fy_series[peak_f], "o", color="#f1c40f", ms=5, zorder=5)
    ax_fy.annotate(
        f"peak {fy_series[peak_f]:.4f}",
        xy=(peak_f, fy_series[peak_f]),
        xytext=(0.5, 0.95), textcoords="axes fraction",
        fontsize=7.5, color="#f1c40f", ha="center", va="top",
        arrowprops=dict(arrowstyle="->", color="#f1c40f", lw=1),
    )

    # ── Mz graph ──────────────────────────────────────────────────────────────
    mz_vline = _setup_series_axis(ax_mz, mz_series, "#9b59b6", "Mz  [sim N·m]", "Mz", MZ_YLIM)

    # ── Stiffness zone bar chart ───────────────────────────────────────────────
    cmap_z = matplotlib.colormaps["viridis"]
    E_n_z  = (zone_label - e_min_scale) / (e_max_scale - e_min_scale + 1e-12)
    bar_colors = [cmap_z(float(np.clip(e, 0, 1))) for e in E_n_z]

    ax_zones.grid(True, axis="y", color="#3a3a55", lw=0.6, alpha=0.5, zorder=0)
    bars = ax_zones.bar(np.arange(n_zones), zone_label, color=bar_colors,
                        edgecolor="#555", linewidth=0.8, zorder=2, width=0.65)
    bars[probe_zone].set_edgecolor("#ffffff")
    bars[probe_zone].set_linewidth(2.5)
    ax_zones.set_xlabel("Zone index")
    ax_zones.set_ylabel("Young's modulus E")
    ax_zones.set_title(f"Tissue stiffness profile — probe in zone {probe_zone}",
                       color="#eee", fontsize=9, fontweight="bold")
    ax_zones.set_xticks(np.arange(n_zones))
    ax_zones.tick_params(colors="#ccc")
    for i, e in enumerate(zone_label):
        ax_zones.text(i, e + zone_label.max() * 0.03, f"{e:.3f}",
                      ha="center", va="bottom", fontsize=7, color="#ccc")

    # Colorbar linking the viridis colors used here and on the tissue in the
    # scene view to the underlying Young's modulus scale.
    sm = ScalarMappable(norm=Normalize(vmin=e_min_scale, vmax=e_max_scale), cmap=cmap_z)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_zones, pad=0.015, fraction=0.045)
    cbar.set_label("stiffness", color="#ccc", fontsize=8)
    cbar.ax.tick_params(colors="#ccc", labelsize=7)
    cbar.outline.set_edgecolor("#555")

    # ── Animation update ──────────────────────────────────────────────────────

    def update(fi: int):
        ax_scene.cla()

        ax_scene.set_facecolor("#233977")
        ax_scene.tick_params(colors="#ccc")
        ax_scene.xaxis.label.set_color("#ccc")
        ax_scene.yaxis.label.set_color("#ccc")
        for spine in ax_scene.spines.values():
            spine.set_edgecolor("#555")
        ax_scene.set_xlim(scene_x_min, scene_x_max)
        ax_scene.set_ylim(y_min - 0.06, y_max + HANDLE_OFFSET + HANDLE_H + 0.12)
        ax_scene.set_aspect("equal")
        ax_scene.set_xlabel("x  [sim m]")
        ax_scene.set_ylabel("y  [sim m]")
        phase_str = "pressing ▼" if frames[fi]["phase"] == "down" else "retracting ▲"
        ax_scene.set_title(
            f"Frame {fi + 1}/{total_frames}  ·  x={x_pos:.2f}  ·  {phase_str}",
            color="#eee", fontsize=9,
        )

        ax_scene.add_patch(mpatches.Rectangle(
            (x_min, y_min), tcfg.width, tcfg.height,
            fc="none", ec="#888", ls="--", lw=1.0, zorder=2,
        ))
        ax_scene.axhline(y_min, xmin=0.05, xmax=0.95,
                         color="#e74c3c", lw=3, alpha=0.45, zorder=3)
        # Placed above the line near its right end, well clear of the
        # Fx/Fy/Mz readout text in the bottom-left corner.
        ax_scene.text(x_max - 0.35, y_min + 0.01,
                      "fixed boundary", fontsize=6.5, color="#e74c3c", alpha=0.8)

        for b in zone_boundaries:
            ax_scene.axvline(b, color="#aaa", lw=0.7, ls=":", alpha=0.55, zorder=2)

        _draw_tissue(ax_scene, tissue, frames[fi], e_min_scale, e_max_scale)

        in_contact = frames[fi]["contact_pt"] is not None
        handle_cx, handle_cy = _draw_probe(ax_scene, frames[fi], in_contact, probe_radius)

        cp = frames[fi]["contact_pt"]
        if cp is not None:
            ax_scene.plot(cp[0], cp[1], "o", color="#e74c3c", markersize=10, zorder=8)
            ax_scene.text(cp[0] + 0.05, cp[1] + 0.025,
                          "contact", fontsize=7.5, color="#e74c3c", zorder=9)

        fx_now = float(frames[fi]["sensor"][0])
        fy_now = float(frames[fi]["sensor"][1])
        mz_now = float(frames[fi]["sensor"][2])
        _draw_force_arrow(ax_scene, handle_cx, handle_cy, fx_now, fy_now, max_force)

        ax_scene.text(
            x_min + 0.02, y_min - 0.045,
            f"Fx = {fx_now:.5f} N   |   Fy = {fy_now:.5f} N   |   Mz = {mz_now:.5f} N·m"
            f"   |   theta = {angle_deg:.1f}°",
            fontsize=8.5, color="#27ae60", zorder=9,
        )

        fx_vline.set_xdata([fi, fi])
        fy_vline.set_xdata([fi, fi])
        mz_vline.set_xdata([fi, fi])
        return []

    anim = FuncAnimation(
        fig, update, frames=total_frames, interval=380, repeat=True,
    )

    # NOTE: tight_layout() warns because ax_scene uses set_aspect("equal"),
    # which it can't reconcile with the GridSpec. Use fixed margins instead.
    fig.subplots_adjust(left=0.06, right=0.97, top=0.96, bottom=0.06)

    if args.save:
        import os
        out_dir = "output/sim"
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"poke_seed{args.seed}_x{x_pos:.2f}_angle{angle_deg:.1f}.gif")
        print(f"Saving GIF → {out_path} ...")
        anim.save(out_path, writer="pillow", fps=3)
        print("Saved.")

    plt.show()


if __name__ == "__main__":
    main()
