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

  SENSOR — Newton's 3rd law
  ─────────────────────────
    Fy_i = 2k · penetration_i   (upward reaction per contact vertex)
    Fy   = Σ Fy_i               (total force at sensor on handle)
    Mz   = Σ Fy_i · (x_i − x_ctr)   (torque about probe centre)

  Fy LARGE  → tissue is stiff (little deformation → deep penetration).
  Mz ZERO   → symmetric contact, probe fully inside one zone.
  Mz ≠ ZERO → probe straddles two zones of different stiffness.
"""
import argparse
import sys

import numpy as np
import matplotlib

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec

from sim.configs import TissueConfig, SimulationConfig
from sim.tissue import build_tissue
from sim.shapes import create_circular_probe
from sim.trajectory import TwoPointTrajectory
from sim.simulation import SoftObjectSimulation


# ──────────────────────────────────────────────────────────────────────────────
# Fixed physical constants
# ──────────────────────────────────────────────────────────────────────────────

PROBE_RADIUS    = 0.05
PENETRATION     = 0.07   # how far probe centre goes below tissue surface top
K_COLLISION     = 0.12
STEPS_PER_FRAME = 300
HANDLE_OFFSET   = 0.28   # handle centre is this far above probe centre
HANDLE_W        = 0.10
HANDLE_H        = 0.045


# ──────────────────────────────────────────────────────────────────────────────
# Simulation
# ──────────────────────────────────────────────────────────────────────────────

def _make_sim(tissue, probe_center_x, y_start, y_end, n_frames):
    traj = TwoPointTrajectory(
        T=0.1 * n_frames,
        x0=probe_center_x, y0=y_start,
        x1=probe_center_x, y1=y_end,
    )
    probe = create_circular_probe(
        radius=PROBE_RADIUS, num_points=8,
        x_center=probe_center_x, y_center=y_start,
        trajectory=traj,
    )
    scfg = SimulationConfig(
        steps=STEPS_PER_FRAME,
        dt=0.1,
        warmup=True,
        collision_spring_constant=K_COLLISION,
        frames=n_frames,
        save_vectors=False,
        save_images=False,
    )
    return SoftObjectSimulation([tissue, probe], scfg)


def run_poke(x_pos: float, seed: int = 0, n_down: int = 8, n_up: int = 8, n_zones: int = 5):
    """
    Run a full poke (down + up) at x_pos.
    Returns list of per-frame dicts, the tissue object, zone labels, and config.
    """
    np.random.seed(seed)          # seeds global RNG used by sensor noise
    rng  = np.random.default_rng(seed)
    tcfg = TissueConfig(n_zones=n_zones)
    tissue, zone_label_list = build_tissue(tcfg, rng=rng)
    zone_label = np.array(zone_label_list, dtype=np.float32)

    y_hover  = tcfg.height + PROBE_RADIUS + 0.02
    y_bottom = tcfg.height - PENETRATION
    total    = n_down + n_up

    sim_down = _make_sim(tissue, x_pos, y_hover, y_bottom, n_down)
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

    sim_up = _make_sim(tissue, x_pos, y_bottom, y_hover, n_up)
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


def _draw_probe(ax, frame_data, in_contact: bool):
    """Draw probe tip circle, shaft and handle with sensor marker."""
    pv  = frame_data["probe_verts"]
    pc  = pv.mean(axis=0)
    ang = np.linspace(0, 2 * np.pi, 64)

    tip_col = "#e74c3c" if in_contact else "#95a5a6"
    ax.fill(pc[0] + PROBE_RADIUS * np.cos(ang),
            pc[1] + PROBE_RADIUS * np.sin(ang),
            fc=tip_col, ec="k", lw=0.8, alpha=0.75, zorder=5)

    handle_cx = pc[0]
    handle_cy = pc[1] + HANDLE_OFFSET
    ax.plot([pc[0], handle_cx],
            [pc[1] + PROBE_RADIUS, handle_cy - HANDLE_H / 2],
            color="#555", lw=2.5, zorder=4)
    ax.add_patch(mpatches.Rectangle(
        (handle_cx - HANDLE_W / 2, handle_cy - HANDLE_H / 2),
        HANDLE_W, HANDLE_H,
        fc="#bdc3c7", ec="#7f8c8d", lw=1, zorder=4,
    ))
    ax.plot(handle_cx, handle_cy, "D", color="#2980b9", markersize=9, zorder=6)
    ax.text(handle_cx + 0.06, handle_cy,
            "sensor", fontsize=7, color="#2980b9", va="center", zorder=7)

    return handle_cx, handle_cy


def _draw_force_arrow(ax, handle_cx, handle_cy, fy: float, max_fy: float):
    """Draw upward arrow at sensor proportional to Fy."""
    if fy < 1e-9 or max_fy < 1e-9:
        return
    arrow_len = 0.18 * fy / max_fy
    ax.annotate(
        "",
        xy=(handle_cx, handle_cy + 0.02 + arrow_len),
        xytext=(handle_cx, handle_cy + 0.02),
        arrowprops=dict(arrowstyle="-|>", color="#27ae60", lw=2.0),
        zorder=7,
    )
    ax.text(handle_cx + 0.04, handle_cy + 0.02 + arrow_len * 0.55,
            f"Fy={fy:.4f}", fontsize=7.5, color="#27ae60", va="center", zorder=7)


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
    parser.add_argument("--save",   action="store_true",
                        help="Save animation as GIF to output/sim/")
    args = parser.parse_args()

    x_pos = float(np.clip(args.x_pos, 0.05, 1.95))
    n_down = args.frames // 2
    n_up   = args.frames - n_down
    total_frames = args.frames

    print(f"\nRunning poke at x={x_pos:.2f}  seed={args.seed}  frames={total_frames}")
    print("Sensor: Fy = 2k × penetration (Newton's 3rd law)")

    frames, tissue, zone_label, tcfg = run_poke(
        x_pos, seed=args.seed, n_down=n_down, n_up=n_up, n_zones=args.n_zones,
    )

    fy_series = np.array([f["sensor"][1] for f in frames])
    mz_series = np.array([f["sensor"][2] for f in frames])
    t_axis    = np.arange(total_frames)
    max_fy    = max(abs(fy_series).max(), 1e-9)

    n_zones    = len(zone_label)
    zone_width = tcfg.width / n_zones
    probe_zone = min(int(x_pos / zone_width), n_zones - 1)

    orig_verts   = tissue.original_positions.numpy()
    x_min, y_min = orig_verts.min(axis=0)
    x_max, y_max = orig_verts.max(axis=0)

    # Absolute E scale — same seed always maps to same color
    e_min_scale = tcfg.young_modulus_min
    e_max_scale = tcfg.young_modulus_max

    zone_boundaries = [zone_width * (i + 1) for i in range(n_zones - 1)]
    near_boundary   = any(abs(x_pos - b) < PROBE_RADIUS * 2 for b in zone_boundaries)

    print(f"Zone E values: {[f'{e:.4f}' for e in zone_label]}")
    print(f"Probe zone {probe_zone}  E={zone_label[probe_zone]:.4f}"
          f"  {'← near zone boundary' if near_boundary else ''}")
    print(f"Max Fy: {max_fy:.5f}  |  Max |Mz|: {abs(mz_series).max():.6f}")
    print("\nShowing animation. Close the window to exit.\n")

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 8))
    fig.patch.set_facecolor("#1a1a2e")
    gs = GridSpec(2, 2, figure=fig,
                  width_ratios=[2.2, 1.0],
                  height_ratios=[2.8, 1.0],
                  hspace=0.42, wspace=0.38)

    ax_scene = fig.add_subplot(gs[0, 0])
    ax_zones = fig.add_subplot(gs[1, 0])
    ax_fy    = fig.add_subplot(gs[0, 1])
    ax_mz    = fig.add_subplot(gs[1, 1])

    for ax in (ax_scene, ax_zones, ax_fy, ax_mz):
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="#ccc")
        ax.xaxis.label.set_color("#ccc")
        ax.yaxis.label.set_color("#ccc")
        ax.title.set_color("#eee")
        for spine in ax.spines.values():
            spine.set_edgecolor("#555")

    # ── Fy graph ──────────────────────────────────────────────────────────────
    ax_fy.axhline(0, color="#555", lw=0.8, ls="--")
    ax_fy.axvspan(0,      n_down,        alpha=0.12, color="#e74c3c")
    ax_fy.axvspan(n_down, total_frames,  alpha=0.12, color="#3498db")
    ax_fy.plot(t_axis, fy_series, color="#27ae60", lw=2.2, zorder=3)
    ax_fy.set_xlabel("Frame")
    ax_fy.set_ylabel("Fy  [sim N]")
    ax_fy.set_title("Reaction force at sensor — Newton's 3rd law", color="#eee", fontsize=8.5)
    ax_fy.set_xlim(-0.5, total_frames - 0.5)

    ax_fy.text(n_down / 2, ax_fy.get_ylim()[1] * 0.06,
               "▼ pressing", fontsize=7.5, color="#e74c3c", ha="center")
    ax_fy.text(n_down + n_up / 2, ax_fy.get_ylim()[1] * 0.06,
               "▲ retracting", fontsize=7.5, color="#3498db", ha="center")

    contact_frames = np.where([f["contact_pt"] is not None for f in frames])[0]
    if len(contact_frames):
        cf0 = contact_frames[0]
        ax_fy.axvline(cf0, color="#e74c3c", lw=1.2, ls=":", alpha=0.8)
        ax_fy.text(cf0 + 0.15, max_fy * 0.88,
                   "contact\nstarts", fontsize=7, color="#e74c3c", va="top")

    peak_f = int(np.argmax(fy_series))
    ax_fy.annotate(
        f"peak {fy_series[peak_f]:.4f}",
        xy=(peak_f, fy_series[peak_f]),
        xytext=(peak_f - 2.0, fy_series[peak_f] * 1.15),
        fontsize=7.5, color="#f1c40f",
        arrowprops=dict(arrowstyle="->", color="#f1c40f", lw=1),
    )

    ax_fy.text(0.02, 0.97,
               f"Larger Fy → stiffer tissue\nE={zone_label[probe_zone]:.4f}",
               fontsize=7, color="#aaa", transform=ax_fy.transAxes, va="top")

    fy_vline = ax_fy.axvline(0, color="#fff", lw=1.5, alpha=0.7, zorder=4)

    # ── Mz graph ──────────────────────────────────────────────────────────────
    ax_mz.axhline(0, color="#888", lw=1.2, ls="--")
    ax_mz.axvspan(0,      n_down,       alpha=0.12, color="#e74c3c")
    ax_mz.axvspan(n_down, total_frames, alpha=0.12, color="#3498db")
    ax_mz.plot(t_axis, mz_series, color="#9b59b6", lw=2.2, zorder=3)
    ax_mz.set_xlabel("Frame")
    ax_mz.set_ylabel("Mz  [sim N·m]")
    ax_mz.set_title("Torque at sensor — non-zero at zone boundaries", color="#eee", fontsize=8.5)
    ax_mz.set_xlim(-0.5, total_frames - 0.5)
    mz_range = max(abs(mz_series).max() * 1.5, 1e-9)
    ax_mz.set_ylim(-mz_range, mz_range)

    boundary_msg = ("⚠ near zone boundary\nMz ≠ 0 expected"
                    if near_boundary else
                    "inside one zone\nMz ≈ 0 expected")
    ax_mz.text(0.98, 0.97, boundary_msg, fontsize=7.5, color="#ccc",
               transform=ax_mz.transAxes, ha="right", va="top",
               bbox=dict(fc="#1a1a2e", ec="#555", pad=3))

    mz_vline = ax_mz.axvline(0, color="#fff", lw=1.5, alpha=0.7, zorder=4)

    # ── Stiffness zone bar chart ───────────────────────────────────────────────
    cmap_z = matplotlib.colormaps["viridis"]
    E_n_z  = (zone_label - e_min_scale) / (e_max_scale - e_min_scale + 1e-12)
    bar_colors = [cmap_z(float(np.clip(e, 0, 1))) for e in E_n_z]

    bars = ax_zones.bar(np.arange(n_zones), zone_label, color=bar_colors,
                        edgecolor="#555", linewidth=0.8)
    bars[probe_zone].set_edgecolor("#ffffff")
    bars[probe_zone].set_linewidth(2.5)
    ax_zones.set_xlabel("Zone index")
    ax_zones.set_ylabel("Young's modulus E")
    ax_zones.set_title(f"Tissue stiffness profile — probe in zone {probe_zone}",
                       color="#eee", fontsize=8.5)
    ax_zones.set_xticks(np.arange(n_zones))
    ax_zones.tick_params(colors="#ccc")
    for i, e in enumerate(zone_label):
        ax_zones.text(i, e + zone_label.max() * 0.03, f"{e:.3f}",
                      ha="center", va="bottom", fontsize=7, color="#ccc")

    # ── Animation update ──────────────────────────────────────────────────────

    def update(fi: int):
        ax_scene.cla()

        ax_scene.set_facecolor("#233977")
        ax_scene.tick_params(colors="#ccc")
        ax_scene.xaxis.label.set_color("#ccc")
        ax_scene.yaxis.label.set_color("#ccc")
        for spine in ax_scene.spines.values():
            spine.set_edgecolor("#555")
        ax_scene.set_xlim(x_min - 0.15, x_max + 0.15)
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
        ax_scene.text(x_min + 0.05, y_min - 0.035,
                      "fixed boundary", fontsize=6.5, color="#e74c3c", alpha=0.8)

        for b in zone_boundaries:
            ax_scene.axvline(b, color="#aaa", lw=0.7, ls=":", alpha=0.55, zorder=2)

        _draw_tissue(ax_scene, tissue, frames[fi], e_min_scale, e_max_scale)

        in_contact = frames[fi]["contact_pt"] is not None
        handle_cx, handle_cy = _draw_probe(ax_scene, frames[fi], in_contact)

        cp = frames[fi]["contact_pt"]
        if cp is not None:
            ax_scene.plot(cp[0], cp[1], "o", color="#e74c3c", markersize=10, zorder=8)
            ax_scene.text(cp[0] + 0.05, cp[1] + 0.025,
                          "contact", fontsize=7.5, color="#e74c3c", zorder=9)

        fy_now = float(frames[fi]["sensor"][1])
        mz_now = float(frames[fi]["sensor"][2])
        _draw_force_arrow(ax_scene, handle_cx, handle_cy, fy_now, max_fy)

        ax_scene.text(
            x_min + 0.02, y_min - 0.045,
            f"Fy = {fy_now:.5f} N   |   Mz = {mz_now:.5f} N·m",
            fontsize=8.5, color="#27ae60", zorder=9,
        )

        fy_vline.set_xdata([fi, fi])
        mz_vline.set_xdata([fi, fi])
        return []

    anim = FuncAnimation(
        fig, update, frames=total_frames, interval=380, repeat=True,
    )

    plt.tight_layout(pad=1.4)

    if args.save:
        import os
        out_dir = "output/sim"
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"poke_x{x_pos:.2f}_seed{args.seed}.gif")
        print(f"Saving GIF → {out_path} ...")
        anim.save(out_path, writer="pillow", fps=3)
        print("Saved.")

    plt.show()


if __name__ == "__main__":
    main()
