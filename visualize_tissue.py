"""
visualize_tissue.py — Stiffness-field viewer (step / sigmoid tissue).

The tissue's Young's modulus follows a sigmoid profile along x:

    E(x) = E_left + (E_right - E_left) * sigmoid(k * (x - x0))
    sigmoid(d) = 1 / (1 + e^(-d))

discretized to one E value per mesh column (n_columns thin rectangles,
each a perfect rectangle of whole mesh cells).

Tissue modes:
  step     -> two rectangles with a hard change at x0 (the k = inf limit)
  sigmoid  -> smooth gradient controlled by k (small k = gentle, large k = sharp)

Actions:
  view     -> one static figure: the E-field + its E(x) profile (no simulation)
  poke     -> run one probe poke on the tissue and animate it

Ground-truth label per experiment: (E_left, E_right, x0, k), k = inf for step.

Usage:
    python visualize_tissue.py step view
    python visualize_tissue.py step poke --x_pos 1.0 --seed 2
    python visualize_tissue.py sigmoid view --k 15
    python visualize_tissue.py sigmoid view --seed 3 --save
    python visualize_tissue.py sigmoid poke --k 50 --x0 1.0 --x_pos 1.0
"""
import argparse
import dataclasses
import math
import os
import sys

import numpy as np
import pyrallis
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec

from sim.configs import ExperimentConfig, TissueConfig
from sim.tissue import (
    SigmoidFieldParams,
    build_tissue_sigmoid,
    grid_counts,
    sample_sigmoid_params,
    sample_step_params,
    sigmoid_column_stiffness,
    snap_to_grid,
)
from sim.trajectory import poke_trajectory_points
from visualize_sim import (
    _make_sim,
    _draw_probe,
    _draw_force_arrow,
    HANDLE_OFFSET,
    HANDLE_H,
    FX_YLIM,
    FY_YLIM,
    MZ_YLIM,
)

FIG_BG    = "#1a1a2e"
PANEL_BG  = "#16213e"
SCENE_BG  = "#233977"
CMAP      = "viridis"
DEFAULT_N_COLUMNS = 128    # fallback when neither --n_columns nor the config sets it


# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────────────

def _k_str(k: float) -> str:
    return "inf (step)" if math.isinf(k) else f"{k:g}"


def _style_axis(ax, facecolor=PANEL_BG):
    ax.set_facecolor(facecolor)
    ax.tick_params(colors="#ccc")
    ax.xaxis.label.set_color("#ccc")
    ax.yaxis.label.set_color("#ccc")
    ax.title.set_color("#eee")
    for spine in ax.spines.values():
        spine.set_edgecolor("#555")


def _field_norm(column_E: np.ndarray) -> Normalize:
    """Colour scale spanning THIS tissue's E range (not the global config
    range), so the full colormap is always used and the field never looks
    washed out. Degenerate (flat) fields get a small symmetric pad."""
    e_lo, e_hi = float(column_E.min()), float(column_E.max())
    if e_hi - e_lo < 1e-9:
        pad = max(abs(e_lo) * 0.1, 1e-6)
        return Normalize(e_lo - pad, e_hi + pad)
    return Normalize(e_lo, e_hi)


def _draw_tissue_fast(ax, tissue, verts, norm: Normalize):
    """
    Draw all tissue triangles coloured by Young's modulus in ONE
    PolyCollection. With 128 columns the mesh has ~6.6k triangles — a
    per-triangle ax.fill() loop (as in visualize_sim) is far too slow here.
    """
    E = tissue.young_modulus.numpy()
    polys = verts[tissue.triangles]  # (T, 3, 2)
    coll = PolyCollection(
        polys, array=E, cmap=CMAP, norm=norm,
        edgecolors="none", antialiaseds=False, zorder=2,
    )
    ax.add_collection(coll)
    return coll


def _add_colorbar(fig, axes, norm: Normalize):
    sm = ScalarMappable(norm=norm, cmap=CMAP)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, pad=0.015, fraction=0.035)
    cbar.set_label("Young's modulus E", color="#ccc", fontsize=8)
    cbar.ax.tick_params(colors="#ccc", labelsize=7)
    cbar.outline.set_edgecolor("#555")
    return cbar


def _plot_profile(ax, tcfg: TissueConfig, params: SigmoidFieldParams,
                  column_E: np.ndarray, title: str | None = None):
    """E(x) curve: the continuous sigmoid + the per-column staircase actually
    assigned to the mesh, with the transition centre x0 marked."""
    nx, _ = grid_counts(tcfg)
    xs = np.linspace(0.0, tcfg.width, 400)
    bounds = np.linspace(0.0, tcfg.width, nx + 1)

    e_lo, e_hi = float(column_E.min()), float(column_E.max())
    pad = max((e_hi - e_lo) * 0.15, e_hi * 0.05, 1e-6)

    ax.grid(True, color="#3a3a55", lw=0.6, alpha=0.5, zorder=0)
    ax.stairs(column_E, bounds, color="#f0932b", lw=1.3, alpha=0.9, zorder=3,
              label=f"per-column E ({nx} rects)")
    ax.plot(xs, params.young_modulus(xs), color="#2ecc71", lw=1.8, zorder=4,
            label="E(x)")
    ax.axvline(params.x0, color="#e74c3c", lw=1.0, ls="--", alpha=0.8, zorder=2)

    ax.set_xlim(0, tcfg.width)
    ax.set_ylim(e_lo - pad, e_hi + pad)
    ax.text(params.x0, e_hi + pad, " x0", fontsize=7, color="#e74c3c",
            ha="left", va="top")
    ax.set_xlabel("x")
    ax.set_ylabel("E")
    if title:
        ax.set_title(title, color="#eee", fontsize=9, fontweight="bold")


def _params_label(params: SigmoidFieldParams) -> str:
    return (f"E_left = {params.e_left:.4f}    E_right = {params.e_right:.4f}    "
            f"x0 = {params.x0:.3f}    k = {_k_str(params.k)}")


def _file_tag(params: SigmoidFieldParams, nx: int) -> str:
    k_tag = "inf" if math.isinf(params.k) else f"{params.k:g}"
    return (f"eL{params.e_left:.4f}_eR{params.e_right:.4f}"
            f"_x{params.x0:.3f}_k{k_tag}_n{nx}")


def resolve_params(args, tcfg: TissueConfig, tissue_mode: str) -> SigmoidFieldParams:
    """
    Sample the field parameters (E_left, E_right, x0[, k]) from args.seed
    (main() fills in a fresh random seed when --seed is omitted), then apply
    explicit CLI overrides. step mode forces k = inf.
    """
    nx, _ = grid_counts(tcfg)
    rng = np.random.default_rng(args.seed)
    sample = sample_step_params if tissue_mode == "step" else sample_sigmoid_params
    params = sample(tcfg, rng)

    overrides = {}
    if args.k is not None:
        if tissue_mode == "step":
            print("Note: --k is ignored in step mode (k = inf).")
        else:
            overrides["k"] = args.k
    if args.x0 is not None:
        overrides["x0"] = snap_to_grid(args.x0, tcfg.width, nx)
    if args.e_left is not None:
        overrides["e_left"] = args.e_left
    if args.e_right is not None:
        overrides["e_right"] = args.e_right
    params = dataclasses.replace(params, **overrides)

    if tissue_mode == "step":
        params = dataclasses.replace(params, k=math.inf)
    return params


# ──────────────────────────────────────────────────────────────────────────────
# view action: one static figure (field + profile)
# ──────────────────────────────────────────────────────────────────────────────

def run_view(args, tcfg: TissueConfig, tissue_mode: str):
    params = resolve_params(args, tcfg, tissue_mode)
    nx, ny = grid_counts(tcfg)
    print(f"Tissue [{tissue_mode}]: {_params_label(params)}")
    print(f"Mesh: {nx}x{ny} cells")

    print("Building tissue...")
    tissue, params, column_E = build_tissue_sigmoid(tcfg, params=params)
    rest_verts = tissue.original_positions.numpy()
    norm = _field_norm(column_E)

    fig = plt.figure(figsize=(12, 6))
    fig.patch.set_facecolor(FIG_BG)
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1.15, 1.0], hspace=0.45)
    fig.suptitle(
        f"{tissue_mode} tissue — {nx} rectangle columns\n{_params_label(params)}",
        color="#eee", fontsize=11,
    )

    ax_field = fig.add_subplot(gs[0])
    _style_axis(ax_field, SCENE_BG)
    _draw_tissue_fast(ax_field, tissue, rest_verts, norm)
    ax_field.axvline(params.x0, color="#e74c3c", lw=1.1, ls="--", alpha=0.9, zorder=3)
    ax_field.set_xlim(-0.02, tcfg.width + 0.02)
    ax_field.set_ylim(-0.02, tcfg.height + 0.02)
    ax_field.set_aspect("equal")
    if math.isinf(params.k):
        field_title = f"hard step at x0 = {params.x0:.3f}"
    else:
        field_title = (f"transition width ≈ 4/k = {4.0 / params.k:.3f} "
                       f"(tissue width {tcfg.width:g})")
    ax_field.set_title(field_title, color="#eee", fontsize=9, fontweight="bold")

    ax_prof = fig.add_subplot(gs[1])
    _style_axis(ax_prof)
    _plot_profile(ax_prof, tcfg, params, column_E)
    ax_prof.legend(fontsize=7, facecolor=PANEL_BG, edgecolor="#555",
                   labelcolor="#ccc", loc="best")

    _add_colorbar(fig, [ax_field], norm)

    if args.save:
        out_dir = "output/tissue"
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(
            out_dir, f"view_{tissue_mode}_{_file_tag(params, nx)}.png"
        )
        fig.savefig(out_path, dpi=150, facecolor=FIG_BG)
        print(f"Saved → {out_path}")

    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# poke action
# ──────────────────────────────────────────────────────────────────────────────

def _run_poke_on_tissue(tissue, tcfg, x_pos, n_down, n_up,
                        probe_radius, hover_height, tip_penetration_depth,
                        angle_deg, sim_cfg):
    """Run a full poke (down + up) on a prebuilt tissue. Mirrors
    visualize_sim.run_poke but takes the tissue instead of building one."""
    center_penetration_depth = tip_penetration_depth - probe_radius
    pts = poke_trajectory_points(
        x_pos, tcfg.height, hover_height, center_penetration_depth, angle_deg=angle_deg
    )
    (x_hover, y_hover), (x_bottom, y_bottom) = pts
    total = n_down + n_up

    frames = []
    for phase, n, (xa, ya, xb, yb) in (
        ("down", n_down, (x_hover, y_hover, x_bottom, y_bottom)),
        ("up",   n_up,   (x_bottom, y_bottom, x_hover, y_hover)),
    ):
        sim = _make_sim(tissue, xa, ya, xb, yb, n, probe_radius, angle_deg, sim_cfg)
        for i in range(n):
            done = len(frames) + 1
            label = "pressing down" if phase == "down" else "retracting   "
            sys.stdout.write(f"\r  frame {done}/{total} ({label})")
            sys.stdout.flush()
            tv, pv = sim.step()
            frames.append({
                "tissue_verts": tv.copy(),
                "probe_verts":  pv.copy(),
                "sensor":       sim.get_aggregated_sensor().copy(),
                "contact_pt":   sim.get_contact_point(),
                "phase":        phase,
            })
    print()
    return frames


def run_poke(args, exp_cfg: ExperimentConfig, tcfg: TissueConfig, tissue_mode: str):
    probe_radius          = exp_cfg.probe.radius
    hover_height          = exp_cfg.probe.hover_height
    tip_penetration_depth = exp_cfg.probe.tip_penetration_depth

    seed = args.seed
    x_pos = float(np.clip(args.x_pos, 0.05, tcfg.width - 0.05))
    n_down = args.frames // 2
    n_up   = args.frames - n_down
    total_frames = args.frames

    # Poke angle: CLI override > config fixed value > sampled (as in visualize_sim)
    if args.angle_deg is not None:
        angle_deg = args.angle_deg
    elif not math.isnan(exp_cfg.probe.probe_angle_deg):
        angle_deg = exp_cfg.probe.probe_angle_deg
    else:
        angle_deg = float(np.random.default_rng(seed).uniform(
            exp_cfg.probe.angle_min_deg, exp_cfg.probe.angle_max_deg
        ))

    np.random.seed(seed)  # sensor-noise RNG
    params = resolve_params(args, tcfg, tissue_mode)
    nx, ny = grid_counts(tcfg)
    print(f"\nTissue [{tissue_mode}]: {_params_label(params)}")
    print(f"Mesh: {nx}x{ny} cells. Poke at x={x_pos:.2f}, angle={angle_deg:.1f}°, "
          f"{total_frames} frames.")

    print("Building tissue...")
    tissue, params, column_E = build_tissue_sigmoid(tcfg, params=params)
    norm = _field_norm(column_E)

    frames = _run_poke_on_tissue(
        tissue, tcfg, x_pos, n_down, n_up,
        probe_radius, hover_height, tip_penetration_depth,
        angle_deg, exp_cfg.simulation,
    )

    fx_series = np.array([f["sensor"][0] for f in frames])
    fy_series = np.array([f["sensor"][1] for f in frames])
    mz_series = np.array([f["sensor"][2] for f in frames])
    t_axis    = np.arange(total_frames)
    max_force = max(abs(fx_series).max(), abs(fy_series).max(), 1e-9)

    center_penetration_depth = tip_penetration_depth - probe_radius
    _, (x_touch, _) = poke_trajectory_points(
        x_pos, tcfg.height, hover_height, center_penetration_depth, angle_deg=angle_deg
    )
    e_at_touch = float(params.young_modulus(x_touch))
    print(f"Probe touches at x={x_touch:.3f} → local E={e_at_touch:.4f}")
    print(f"Max Fx: {abs(fx_series).max():.5f}  |  Max Fy: {abs(fy_series).max():.5f}  "
          f"|  Max |Mz|: {abs(mz_series).max():.6f}")

    orig_verts   = tissue.original_positions.numpy()
    x_min, y_min = orig_verts.min(axis=0)
    x_max, y_max = orig_verts.max(axis=0)
    all_probe_x = np.concatenate([f["probe_verts"][:, 0] for f in frames])
    scene_x_min = min(x_min, all_probe_x.min()) - 0.08
    scene_x_max = max(x_max, all_probe_x.max()) + 0.08

    # ── Figure layout (like visualize_sim, but the zone bar chart is replaced
    #    by the E(x) profile) ──────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor(FIG_BG)
    gs = GridSpec(3, 2, figure=fig,
                  width_ratios=[2.2, 1.0],
                  height_ratios=[2.0, 1.0, 1.0],
                  hspace=0.6, wspace=0.38)

    ax_scene = fig.add_subplot(gs[:2, 0])
    ax_prof  = fig.add_subplot(gs[2, 0])
    ax_fx    = fig.add_subplot(gs[0, 1])
    ax_fy    = fig.add_subplot(gs[1, 1])
    ax_mz    = fig.add_subplot(gs[2, 1])

    for ax in (ax_scene, ax_prof, ax_fx, ax_fy, ax_mz):
        _style_axis(ax)

    def _setup_series_axis(ax, series, color, ylabel, title, ylim):
        ax.grid(True, color="#3a3a55", lw=0.6, alpha=0.5, zorder=0)
        ax.axhline(0, color="#888", lw=0.8, ls="--", zorder=1)
        ax.axvspan(0,      n_down,       alpha=0.12, color="#e74c3c", zorder=0)
        ax.axvspan(n_down, total_frames, alpha=0.12, color="#3498db", zorder=0)
        ax.plot(t_axis, series, color=color, lw=2.4, zorder=3, solid_capstyle="round")
        ax.fill_between(t_axis, series, 0, color=color, alpha=0.12, zorder=2)
        ax.set_xlabel("Frame")
        ax.set_ylabel(ylabel)
        ax.set_title(title, color="#eee", fontsize=9, fontweight="bold")
        ax.set_xlim(-0.5, total_frames - 0.5)
        ax.set_ylim(*ylim)
        return ax.axvline(0, color="#fff", lw=1.5, alpha=0.7, zorder=4)

    fx_vline = _setup_series_axis(ax_fx, fx_series, "#e67e22", "Fx  [sim N]", "Fx", FX_YLIM)
    fy_vline = _setup_series_axis(ax_fy, fy_series, "#27ae60", "Fy  [sim N]", "Fy", FY_YLIM)
    mz_vline = _setup_series_axis(ax_mz, mz_series, "#9b59b6", "Mz  [sim N·m]", "Mz", MZ_YLIM)

    # ── Stiffness profile panel ───────────────────────────────────────────────
    _plot_profile(ax_prof, tcfg, params, column_E, title=f"E(x):  {_params_label(params)}")
    ax_prof.axvline(x_touch, color="#3498db", lw=1.4, alpha=0.9, zorder=5)
    ax_prof.plot(x_touch, e_at_touch, "o", color="#3498db", ms=6, zorder=6)
    ax_prof.text(x_touch, e_at_touch, "  probe", fontsize=7.5, color="#3498db",
                 va="bottom", zorder=6)
    ax_prof.legend(fontsize=6.5, facecolor=PANEL_BG, edgecolor="#555",
                   labelcolor="#ccc", loc="best")

    # ── Animation update ──────────────────────────────────────────────────────

    def update(fi: int):
        ax_scene.cla()
        _style_axis(ax_scene, SCENE_BG)
        ax_scene.set_xlim(scene_x_min, scene_x_max)
        ax_scene.set_ylim(y_min - 0.06, y_max + HANDLE_OFFSET + HANDLE_H + 0.12)
        ax_scene.set_aspect("equal")
        ax_scene.set_xlabel("x  [sim m]")
        ax_scene.set_ylabel("y  [sim m]")
        phase_str = "pressing ▼" if frames[fi]["phase"] == "down" else "retracting ▲"
        ax_scene.set_title(
            f"Frame {fi + 1}/{total_frames}  ·  x={x_pos:.2f}  ·  "
            f"k={_k_str(params.k)}  ·  {phase_str}",
            color="#eee", fontsize=9,
        )

        ax_scene.axhline(y_min, xmin=0.05, xmax=0.95,
                         color="#e74c3c", lw=3, alpha=0.45, zorder=3)
        ax_scene.text(x_max - 0.35, y_min + 0.01,
                      "fixed boundary", fontsize=6.5, color="#e74c3c", alpha=0.8)

        _draw_tissue_fast(ax_scene, tissue, frames[fi]["tissue_verts"], norm)
        ax_scene.axvline(params.x0, color="#e74c3c", lw=0.9, ls="--",
                         alpha=0.7, zorder=3)

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

    anim = FuncAnimation(fig, update, frames=total_frames, interval=380, repeat=True)
    fig.subplots_adjust(left=0.06, right=0.97, top=0.96, bottom=0.06)

    if args.save:
        out_dir = "output/tissue"
        os.makedirs(out_dir, exist_ok=True)
        nx, _ = grid_counts(tcfg)
        out_path = os.path.join(
            out_dir,
            f"poke_{tissue_mode}_seed{seed}_x{x_pos:.2f}_"
            f"angle{angle_deg:.1f}_{_file_tag(params, nx)}.gif",
        )
        print(f"Saving GIF → {out_path} ...")
        anim.save(out_path, writer="pillow", fps=3)
        print("Saved.")

    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Stiffness-field viewer: choose a tissue mode, then an action.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("tissue", choices=["step", "sigmoid"],
                        help="step: two rectangles with a hard change at x0 (k = inf). "
                             "sigmoid: gradient controlled by k.")
    parser.add_argument("action", choices=["view", "poke"],
                        help="view: static E-field figure (no simulation). "
                             "poke: run + animate one probe poke.")

    parser.add_argument("--config", default="configs/sigmoid.yaml",
                        help="Experiment config (probe geometry + sim physics + tissue)")
    parser.add_argument("--n_columns", type=int, default=None,
                        help="Stiffness rectangles along x (mesh columns). "
                             "Default: the config's tissue.n_columns "
                             f"(or {DEFAULT_N_COLUMNS} if the config leaves it 0).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for sampling E_left/E_right/x0 (+k for sigmoid) and "
                             "the poke angle. Default: a fresh random seed each run "
                             "(printed, so you can reproduce it with --seed).")
    parser.add_argument("--save", action="store_true", default=False,
                        help="Save PNG (view) / GIF (poke) to output/tissue/")

    f = parser.add_argument_group("field parameter overrides")
    f.add_argument("--k",       type=float, default=None,
                   help="Sigmoid steepness (ignored in step mode)")
    f.add_argument("--x0",      type=float, default=None, help="Transition centre")
    f.add_argument("--e_left",  type=float, default=None, help="E on the left side")
    f.add_argument("--e_right", type=float, default=None, help="E on the right side")

    p = parser.add_argument_group("poke action")
    p.add_argument("--x_pos",   type=float, default=1.0,  help="Poke x position")
    p.add_argument("--frames",  type=int,   default=16,
                   help="Total frames, split evenly between pressing and retracting")
    p.add_argument("--angle_deg", type=float, default=None,
                   help="Poke tilt from vertical (deg). Default: config / sampled.")
    args = parser.parse_args()

    # No --seed -> draw a fresh one from OS entropy so every run is different,
    # and print it so a nice-looking run can be reproduced exactly.
    if args.seed is None:
        args.seed = int(np.random.default_rng().integers(0, 2 ** 31))
        print(f"No --seed given → using random seed {args.seed} "
              f"(rerun with --seed {args.seed} to reproduce)")

    with open(args.config) as f:
        exp_cfg = pyrallis.load(ExperimentConfig, f)

    n_columns = args.n_columns if args.n_columns is not None else exp_cfg.tissue.n_columns
    if n_columns <= 0:
        n_columns = DEFAULT_N_COLUMNS
    tcfg = dataclasses.replace(exp_cfg.tissue, n_columns=n_columns)

    if args.action == "view":
        run_view(args, tcfg, args.tissue)
    else:
        run_poke(args, exp_cfg, tcfg, args.tissue)


if __name__ == "__main__":
    main()
