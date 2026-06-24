"""
Parallel dataset generation.

Each experiment:
  1. Sample a random stiffness profile (n_zones Young's moduli).
  2. Build the rectangle tissue phantom.
  3. Scan the probe across n_scan_positions evenly spaced x-coordinates.
  4. At each position: run a poke-and-return trajectory, record sensor readings.
  5. Save sensor_log.pkl + label.npy + config.yaml to disk.

Usage:
    python -m sim.generate_dataset \
        --data_folder ./data/dataset \
        --num_experiments 1000 \
        --num_processes 8
"""
import argparse
import contextlib
import io
import math
import multiprocessing
import os
import pickle

import numpy as np
import pyrallis
import torch

from data.configs import DatasetConfig
from sim.configs import ExperimentConfig, SimulationConfig
from sim.tissue import build_tissue, scan_probe_positions
from sim.shapes import create_circular_probe
from sim.simulation import SoftObjectSimulation, BadAngleError
from sim.trajectory import TwoPointTrajectory, ThreePointTrajectory, poke_trajectory_points

MAX_ANGLE_RESAMPLE_ATTEMPTS = 10


def run_single_experiment(
    exp_idx: int,
    data_folder: str,
    dataset_config: DatasetConfig,
    exp_config: ExperimentConfig,
    rng: np.random.Generator,
) -> str:
    """
    Run one experiment and save to disk.
    Returns a log string (stdout capture).
    """
    exp_folder = os.path.join(data_folder, f"experiment_{exp_idx}")

    # Skip if already done
    done_marker = os.path.join(exp_folder, "done.txt")
    if os.path.exists(done_marker):
        return f"Experiment {exp_idx}: skipped (already exists).\n"

    os.makedirs(exp_folder, exist_ok=True)
    log = []

    # ------------------------------------------------------------------
    # 1. Build tissue
    # ------------------------------------------------------------------
    tissue, zone_moduli = build_tissue(exp_config.tissue, rng=rng)
    np.save(os.path.join(exp_folder, "label.npy"), np.array(zone_moduli, dtype=np.float32))

    # ------------------------------------------------------------------
    # 2. Scan probe positions
    # ------------------------------------------------------------------
    x_positions = scan_probe_positions(
        exp_config.tissue,
        n_positions=dataset_config.num_scan_positions,
    )

    tissue_top_y = exp_config.tissue.height
    hover_height = exp_config.probe.hover_height
    # poke_trajectory_points() works in terms of the probe CENTRE's travel,
    # so convert the configured TIP penetration depth by subtracting radius.
    center_penetration_depth = exp_config.probe.tip_penetration_depth - exp_config.probe.radius

    all_poke_readings: list[np.ndarray] = []  # list of (frames_per_poke, 3) arrays

    for pos_idx, x_pos in enumerate(x_positions):
        for attempt in range(MAX_ANGLE_RESAMPLE_ATTEMPTS):
            # Reset tissue to rest state before each (re)attempted poke
            tissue.reset_positions()

            if not math.isnan(exp_config.probe.probe_angle_deg):
                theta_deg = exp_config.probe.probe_angle_deg
            else:
                theta_deg = float(rng.uniform(
                    exp_config.probe.angle_min_deg, exp_config.probe.angle_max_deg
                ))

            # Build poke trajectory (down + up), tilted by theta_deg from vertical
            pts = poke_trajectory_points(
                x_pos, tissue_top_y, hover_height, center_penetration_depth, angle_deg=theta_deg
            )
            T = dataset_config.frames_per_poke * exp_config.simulation.dt

            # Go down
            down_traj = TwoPointTrajectory(T=T / 2, x0=pts[0][0], y0=pts[0][1], x1=pts[1][0], y1=pts[1][1])
            # Go back up
            up_traj = TwoPointTrajectory(T=T / 2, x0=pts[1][0], y0=pts[1][1], x1=pts[0][0], y1=pts[0][1])

            poke_readings: list[np.ndarray] = []
            any_contact = False

            for (traj, start_pt), n_frames in [
                ((down_traj, pts[0]), dataset_config.frames_per_poke // 2),
                ((up_traj, pts[1]), dataset_config.frames_per_poke // 2),
            ]:
                probe = create_circular_probe(
                    radius=exp_config.probe.radius,
                    num_points=exp_config.probe.num_points,
                    x_center=start_pt[0],
                    y_center=start_pt[1],
                    trajectory=traj,
                )

                sim_config = SimulationConfig(
                    collision_spring_constant=exp_config.simulation.collision_spring_constant,
                    steps=exp_config.simulation.steps,
                    dt=exp_config.simulation.dt,
                    learning_rate=exp_config.simulation.learning_rate,
                    adam_beta_1=exp_config.simulation.adam_beta_1,
                    adam_beta_2=exp_config.simulation.adam_beta_2,
                    warmup=exp_config.simulation.warmup,
                    probe_force_noise_std=exp_config.simulation.probe_force_noise_std,
                    probe_angle_deg=theta_deg,
                    frames=n_frames,
                    save_vectors=True,
                    save_images=False,
                    save_video=False,
                )

                sim = SoftObjectSimulation([tissue, probe], sim_config)
                for _ in range(n_frames):
                    sim.step()
                    if sim.get_contact_point() is not None:
                        any_contact = True
                    poke_readings.append(sim.get_aggregated_sensor())

            if any_contact:
                break

            # Bad angle: the probe never touched the tissue during this poke.
            if not math.isnan(exp_config.probe.probe_angle_deg):
                raise BadAngleError(
                    f"Configured probe_angle_deg={theta_deg:.1f} never makes contact "
                    f"at poke x_pos={x_pos:.3f} (experiment {exp_idx}, pos {pos_idx})."
                )
            log.append(
                f"  pos {pos_idx}: x={x_pos:.3f} angle={theta_deg:.1f} -> bad angle "
                f"(no contact), resampling (attempt {attempt + 1}/{MAX_ANGLE_RESAMPLE_ATTEMPTS})"
            )
        else:
            raise BadAngleError(
                f"Could not find a contact-making probe angle for poke "
                f"x_pos={x_pos:.3f} (experiment {exp_idx}, pos {pos_idx}) after "
                f"{MAX_ANGLE_RESAMPLE_ATTEMPTS} attempts."
            )

        all_poke_readings.append(np.stack(poke_readings, axis=0))  # (frames_per_poke, 3)
        log.append(
            f"  pos {pos_idx}: x={x_pos:.3f} angle={theta_deg:.1f}  "
            f"max_force={np.abs(poke_readings).max():.4f}"
        )

    # all_poke_readings: list of length n_scan_positions, each (frames_per_poke, 3)
    sensor_array = np.stack(all_poke_readings, axis=0)  # (n_positions, frames_per_poke, 3)
    with open(os.path.join(exp_folder, "sensor_log.pkl"), "wb") as f:
        pickle.dump(sensor_array, f)

    # Save config
    with open(os.path.join(exp_folder, "config.yaml"), "w") as f:
        pyrallis.dump(exp_config, f)

    np.save(os.path.join(exp_folder, "x_positions.npy"), np.array(x_positions, dtype=np.float32))

    with open(done_marker, "w") as f:
        f.write("done")

    return "\n".join([f"Experiment {exp_idx}:"] + log) + "\n"


def _worker(args):
    """Multiprocessing worker wrapper."""
    exp_idx, data_folder, dataset_config, exp_config_path, start_index = args
    rng = np.random.default_rng(start_index + exp_idx)
    torch.manual_seed(start_index + exp_idx)
    np.random.seed(start_index + exp_idx)

    exp_config = pyrallis.parse(config_class=ExperimentConfig, config_path=exp_config_path, args=[])

    try:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            result = run_single_experiment(exp_idx, data_folder, dataset_config, exp_config, rng)
        return result + out.getvalue()
    except Exception as e:
        return f"Experiment {exp_idx} FAILED: {e}\n"


def generate_dataset(dataset_config: DatasetConfig):
    os.makedirs(dataset_config.data_folder, exist_ok=True)

    inputs = [
        (
            dataset_config.start_index + i,
            dataset_config.data_folder,
            dataset_config,
            dataset_config.experiment_config_path,
            dataset_config.start_index,
        )
        for i in range(dataset_config.num_experiments)
    ]

    if dataset_config.num_processes == 1:
        outputs = [_worker(args) for args in inputs]
    else:
        with multiprocessing.Pool(processes=dataset_config.num_processes) as pool:
            outputs = pool.map(_worker, inputs)

    # Summary
    failed = [o for o in outputs if "FAILED" in o]
    print(f"Done. {len(outputs) - len(failed)}/{len(outputs)} experiments succeeded.")
    if failed:
        print("FAILED:")
        for f in failed:
            print(f)

    failed_path = os.path.join(dataset_config.data_folder, "failed_experiments.txt")
    with open(failed_path, "w") as f:
        f.write("\n".join(failed))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", default="./data/dataset")
    parser.add_argument("--num_experiments", type=int, default=500)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--num_processes", type=int, default=4)
    parser.add_argument("--num_scan_positions", type=int, default=10)
    parser.add_argument("--frames_per_poke", type=int, default=10)
    parser.add_argument("--experiment_config_path", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = DatasetConfig(
        data_folder=args.data_folder,
        num_experiments=args.num_experiments,
        start_index=args.start_index,
        num_processes=args.num_processes,
        num_scan_positions=args.num_scan_positions,
        frames_per_poke=args.frames_per_poke,
        experiment_config_path=args.experiment_config_path,
    )
    generate_dataset(cfg)
