"""
Preprocess raw simulation output into a single HDF5 dataset.

Layout of one experiment group in HDF5:
  /experiment_N/
    sensor_log     (n_positions, frames_per_poke, 3)   float32
    x_positions    (n_positions,)                       float32
    label          (n_zones,)                           float32

Usage:
    python -m data.preprocess \
        --data_folder ./data/dataset \
        --output_file ./data/dataset.h5
"""
import argparse
import os
import pickle

import h5py
import numpy as np
from tqdm import tqdm


def preprocess(data_folder: str, output_file: str, max_index: int = int(1e9)):
    """Pack all experiments into a single HDF5 file."""

    failed_file = os.path.join(data_folder, "failed_experiments.txt")
    skip_set: set[int] = set()
    if os.path.exists(failed_file):
        with open(failed_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        skip_set.add(int(line.split()[-1].rstrip(":")))
                    except ValueError:
                        pass

    exp_folders = sorted(
        [
            d for d in os.listdir(data_folder)
            if d.startswith("experiment_") and os.path.isdir(os.path.join(data_folder, d))
        ],
        key=lambda x: int(x.split("_")[-1]),
    )

    with h5py.File(output_file, "w") as h5f:
        h5f.attrs["version"] = 1
        n_written = 0

        for folder_name in tqdm(exp_folders, desc="Preprocessing"):
            idx = int(folder_name.split("_")[-1])
            if idx in skip_set or idx > max_index:
                continue

            folder_path = os.path.join(data_folder, folder_name)
            sensor_path = os.path.join(folder_path, "sensor_log.pkl")
            label_path = os.path.join(folder_path, "label.npy")
            xpos_path = os.path.join(folder_path, "x_positions.npy")
            done_path = os.path.join(folder_path, "done.txt")

            if not all(os.path.exists(p) for p in [sensor_path, label_path, done_path]):
                print(f"Skipping {folder_name}: missing files")
                continue

            try:
                with open(sensor_path, "rb") as f:
                    sensor_log = pickle.load(f)  # (n_positions, frames_per_poke, 3)
                label = np.load(label_path)
                x_positions = np.load(xpos_path) if os.path.exists(xpos_path) else None

                grp = h5f.create_group(folder_name)
                grp.create_dataset("sensor_log", data=np.array(sensor_log, dtype=np.float32))
                grp.create_dataset("label", data=label.astype(np.float32))
                if x_positions is not None:
                    grp.create_dataset("x_positions", data=x_positions.astype(np.float32))

                n_written += 1

            except Exception as e:
                print(f"Error processing {folder_name}: {e}")
                if folder_name in h5f:
                    del h5f[folder_name]

        print(f"Written {n_written} experiments to {output_file}")
        h5f.attrs["n_experiments"] = n_written


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", default="./data/dataset")
    parser.add_argument("--output_file", default="./data/dataset.h5")
    parser.add_argument("--max_index", type=int, default=int(1e9))
    args = parser.parse_args()
    preprocess(args.data_folder, args.output_file, args.max_index)
