"""
PyTorch Dataset wrapping the HDF5 file produced by data/preprocess.py.

Each item:
    sensor_log : (n_positions, frames_per_poke, 3)   float32 tensor
    label      : (n_zones,)                          float32 tensor
    x_positions: (n_positions,)                      float32 tensor (optional)
"""
from __future__ import annotations

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class ArthroscopyDataset(Dataset):
    def __init__(
        self,
        h5_path: str,
        normalize_label: bool = True,
        label_min: float | None = None,
        label_max: float | None = None,
    ):
        """
        Args:
            h5_path: path to the HDF5 file from data/preprocess.py
            normalize_label: if True, normalize label to [0, 1]
            label_min/max: override normalization range (use training set stats for val/test)
        """
        self.h5_path = h5_path
        self.normalize_label = normalize_label

        with h5py.File(h5_path, "r") as f:
            self.keys = sorted(f.keys(), key=lambda k: int(k.split("_")[-1]))

        self._label_min = label_min
        self._label_max = label_max

        if normalize_label and label_min is None:
            self._compute_label_stats()

    def _compute_label_stats(self):
        all_labels = []
        with h5py.File(self.h5_path, "r") as f:
            for k in self.keys:
                all_labels.append(f[k]["label"][:])
        labels = np.concatenate(all_labels)
        self._label_min = float(labels.min())
        self._label_max = float(labels.max())

    @property
    def label_min(self) -> float:
        return self._label_min

    @property
    def label_max(self) -> float:
        return self._label_max

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        key = self.keys[idx]
        with h5py.File(self.h5_path, "r") as f:
            grp = f[key]
            sensor = torch.from_numpy(grp["sensor_log"][:].astype(np.float32))
            label = torch.from_numpy(grp["label"][:].astype(np.float32))
            x_pos = (
                torch.from_numpy(grp["x_positions"][:].astype(np.float32))
                if "x_positions" in grp
                else torch.zeros(sensor.shape[0])
            )

        if self.normalize_label and self._label_min is not None:
            denom = self._label_max - self._label_min
            label = (label - self._label_min) / (denom + 1e-8)

        return {"sensor": sensor, "label": label, "x_positions": x_pos}


def make_dataloaders(
    h5_path: str,
    batch_size: int = 32,
    val_split: float = 0.15,
    test_split: float = 0.10,
    num_workers: int = 4,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, ArthroscopyDataset]:
    """
    Split dataset into train/val/test and return DataLoaders.
    Also returns the full dataset for accessing label stats.
    """
    full_dataset = ArthroscopyDataset(h5_path)

    n = len(full_dataset)
    n_test = int(n * test_split)
    n_val = int(n * val_split)
    n_train = n - n_test - n_val

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        full_dataset, [n_train, n_val, n_test], generator=generator
    )

    # Share label normalization stats from the full dataset
    def make_loader(ds, shuffle):
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return (
        make_loader(train_ds, shuffle=True),
        make_loader(val_ds, shuffle=False),
        make_loader(test_ds, shuffle=False),
        full_dataset,
    )
