"""
Training script.

Usage:
    python -m training.train \
        --data_path ./data/dataset.h5 \
        --output_dir ./runs/exp_001

Config can also be provided via YAML:
    python -m training.train --config_path configs/train.yaml
"""
from __future__ import annotations

import os
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import pyrallis

from training.configs import TrainingConfig
from data.dataset import make_dataloaders
from model.model import ArthroscopyModel
from training.losses import MSELoss, HuberLoss


def train(config: TrainingConfig):
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(config.output_dir, exist_ok=True)

    # Save config
    with open(os.path.join(config.output_dir, "config.yaml"), "w") as f:
        pyrallis.dump(config, f)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_loader, val_loader, test_loader, full_ds = make_dataloaders(
        h5_path=config.data_path,
        batch_size=config.batch_size,
        val_split=config.val_split,
        test_split=config.test_split,
        num_workers=config.num_workers,
        seed=config.seed,
    )
    print(f"Dataset: {len(full_ds)} experiments | label range [{full_ds.label_min:.4f}, {full_ds.label_max:.4f}]")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = ArthroscopyModel(config.model).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {n_params:,} trainable parameters")

    # ------------------------------------------------------------------
    # Optimizer & scheduler
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    if config.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs, eta_min=config.learning_rate * 0.01
        )
    elif config.lr_scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    else:
        scheduler = None

    criterion = HuberLoss()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    history: list[dict] = []
    best_val_loss = float("inf")

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            sensor = batch["sensor"].to(device)   # (B, P, F, 3)
            label = batch["label"].to(device)      # (B, n_zones)

            pred = model(sensor)
            loss = criterion(pred, label)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                sensor = batch["sensor"].to(device)
                label = batch["label"].to(device)
                pred = model(sensor)
                val_loss += criterion(pred, label).item()
        val_loss /= len(val_loader)

        if scheduler:
            scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:3d}/{config.epochs}  train={train_loss:.5f}  val={val_loss:.5f}  lr={lr:.2e}")

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        # Checkpoint best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(config.output_dir, "best_model.pt"))

    # Save final model and training history
    torch.save(model.state_dict(), os.path.join(config.output_dir, "final_model.pt"))
    with open(os.path.join(config.output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest val loss: {best_val_loss:.5f}")
    print(f"Checkpoints saved to: {config.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default=None)
    args, remaining = parser.parse_known_args()

    if args.config_path:
        config = pyrallis.parse(config_class=TrainingConfig, config_path=args.config_path, args=remaining)
    else:
        config = pyrallis.parse(config_class=TrainingConfig, args=remaining)

    train(config)
