# Arthroscopy POC — Claude Code Context

## Project Goal
Thesis POC for arthroscopic tissue stiffness estimation.
- A probe with a force/torque sensor scans shoulder joint tissue.
- Goal: predict tissue stiffness (Young's modulus) from sensor readings.
- Approach: FEM simulation → synthetic dataset → encoder-decoder ML model.

## Tech Stack
- Python 3.10+
- PyTorch (model + FEM autodiff via `requires_grad`)
- pyrallis (YAML ↔ dataclass config)
- HDF5 / h5py (dataset storage)
- scipy Delaunay (mesh generation)

---

## Project Structure
```
arthroscopy_poc/
├── CLAUDE.md               ← you are here
├── pyproject.toml          ← dependencies + packaging
├── configs/
│   └── default.yaml        ← default simulation config
├── sim/                    ← simulation module
│   ├── configs.py          ← simulation config dataclasses (Tissue, Probe, Simulation, Experiment)
│   ├── trajectory.py       ← probe movement trajectories
│   ├── shapes.py           ← FEM + spring-mass shapes (SoftShapeFiniteElement, etc.)
│   ├── tissue.py           ← rectangle tissue phantom builder
│   ├── simulation.py       ← SoftObjectSimulation engine
│   └── generate_dataset.py ← parallel dataset generation script
├── data/
│   ├── configs.py          ← dataset generation config (DatasetConfig)
│   ├── dataset.py          ← PyTorch Dataset + DataLoader helpers
│   └── preprocess.py       ← raw sim output → HDF5
├── model/
│   ├── encoder.py          ← PokeEncoder + ScanEncoder (1D-CNN)
│   ├── decoder.py          ← StiffnessDecoder (MLP)
│   └── model.py            ← ArthroscopyModel (full pipeline)
├── training/
│   ├── configs.py          ← model + training config (ModelConfig, TrainingConfig)
│   ├── losses.py           ← MSE, Huber, RankingLoss
│   └── train.py            ← training loop
└── tests/
    └── test_sim.py
```

---

## Key Concepts

### Tissue Phantom
- Rectangle: width=2.0, height=0.4 (configurable in `TissueConfig`)
- Divided into `n_zones` stiffness zones along the x-axis
- Each zone has a Young's modulus E ∈ [E_min, E_max] — this is the **label**
- Bottom edge is fixed (Dirichlet BC); probe approaches from above

### Probe
- Circular tip (simplified arthroscope instrument)
- Performs "poke" trajectories: descend to `tissue_top_y - penetration_depth`, then return
- Scans `n_scan_positions` evenly-spaced x-coordinates across the tissue

### Sensor Model
- Per probe vertex: `[Fx, Fy, Mz]` (contact force + moment about probe center)
- Aggregated to a single `(3,)` vector per frame via sum
- Shape of one experiment: `(n_positions, frames_per_poke, 3)`

### FEM Physics
- Linear elastic plane stress
- Strain energy minimized per frame via Adam optimizer
- Collision penalty: spring constant × penetration² added to energy
- See `SoftShapeFiniteElement.compute_strain_energy_pytorch()` in `sim/shapes.py`

---

## How to Run

### Install
```bash
pip install -r requirements.txt
```

### Generate a small test dataset (1 experiment, no multiprocessing)
```bash
python -m sim.generate_dataset \
    --data_folder ./data/test_dataset \
    --num_experiments 1 \
    --num_processes 1 \
    --experiment_config_path configs/default.yaml
```

### Preprocess to HDF5
```bash
python -m data.preprocess \
    --data_folder ./data/test_dataset \
    --output_file ./data/test.h5
```

### Train
```bash
python -m training.train \
    --data_path ./data/test.h5 \
    --output_dir ./runs/exp_001
```

---

## Config System
All configs are pyrallis dataclasses, split by concern:

- `sim/configs.py` — simulation physics
  - `TissueConfig` — tissue geometry and stiffness range
  - `ProbeConfig` / `TrajectoriesConfig` — probe geometry and motion
  - `SimulationConfig` — FEM optimizer, noise, saving
  - `ExperimentConfig` — top-level (Tissue + Probe + Simulation)
- `data/configs.py` — dataset generation
  - `DatasetConfig` — dataset generation parameters
- `training/configs.py` — model + training
  - `ModelConfig` → `EncoderConfig` + `DecoderConfig`
  - `TrainingConfig` — training hyperparameters

Load from YAML: `pyrallis.parse(config_class=ExperimentConfig, config_path="configs/default.yaml", args=[])`

---

## Current Status (as of project init)
- [x] Simulation core (FEM, shapes, trajectories)
- [x] Rectangle tissue phantom with stiffness zones
- [x] Force/torque sensor model (Fx, Fy, Mz)
- [x] Dataset generation pipeline (parallel)
- [x] HDF5 preprocessing
- [x] PyTorch Dataset + DataLoader
- [x] Encoder (1D-CNN per poke) + Decoder (MLP)
- [x] Training loop
- [ ] **First run**: generate dataset, sanity-check sensor signals
- [ ] Tune simulation params (collision_spring_constant, steps, penetration_depth)
- [ ] Evaluate model, analyse what the encoder learns
- [ ] Decide on final model architecture (may replace CNN with Transformer)

---

## Known Issues / TODOs
- `SoftObjectSimulation._closest_edge_tensor` is currently a slow Python loop.
  Vectorise with torch batched ops for speed if needed.
- Sensor aggregation (sum over probe vertices) is a crude approximation of a
  real 6-DOF sensor. May want to model the probe shaft mechanics more carefully.
- `generate_dataset.py` resets tissue state between pokes but not between experiments
  (tissue is rebuilt each time). This is correct but slow; consider caching mesh topology.
- Model `output_dim` in `DecoderConfig` must match `TissueConfig.n_zones`. Currently
  set manually — add a check or auto-wire.

---

## Ported From
Original code: breast palpation simulation (breast_palpation project).
Key changes from that project:
- Shape: semi-circle → rectangle
- Label: binary lump detection → continuous stiffness profile (regression)
- Sensor: Fx/Fy per vertex → aggregated (Fx, Fy, Mz) per frame
- Probe trajectory: arc/sweep → linear scan with pokes
- `set_zone_stiffness()` replaces `add_lump()`
