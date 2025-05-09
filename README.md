# UCG-mini-MuMMI: UCG-to-CG Backmapping Model

This repository implements a deep learning-based **UCG-to-CG backmapping model** for multiscale molecular dynamics simulations.  
It is part of the **Mini-MuMMI** project, designed to extend multiscale protein modeling workflows by enabling efficient **reconstruction of coarse-grained (CG) protein structures** from ultra-coarse-grained (UCG) representations.

The core model leverages a **Transformer-based diffusion framework** to backmap UCG positions into detailed CG bead structures, allowing seamless transitions between simulation resolutions.

## Overview of System Architecture

- **Input:**  
  - UCG positions (`positions_ucg` arrays in `.npz` files)
  - UCG-to-CG mapping indices (`all_indices_per_cluster.npz`)
- **Model Core:**
  - A **Transformer** backbone processes UCG positions and predicts CG displacements.
  - A **Variance Preserving Diffusion Process** refines predictions across timesteps.
- **Output:**
  - Predicted CG bead positions (`pred-cg-500.npy`)
  - RMSD evaluation plots (`val_rmsds_500.png`)

### Components

| Component | Description |
|-----------|-------------|
| **Data Module** (`datamodules/ucg2cg.py`) | Loads UCG and CG data for training, validation, inference |
| **Model** (`modules/diffusion_model.py`) | Transformer + Diffusion model for noise prediction |
| **Diffuser** (`graphite/diffusion/general.py`) | Implements forward and reverse stochastic processes |
| **Basis Functions** (`graphite/nn/basis.py`) | Embedding utilities (Gaussian Fourier features) |
| **Training scripts** (`train_local.py`, `train_lassen.py`) | Train model locally or distributed |
| **Inference script** (`inference.py`) | Generate CG structures from UCG input |
| **Analysis scripts** (`model_analysis.py`, `model_analysis_distributed.py`) | Evaluate model performance |

## Quick Start

### Testing Training (Local)

```bash
python lit_ras/train_local.py
```

- Runs on CPU or MPS (Mac GPU backend)
- Only use small sample data in `sample-data/`
- For **debugging/testing only**, **not real training**

### Full Training (Lassen Cluster)

Submit distributed training job:

```bash
bsub < jobs/training_job.sh
```

- Distributed training across 8 nodes
- Uses real full datasets under `/p/gpfs1/splash/hmc_project/...`
- Trains model with `train_lassen.py`
- Checkpoints saved in `./lit_logs/`

### Testing Inference: Generate CG Structures (Local - Jupyter Notebook)

The repository includes `inference_test.ipynb` outside the `lit_ras/` directory. This notebook:
- Can be used to test the inference script locally using the `sample-data/`
- Visualizes the protein structure during the diffusion process across defined timesteps
- Provides an interactive way to evaluate model predictions

### Full Inference: Generate CG Structures (Lassen Cluster)

Submit distributed inference job:

```bash
bsub < jobs/inference_job.sh
```

Or manually:

```bash
python lit_ras/inference.py \
    --ucg-file /p/gpfs1/splash/hmc_project/ucg_npz_data_ucg_40site_aligned_to_gdom_and_crd_membrane_alignment/pfpatch_000005132579_ucg.npz \
    --out-dir /p/gpfs1/ipe1/LLNLMLBackmapping \
    --cg-generator /p/gpfs1/ipe1/LLNLMLBackmapping/lit_logs/ras-raf-test/version_4/checkpoints/epoch=1800-step=585325.ckpt
```

- Input: Real UCG trajectory `.npz`
- Output: Predicted CG bead structure `.npy` file

### Model Evaluation (RMSD Analysis)

Submit distributed RMSD analysis job:

```bash
bsub < jobs/analysis_job.sh
```

Or manually:

```bash
python lit_ras/model_analysis_distributed.py \
    --out-filename /path/to/val_rmsds_500.png \
    --cg-generator /path/to/checkpoint.ckpt
```
- Gathers RMSD statistics across GPUs
- Saves plot for validation


## Inputs

| File/Directory | Purpose |
|----------------|---------|
| **sample-data/** | **(Testing only)** Small toy dataset for local debugging |
| **/p/gpfs1/splash/hmc_project/...** | Real CG and UCG datasets for training and inference |
| **/p/gpfs1/splash/hmc_project/cg_fingerprints_aligned_to_gdom_and_crd_membrane_alignment/all_indices_per_cluster.npz** | Mapping of CG beads to UCG beads |

## Outputs

| Output | Description |
|--------|------------|
| `pred-cg-500.npy` | Predicted CG structures generated from UCG input |
| `val_rmsds_500.png` | RMSD distribution plot from validation |
| `distributed-analysis-*.log` | Job logs for distributed analysis |
| `v2cg-*.log` | Logs for inference jobs |


## Important Clarifications

| Environment | Data Used |
|-------------|-----------|
| `train_local.py` + `sample-data/` | For **testing and debugging only** |
| `train_lassen.py` + `/p/gpfs1/splash/hmc_project/...` | For **real model training and inference** |

### Lassen Account Configuration

In all job scripts (`.sh` files), the Anaconda environment is activated using:
```bash
source /usr/workspace/ipe1/anaconda/bin/activate
```

**Important:** Replace `ipe1` with your own Lassen username in all job scripts.

## Requirements

- Python 3.8+
- PyTorch >= 1.13
- PyTorch Lightning >= 1.7
- CUDA 11.8
- Conda environment `opence-1.9.1`
- LLNL Cluster (e.g., Lassen) with `lrun`, `bsub`, multi-GPU support

## Batch Scripts Summary

| Script | Purpose |
|--------|---------|
| `training_job.sh` | Distributed training using `train_lassen.py` |
| `inference_job.sh` | Distributed inference using `inference.py` |
| `analysis_job.sh` | Distributed RMSD evaluation using `model_analysis_distributed.py` |

## Folder Structure

```
├── lit_ras/
│   ├── datamodules/
│   │   └── ucg2cg.py
│   ├── modules/
│   │   └── diffusion_model.py
│   ├── graphite/
│   │   ├── diffusion/
│   │   │   └── general.py
│   │   └── nn/
│   │       └── basis.py
│   ├── utils/
│   │   ├── datautils.py
│   │   └── viz.py
│   ├── train_local.py
│   ├── train_lassen.py
│   ├── inference.py
│   ├── model_analysis.py
│   └── model_analysis_distributed.py
├── jobs/
│   ├── training_job.sh
│   ├── inference_job.sh
│   └── analysis_job.sh
├── inference_test.ipynb
└── sample-data/
```

## Acknowledgments

This work is part of the UCG-mini-MuMMI project at Harvey Mudd College Clinic Program with Lawrence Livermore National Laboratory.
