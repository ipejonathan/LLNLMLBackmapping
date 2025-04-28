"""
train_lassen.py

Distributed training script for LitUCG2CGNoiseNet on Lassen (LC cluster).

- Initializes distributed environment for multi-GPU training.
- Creates the PyTorch Lightning model and datamodule instances.
- Launches training with DDP strategy across 8 nodes.
"""

####################### LC Hack: Workaround for CPU Affinity #######################
import os
del os.environ['OMP_PLACES']
del os.environ['OMP_PROC_BIND']

import torch
import numpy as np

####################### LC Hack: Distributed Setup #######################

import torch.distributed as dist

if 'OMPI_COMM_WORLD_RANK' in os.environ:
    os.environ["RANK"] = os.environ['OMPI_COMM_WORLD_RANK']
if 'OMPI_COMM_WORLD_SIZE' in os.environ:
    os.environ["WORLD_SIZE"] = os.environ['OMPI_COMM_WORLD_SIZE']
if 'OMPI_COMM_WORLD_LOCAL_RANK' in os.environ:
    os.environ["LOCAL_RANK"] = os.environ['OMPI_COMM_WORLD_LOCAL_RANK']

dist.init_process_group(backend="nccl", init_method="env://")

####################### Create Lightning Modules and Datamodule Instances #######################

from datamodules.ucg2cg import UCG2CGDataModule
from modules.diffusion_model import LitUCG2CGNoiseNet
from glob import glob

# Load dataset
datamodule = UCG2CGDataModule(
    cg_files       = sorted(glob('/p/gpfs1/splash/hmc_project/cg_fingerprints_aligned_to_gdom_and_crd_membrane_alignment/pfpatch_*.npz')),
    ucg_files      = sorted(glob('/p/gpfs1/splash/hmc_project/ucg_npz_data_ucg_40site_aligned_to_gdom_and_crd_membrane_alignment/pfpatch_*_ucg.npz')),
    ucg_index_file = "/p/gpfs1/splash/hmc_project/cg_fingerprints_aligned_to_gdom_and_crd_membrane_alignment/all_indices_per_cluster.npz",
    batch_size     = 64,
    num_workers    = 8,
    train_size     = 0.9,
)

# Initialize the model
noise_net = LitUCG2CGNoiseNet(
    init_dim       = 120, # 39 UCG beads × 3 + 1 special token × 3
    dim            = 512,
    ff_dim         = 2048,
    num_heads      = 8,
    num_layers     = 4,
    ucg_index_file = "/p/gpfs1/splash/hmc_project/cg_fingerprints_aligned_to_gdom_and_crd_membrane_alignment/all_indices_per_cluster.npz",
    dropout        = 0.0,
    learn_rate     = 1e-4,
)

# Optional: load model from checkpoint instead of initializing from scratch
# ckpt_path = './lit_logs/ras-raf-test/version_2/checkpoints/epoch=44-step=14625.ckpt'
# noise_net = LitUCG2CGNoiseNet.load_from_checkpoint(ckpt_path)

################################# Start Training Session #####################################

import pytorch_lightning as L
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

trainer = L.Trainer(
    max_steps = 1000000,
    log_every_n_steps  = 50, # Logging frequency based on number of batches
    val_check_interval = 1.0, # Validate at the end of every epoch
    strategy  ='ddp', # Distributed Data Parallel
    accelerator='gpu',
    devices='auto', # Use all visible GPUs
    num_nodes=8, # Training across 8 nodes
    logger    = TensorBoardLogger(save_dir='./lit_logs/', name='ras-raf-test'),
    callbacks = [TQDMProgressBar(refresh_rate=100)],
)

# Optional debug prints
# print(f"Datamodule: {datamodule}")
# print(f"Noise Net: {noise_net}")

# Start training
trainer.fit(
    noise_net, datamodule,
    # ckpt_path = ckpt_path, # Uncomment if resuming training
)

# Optional debug print
# print("Training Finished")
