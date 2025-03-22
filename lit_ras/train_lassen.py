# LC HACK: workaround so that "import torch" will not change CPU affinity
import os
del os.environ['OMP_PLACES']
del os.environ['OMP_PROC_BIND']

import torch
import numpy as np

################################# LC hack for distributed training #########################

import torch.distributed as dist

if 'OMPI_COMM_WORLD_RANK' in os.environ:
    os.environ["RANK"] = os.environ['OMPI_COMM_WORLD_RANK']
if 'OMPI_COMM_WORLD_SIZE' in os.environ:
    os.environ["WORLD_SIZE"] = os.environ['OMPI_COMM_WORLD_SIZE']
if 'OMPI_COMM_WORLD_LOCAL_RANK' in os.environ:
    os.environ["LOCAL_RANK"] = os.environ['OMPI_COMM_WORLD_LOCAL_RANK']

dist.init_process_group(backend="nccl", init_method="env://")

################################# Create lightning module and datamodule instances ####################################

from datamodules.ucg2cg import UCG2CGDataModule
from modules.diffusion_model import LitUCG2CGNoiseNet
from glob import glob

datamodule = UCG2CGDataModule(
    cg_files       = sorted(glob('/p/gpfs1/splash/hmc_project/cg_fingerprints_aligned_to_gdom_and_crd_membrane_alignment/pfpatch_*.npz')),
    ucg_files      = sorted(glob('/p/gpfs1/splash/hmc_project/ucg_npz_data_ucg_40site_aligned_to_gdom_and_crd_membrane_alignment/pfpatch_*_ucg.npz')),
    ucg_index_file = "/p/gpfs1/splash/hmc_project/cg_fingerprints_aligned_to_gdom_and_crd_membrane_alignment/all_indices_per_cluster.npz",
    batch_size     = 64,
    num_workers    = 8,
    train_size     = 0.9,
)

noise_net = LitUCG2CGNoiseNet(
    init_dim       = 120, # 39x3 + 1x3
    dim            = 512,
    ff_dim         = 2048,
    num_heads      = 8,
    num_layers     = 8,
    ucg_index_file = "/p/gpfs1/splash/hmc_project/cg_fingerprints_aligned_to_gdom_and_crd_membrane_alignment/all_indices_per_cluster.npz",
    dropout        = 0.0,
    learn_rate     = 1e-4,
)

# Load model weights from a saved checkpoint
ckpt_path = './lit_logs/ras-raf-test/version_2/checkpoints/epoch=44-step=14625.ckpt'
noise_net = LitUCG2CGNoiseNet.load_from_checkpoint(ckpt_path)

################################# Start training session #####################################

import pytorch_lightning as L
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

trainer = L.Trainer(
    max_steps = 1000000,
    log_every_n_steps  = 3000,
    val_check_interval = 1.0,
    # limit_val_batches  = 1000,
    strategy  ='ddp',
    accelerator='gpu',
    devices=4,
    num_nodes=8,
    logger    = TensorBoardLogger(save_dir='./lit_logs/', name='ras-raf-test'),
    callbacks = [TQDMProgressBar(refresh_rate=100)],
)

# print(f"Datamodule: {datamodule}")
# print(f"Noise Net: {noise_net}")

# print("Starting Training")
trainer.fit(
    noise_net, datamodule,
    # ckpt_path = ckpt_path,
)
# print("Training Finished")
