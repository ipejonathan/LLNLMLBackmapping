"""
model_analysis.py

Validation RMSD analysis script for a trained LitUCG2CGNoiseNet model.

- Loads validation data
- Generates CG displacements via reverse diffusion
- Computes RMSDs against ground truth
- Plots the RMSD distribution
"""

####################### LC Hack: Workaround for CPU Affinity #######################

import os
del os.environ['OMP_PLACES']
del os.environ['OMP_PROC_BIND']

import torch
import numpy as np

####################### LC Hack for Distributed Setup #######################

import torch.distributed as dist

if 'OMPI_COMM_WORLD_RANK' in os.environ:
    os.environ["RANK"] = os.environ['OMPI_COMM_WORLD_RANK']
if 'OMPI_COMM_WORLD_SIZE' in os.environ:
    os.environ["WORLD_SIZE"] = os.environ['OMPI_COMM_WORLD_SIZE']
if 'OMPI_COMM_WORLD_LOCAL_RANK' in os.environ:
    os.environ["LOCAL_RANK"] = os.environ['OMPI_COMM_WORLD_LOCAL_RANK']

dist.init_process_group(backend="nccl", init_method="env://")

####################### Load Lightning Modules and Data #######################

from datamodules.ucg2cg import UCG2CGDataModule
from modules.diffusion_model import LitUCG2CGNoiseNet
from glob import glob

# Load datamodule
datamodule = UCG2CGDataModule(
    cg_files       = sorted(glob('/p/gpfs1/splash/hmc_project/cg_fingerprints_aligned_to_gdom_and_crd_membrane_alignment/pfpatch_*.npz')),
    ucg_files      = sorted(glob('/p/gpfs1/splash/hmc_project/ucg_npz_data_ucg_40site_aligned_to_gdom_and_crd_membrane_alignment/pfpatch_*_ucg.npz')),
    ucg_index_file = "/p/gpfs1/splash/hmc_project/cg_fingerprints_aligned_to_gdom_and_crd_membrane_alignment/all_indices_per_cluster.npz",
    batch_size     = 64,
    num_workers    = 8,
    train_size     = 0.9,
)

# Load model checkpoint
ckpt_path = '/p/gpfs1/ipe1/LLNLMLBackmapping/lit_logs/ras-raf-test/version_3/checkpoints/epoch=42-step=13975.ckpt'
noise_net = LitUCG2CGNoiseNet.load_from_checkpoint(
    ckpt_path,
    ucg_index_file = "/p/gpfs1/splash/hmc_project/cg_fingerprints_aligned_to_gdom_and_crd_membrane_alignment/all_indices_per_cluster.npz"
)

####################### Perform Validation Analysis #######################

import torch
import numpy as np

from utils.datautils import DataUtils
from utils.viz import plot_rmsds


def generate_rmsd_plot(num_steps, file_name):
    """
    Generates RMSD plot from validation data.

    Args:
        num_steps (int): Number of diffusion steps during generation.
        file_name (str): Path to save the RMSD plot.
    """
    all_rmsds = []

    datamodule.setup()
    val_loader = datamodule.val_dataloader()

    noise_net.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for batch in val_loader:
        ucg_pos = batch["ucg_pos"].to(device)  # (B, N, 3)
        cg_pos = batch["cg_disp"].numpy()    # (B, N, 3)

        with torch.no_grad():
            pred_cg_pos = noise_net.generate(ucg_pos, num_steps)
            
        pred_cg_pos = pred_cg_pos[:,-1,:,:] # Take final timestep output
        pred_cg_pos = pred_cg_pos.numpy()

        batch_rmsds = DataUtils.rmsd(pred_cg_pos, cg_pos)  # (B,)
        all_rmsds.extend(batch_rmsds.tolist())

    all_rmsds = np.array(all_rmsds)

    # Plot RMSD distribution
    title = "Validation RMSD Distribution - " + str(num_steps) + " steps"
    plot_rmsds(all_rmsds, title=title, filename=file_name)

####################### Run Analysis #######################

generate_rmsd_plot(num_steps=500, file_name="val_rmsds_500.png")
print("job finished")

# Uncomment to generate more plots:
# generate_rmsd_plot(num_steps=100, file_name="val_rmsds_100.png")
# generate_rmsd_plot(num_steps=500, file_name="val_rmsds_500.png")