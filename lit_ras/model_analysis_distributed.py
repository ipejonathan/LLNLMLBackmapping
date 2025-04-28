"""
model_analysis_distributed.py

Distributed evaluation script for a trained LitUCG2CGNoiseNet model.

- Loads a validation dataset.
- Runs reverse diffusion to generate CG structures.
- Computes per-batch RMSD (Root-Mean-Square Deviation) between predicted and true CG displacements.
- Gathers all RMSDs across GPUs and plots the distribution.
"""

import os
import torch
import numpy as np
import argparse
from glob import glob
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

from modules.diffusion_model import LitUCG2CGNoiseNet
from datamodules.ucg2cg import UCG2CGDataModule
from utils.viz import plot_rmsds_nice
from utils.datautils import DataUtils

####################### LC Distributed Setup Hack #######################

if 'OMPI_COMM_WORLD_RANK'       in os.environ: os.environ["RANK"]       = os.environ['OMPI_COMM_WORLD_RANK']
if 'OMPI_COMM_WORLD_SIZE'       in os.environ: os.environ["WORLD_SIZE"] = os.environ['OMPI_COMM_WORLD_SIZE']
if 'OMPI_COMM_WORLD_LOCAL_RANK' in os.environ: os.environ["LOCAL_RANK"] = os.environ['OMPI_COMM_WORLD_LOCAL_RANK']


####################### Main #######################

if __name__ == '__main__':

    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-filename',       type=str, help='path to output directory and file name')
    parser.add_argument('--cg-generator',  type=str, help='path to UCG-to-CG generator checkpoint file')
    args = parser.parse_args()

    ####################### Distributed Setup #######################

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.distributed.init_process_group(backend="gloo")
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    device_id = rank % torch.cuda.device_count()
    device = torch.device(f'cuda:{device_id}')
    print(f'{world_size=:}, {rank=:}, {device=:}', flush=True)

    ####################### Data Setup #######################

    datamodule = UCG2CGDataModule(
        cg_files       = sorted(glob('/p/gpfs1/splash/hmc_project/cg_fingerprints_aligned_to_gdom_and_crd_membrane_alignment/pfpatch_*.npz')),
        ucg_files      = sorted(glob('/p/gpfs1/splash/hmc_project/ucg_npz_data_ucg_40site_aligned_to_gdom_and_crd_membrane_alignment/pfpatch_*_ucg.npz')),
        ucg_index_file = "/p/gpfs1/splash/hmc_project/cg_fingerprints_aligned_to_gdom_and_crd_membrane_alignment/all_indices_per_cluster.npz",
        batch_size     = 64,
        num_workers    = 8,
        train_size     = 0.9,
    )
    datamodule.setup()
    loader = datamodule.val_dataloader()
    
    ####################### Model Setup #######################

    ucg2cg_generator = LitUCG2CGNoiseNet.load_from_checkpoint(
        args.cg_generator,
        ucg_index_file="/p/gpfs1/splash/hmc_project/cg_fingerprints_aligned_to_gdom_and_crd_membrane_alignment/all_indices_per_cluster.npz"
    ).to(device)
    ucg2cg_generator.eval()

    ####################### Generation and RMSD Calculation #######################

    all_rmsds = []
    with torch.inference_mode():
        for batch in tqdm(loader, position=rank, desc=f'GPU {rank}'):
            ucg_pos = batch["ucg_pos"].to(device)  # (B, N, 3)
            cg_disp = batch["cg_disp"].to(device)    # (B, N, 3)

            # Generate CG displacements
            pred_cg_disp_traj = ucg2cg_generator.generate(ucg_pos, num_steps=500)
            
            # Take final step prediction
            pred_cg_disp = pred_cg_disp_traj[:,-1,:,:]
            pred_cg_disp = pred_cg_disp.to(device)

            # Calculate RMSD between predicted and true displacements
            batch_rmsds = DataUtils.rmsd_torch(pred_cg_disp, cg_disp)  # shape: (B,)
            all_rmsds.append(batch_rmsds)
        
        # Concatenate all batches
        all_rmsds = torch.cat(all_rmsds, dim=0)

    ####################### Gather RMSD Values #######################

    all_rmsds_list = all_rmsds.cpu().tolist()

    gather_list = None
    if rank == 0:
        gather_list = [None for _ in range(world_size)]

    torch.distributed.gather_object(all_rmsds_list, gather_list, dst=0)

    ####################### Plot and Save Results #######################

    if rank == 0:
        # Flatten nested lists
        all_rmsds_flat = [r for sublist in gather_list for r in sublist]
        all_rmsds_gpu = np.array(all_rmsds_flat)

        # Plot RMSD distribution
        title = "Validation RMSD Distribution - " + str(500) + " steps"
        plot_rmsds_nice(all_rmsds_gpu, title=title, filename=args.out_filename)

    torch.distributed.destroy_process_group()