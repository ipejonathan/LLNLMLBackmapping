"""
inference.py

Script for distributed inference using a trained LitUCG2CGNoiseNet model.

- Loads UCG position trajectories.
- Runs reverse diffusion to generate CG structures.
- Handles multi-GPU (DistributedDataParallel) setup.
- Saves the generated CG positions to an output directory.
"""

import os
import torch
import numpy as np
import argparse
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from torch.utils.data import DataLoader

from modules.diffusion_model import LitUCG2CGNoiseNet

####################### LC Distributed Setup Hack #######################

# Hack to map environment variables for LC (Livermore Computing) clusters
if 'OMPI_COMM_WORLD_RANK'       in os.environ: os.environ["RANK"]       = os.environ['OMPI_COMM_WORLD_RANK']
if 'OMPI_COMM_WORLD_SIZE'       in os.environ: os.environ["WORLD_SIZE"] = os.environ['OMPI_COMM_WORLD_SIZE']
if 'OMPI_COMM_WORLD_LOCAL_RANK' in os.environ: os.environ["LOCAL_RANK"] = os.environ['OMPI_COMM_WORLD_LOCAL_RANK']

####################### Main #######################

if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--ucg-file',      type=str, help='path to numpy .npz file containing ucg simulation samples')
    parser.add_argument('--out-dir',       type=str, help='path to output directory')
    parser.add_argument('--cg-generator',  type=str, help='path to UCG-to-CG generator checkpoint file')
    args = parser.parse_args()

    # Distributed setup
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.distributed.init_process_group(backend="gloo")
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    device_id = rank % torch.cuda.device_count()
    device = torch.device(f'cuda:{device_id}')
    print(f'{world_size=:}, {rank=:}, {device=:}', flush=True)

    ####################### Data Setup #######################

    data = np.load(args.ucg_file, allow_pickle=True)
    ucg_pos_traj = data['positions_ucg']
    
    # Evenly shard data across GPUs
    local_data = ucg_pos_traj[rank::world_size]
    loader = DataLoader(local_data, batch_size=32)

    ####################### Model Setup #######################
    ucg2cg_generator = LitUCG2CGNoiseNet.load_from_checkpoint(
        args.cg_generator,
        ucg_index_file="/p/gpfs1/splash/hmc_project/cg_fingerprints_aligned_to_gdom_and_crd_membrane_alignment/all_indices_per_cluster.npz"
    ).to(device)
    ucg2cg_generator.eval()

    ####################### Start Generation #######################
    pred_cg = []
    with torch.inference_mode():
        for batch in tqdm(loader, position=rank, desc=f'GPU {rank}'):
            ucg_pos = batch.to(device)

            # Center UCG positions
            origin = ucg_pos.mean(dim=1, keepdim=True)
            ucg_pos = ucg_pos - origin

            # Generate predicted CG displacements via diffusion
            pred_cg_disp = ucg2cg_generator.generate(ucg_pos, num_steps=500)

            B, T, N_cg, D = pred_cg_disp.shape
            N_ucg = ucg_pos.shape[1]   # (, 40)

            scatter_idx = ucg2cg_generator.scatter_idx.to(device).contiguous()  # (751,)

            # Expand scatter_idx to (B, T, 751)
            scatter_idx_expanded = scatter_idx.view(1, 1, -1).expand(B, T, -1)

            # Expand UCG positions to (B, T, 40, 3)
            ucg_pos_expanded = ucg_pos.unsqueeze(1).expand(-1, T, -1, -1)

            # Gather UCG reference positions according to scatter indices
            ucg_ref_pos = torch.gather(
                ucg_pos_expanded,
                dim=2,
                index=scatter_idx_expanded.unsqueeze(-1).expand(-1, -1, -1, 3)
            )

            # Final CG positions = UCG references + predicted displacements (last step)
            final_cg_disp = pred_cg_disp[:, -1, :, :]  # (B, 751, 3)
            final_ucg_ref = ucg_ref_pos[:, -1, :, :]   # (B, 751, 3)
            pred_cg_pos = final_ucg_ref + final_cg_disp

            pred_cg.append(pred_cg_pos.cpu())

    ####################### Gather Results #######################

    # Concatenate local results
    pred_cg = torch.cat(pred_cg)
    local_pred = pred_cg.numpy()

    # Gather all predictions from all ranks
    gathered = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(gathered, local_pred)

    if rank == 0:
        all_pred_cg = np.concatenate(gathered, axis=0)
        np.save(os.path.join(args.out_dir, 'pred-cg-500.npy'), all_pred_cg)

    torch.distributed.destroy_process_group()
