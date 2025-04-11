import os
import torch
import numpy as np
import argparse
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from torch.utils.data import DataLoader
from modules.diffusion_model import LitUCG2CGNoiseNet
from datamodules.ucg2cg import UCG2CGDataModule
from glob import glob
from utils.viz import plot_rmsds_nice

from utils.datautils import DataUtils
# LC hack for distributed setup
if 'OMPI_COMM_WORLD_RANK'       in os.environ: os.environ["RANK"]       = os.environ['OMPI_COMM_WORLD_RANK']
if 'OMPI_COMM_WORLD_SIZE'       in os.environ: os.environ["WORLD_SIZE"] = os.environ['OMPI_COMM_WORLD_SIZE']
if 'OMPI_COMM_WORLD_LOCAL_RANK' in os.environ: os.environ["LOCAL_RANK"] = os.environ['OMPI_COMM_WORLD_LOCAL_RANK']



if __name__ == '__main__':

    # CLI arguments
    parser = argparse.ArgumentParser()
    # parser.add_argument('--v-file',        type=str, help='path to numpy .npy file containing v arrays in the shape of [N, 4]')
    parser.add_argument('--out-filename',       type=str, help='path to output directory and file name')
    parser.add_argument('--cg-generator',  type=str, help='path to UCG-to-CG generator checkpoint file')
    args = parser.parse_args()

    # DDP setup
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.distributed.init_process_group(backend="gloo")
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    device_id = rank % torch.cuda.device_count()
    device = torch.device(f'cuda:{device_id}')
    print(f'{world_size=:}, {rank=:}, {device=:}', flush=True)


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
    
    # Model setup
    ucg2cg_generator = LitUCG2CGNoiseNet.load_from_checkpoint(args.cg_generator, ucg_index_file="/p/gpfs1/splash/hmc_project/cg_fingerprints_aligned_to_gdom_and_crd_membrane_alignment/all_indices_per_cluster.npz").to(device)
    ucg2cg_generator.eval()

    # Start generations
    all_rmsds = []
    with torch.inference_mode():
        
        for batch in tqdm(loader, position=rank, desc=f'GPU {rank}'):
            # pred_cg_i = ucg2cg_generator.generate(batch.to(device), num_steps=128)
            # pred_cg.append(pred_cg_i.cpu())
    
            ucg_pos = batch["ucg_pos"].to(device)  # shape: (B, N, 3)
            cg_disp = batch["cg_disp"].to(device)    # shape: (B, N, 3)

            pred_cg_i = ucg2cg_generator.generate(ucg_pos, num_steps=128)
            
            # take the last frame
            pred_cg_disp = pred_cg_i[:,-1,:,:]
            # pred_cg_pos = pred_cg_pos.cpu().numpy()
            pred_cg_disp = pred_cg_disp.to(device)

            batch_rmsds = DataUtils.rmsd_torch(pred_cg_disp, cg_disp)  # shape: (B,)
            # all_rmsds.extend(batch_rmsds.cpu().tolist())
            all_rmsds.append(batch_rmsds)
        
        all_rmsds = torch.cat(all_rmsds, dim=0)

    # Gather up all outputs from all GPUs
    gather_list = [torch.zeros_like(all_rmsds) for _ in range(world_size)]
    torch.distributed.all_gather(gather_list, all_rmsds)

    if rank == 0:
        all_rmsds_gpu = torch.cat([t.cpu() for t in gather_list]).numpy()

        # Plotting
        title = "Validation RMSD Distribution - " + str(128) + " steps"
        plot_rmsds_nice(all_rmsds_gpu, title=title, filename=args.out_filename)

    torch.distributed.destroy_process_group()
