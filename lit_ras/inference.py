import os
import torch
import numpy as np
import argparse
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from torch.utils.data import DataLoader
from modules.diffusion_model import LitUCG2CGNoiseNet

# LC hack for distributed setup
if 'OMPI_COMM_WORLD_RANK'       in os.environ: os.environ["RANK"]       = os.environ['OMPI_COMM_WORLD_RANK']
if 'OMPI_COMM_WORLD_SIZE'       in os.environ: os.environ["WORLD_SIZE"] = os.environ['OMPI_COMM_WORLD_SIZE']
if 'OMPI_COMM_WORLD_LOCAL_RANK' in os.environ: os.environ["LOCAL_RANK"] = os.environ['OMPI_COMM_WORLD_LOCAL_RANK']

if __name__ == '__main__':
    # CLI arguments
    parser = argparse.ArgumentParser()
    # parser.add_argument('--v-file',        type=str, help='path to numpy .npy file containing v arrays in the shape of [N, 4]')
    parser.add_argument('--ucg-file',      type=str, help='path to numpy .npz file containing ucg simulation samples')
    parser.add_argument('--out-dir',       type=str, help='path to output directory')
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

    # Data setup
    data = np.load(args.ucg_file, allow_pickle=True)
    ucg_pos_traj = data['positions_ucg']
    # print(f'Loaded {v.shape[0]} descriptors (v)', flush=True)
    loader = DataLoader(ucg_pos_traj, batch_size=32)

    # Model setup
    ucg2cg_generator = LitUCG2CGNoiseNet.load_from_checkpoint(args.cg_generator, ucg_index_file="/p/gpfs1/splash/hmc_project/cg_fingerprints_aligned_to_gdom_and_crd_membrane_alignment/all_indices_per_cluster.npz").to(device)
    ucg2cg_generator.eval()

    # Start generations
    pred_cg = []
    with torch.inference_mode():
        for batch in tqdm(loader, position=rank, desc=f'GPU {rank}'):
            ucg_pos = batch.to(device)
            pred_cg_disp = ucg2cg_generator.generate(ucg_pos, num_steps=500)

            B, T, N_cg, D = pred_cg_disp.shape
            N_ucg = ucg_pos.shape[1]            # e.g., 40

            scatter_idx = ucg2cg_generator.scatter_idx.to(device).contiguous()  # (751,)

            # Expand scatter_idx to shape (B, T, 751, 3)
            scatter_idx_expanded = scatter_idx.view(1, 1, -1).expand(B, T, -1)  # (B, T, 751)

            # Now: gather UCG reference positions
            # Step 1: expand ucg_pos to (B, 1, 40, 3) → (B, T, 40, 3)
            ucg_pos_expanded = ucg_pos.unsqueeze(1).expand(-1, T, -1, -1)  # (B, T, 40, 3)

            # Step 2: use gather to select (B, T, 751, 3)
            ucg_ref_pos = torch.gather(
                ucg_pos_expanded,
                dim=2,
                index=scatter_idx_expanded.unsqueeze(-1).expand(-1, -1, -1, 3)
            )  # (B, T, 751, 3)

            # final_cg_disp = pred_cg_disp[:, -1, :, :]  # (B, 751, 3)
            # final_ucg_ref = ucg_ref_pos[:, -1, :, :]   # (B, 751, 3)
            # pred_cg_pos = final_ucg_ref + final_cg_disp
            pred_cg_pos = ucg_ref_pos + pred_cg_disp

            pred_cg.append(pred_cg_pos.cpu())

    # Concatenate the generation outputs per GPU process
    pred_cg = torch.cat(pred_cg)

    # Gather up all outputs from all GPUs
    gather_list = [torch.zeros_like(pred_cg) for _ in range(world_size)]
    torch.distributed.all_gather(gather_list, pred_cg)
    all_pred_cg = torch.cat(gather_list).numpy()

    # Save generation outputs
    if rank == 0:
        np.save(os.path.join(args.out_dir, 'pred-cg-500.npy'), all_pred_cg)

    torch.distributed.destroy_process_group()
