import os
import torch
import numpy as np
import argparse
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from torch.utils.data import DataLoader
from lit_ras.modules.diffusion_model import LitUCG2CGNoiseNet

# LC hack for distributed setup
if 'OMPI_COMM_WORLD_RANK'       in os.environ: os.environ["RANK"]       = os.environ['OMPI_COMM_WORLD_RANK']
if 'OMPI_COMM_WORLD_SIZE'       in os.environ: os.environ["WORLD_SIZE"] = os.environ['OMPI_COMM_WORLD_SIZE']
if 'OMPI_COMM_WORLD_LOCAL_RANK' in os.environ: os.environ["LOCAL_RANK"] = os.environ['OMPI_COMM_WORLD_LOCAL_RANK']

if __name__ == '__main__':
    # CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--v-file',        type=str, help='path to numpy .npy file containing v arrays in the shape of [N, 4]')
    parser.add_argument('--out-dir',       type=str, help='path to output directory')
    # parser.add_argument('--ucg-generator', type=str, help='path to v-to-UCG generator checkpoint file')
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
    v = np.load(args.v_file)
    # print(f'Loaded {v.shape[0]} descriptors (v)', flush=True)
    # logv = torch.tensor(v, dtype=torch.float).log10()  # the v-to-UCG generator takes log(v) values instead of v
    # logv = logv.chunk(world_size)[rank]
    # TODO: load npz file into dataloader (no need to pass into dataset since generate only needs ucg_pos as tensor input)
    loader = DataLoader(logv, batch_size=32)

    # Model setup
    # v2ucg_generator = V2UCG_Generator.load_from_checkpoint(args.ucg_generator).to(device)
    ucg2cg_generator = LitUCG2CGNoiseNet.load_from_checkpoint(args.cg_generator, ucg_index_file='../sample-data/cg/all_indices_per_cluster.npz').to(device)
    # v2ucg_generator.eval()
    ucg2cg_generator.eval()

    # Start generations
    # pred_ucg, pred_cg = [], []
    pred_cg = []
    with torch.inference_mode():
        for batch in tqdm(loader, position=rank, desc=f'GPU {rank}'):
            # pred_ucg_i = v2ucg_generator.generate(batch.to(device), num_steps=128)
            # pred_cg_i  = ucg2cg_generator.generate(pred_ucg_i, num_steps=128)
            pred_cg_i = ucg2cg_generator.generate(batch.to(device), num_steps=128)
            # pred_ucg.append(pred_ucg_i.cpu())
            pred_cg.append(pred_cg_i.cpu())

    # Concatenate the generation outputs per GPU process
    # pred_ucg = torch.cat(pred_ucg)
    pred_cg = torch.cat(pred_cg)

    # Gather up all outputs from all GPUs
    # gather_list = [torch.zeros_like(pred_ucg) for _ in range(world_size)]
    # torch.distributed.all_gather(gather_list, pred_ucg)
    # all_pred_ucg = torch.cat(gather_list).numpy()

    gather_list = [torch.zeros_like(pred_cg) for _ in range(world_size)]
    torch.distributed.all_gather(gather_list, pred_cg)
    all_pred_cg = torch.cat(gather_list).numpy()

    # Save generation outputs
    if rank == 0:
        # np.save(os.path.join(args.out_dir, 'pred-ucg.npy'), all_pred_ucg)
        np.save(os.path.join(args.out_dir, 'pred-cg.npy'), all_pred_cg)

    torch.distributed.destroy_process_group()
