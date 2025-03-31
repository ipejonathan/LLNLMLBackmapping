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

# load datamodule
datamodule = UCG2CGDataModule(
    cg_files       = sorted(glob('/p/gpfs1/splash/hmc_project/cg_fingerprints_aligned_to_gdom_and_crd_membrane_alignment/pfpatch_*.npz')),
    ucg_files      = sorted(glob('/p/gpfs1/splash/hmc_project/ucg_npz_data_ucg_40site_aligned_to_gdom_and_crd_membrane_alignment/pfpatch_*_ucg.npz')),
    ucg_index_file = "/p/gpfs1/splash/hmc_project/cg_fingerprints_aligned_to_gdom_and_crd_membrane_alignment/all_indices_per_cluster.npz",
    batch_size     = 64,
    num_workers    = 8,
    train_size     = 0.9,
)

# load model
ckpt_path = '/p/gpfs1/ipe1/LLNLMLBackmapping/lit_logs/ras-raf-test/version_3/checkpoints/epoch=42-step=13975.ckpt'
noise_net = LitUCG2CGNoiseNet.load_from_checkpoint(ckpt_path, ucg_index_file = "/p/gpfs1/splash/hmc_project/cg_fingerprints_aligned_to_gdom_and_crd_membrane_alignment/all_indices_per_cluster.npz")

# do analysis
import torch
import numpy as np

from utils.datautils import DataUtils
from utils.viz import plot_rmsds


def generate_rmsd_plot(num_steps, file_name):
    # Collect RMSDs
    all_rmsds = []

    datamodule.setup()
    val_loader = datamodule.val_dataloader()

    noise_net.eval()
    device = torch.device()

    for batch in val_loader:
        ucg_pos = batch["ucg_pos"].to(device)  # shape: (B, N, 3)
        cg_pos = batch["cg_disp"].numpy()    # shape: (B, N, 3)

        with torch.no_grad():
            pred_cg_pos = noise_net.generate(ucg_pos, num_steps)
            
        pred_cg_pos = pred_cg_pos[:,-1,:,:]
        pred_cg_pos = pred_cg_pos.cpu().numpy()

        batch_rmsds = DataUtils.rmsd(pred_cg_pos, cg_pos)  # shape: (B,)
        all_rmsds.extend(batch_rmsds.tolist())

    all_rmsds = np.array(all_rmsds)

    # Plotting
    plot_rmsds(all_rmsds, title=f"Validation RMSD Distribution - {num_steps} steps", filename=file_name)


plot_rmsds(num_steps=50, file_name="val_rmsds_50.png")
plot_rmsds(num_steps=100, file_name="val_rmsds_100.png")
plot_rmsds(num_steps=500, file_name="val_rmsds_500.png")