#!/bin/bash
#BSUB -nnodes 1            # Number of nodes
#BSUB -W 1:00              # Walltime
#BSUB -G cancer            # Account
#BSUB -J v2cg
#BSUB -o v2cg-%J.log
#BSUB -q pbatch
#BSUB -alloc_flags ipisolate

# Environment setup
module load cuda/11.8
source /usr/workspace/hsu16/opence191/anaconda/bin/activate
conda activate torch

# Distributed setup
lrun -T1 hostname
firsthost=$(lrun -N1 -n1 hostname)
echo "First host: $firsthost"
export MASTER_ADDR=$firsthost
export MASTER_PORT=23456

## Launch multi-GPU generation
lrun -T4 --gpubind=off python ./lit_ras/inference.py \
    --ucg-file /p/gpfs1/splash/hmc_project/ucg_npz_data_ucg_40site_aligned_to_gdom_and_crd_membrane_alignment/pfpatch_000001928827_ucg.npz \
    --out-dir /p/gpfs1/ipe1/LLNLMLBackmapping \
    --cg-generator /p/gpfs1/ipe1/LLNLMLBackmapping/lit_logs/ras-raf-test/version_4/checkpoints/epoch=1800-step=585325.ckpt
