#!/bin/bash
# inference_job.sh
#
# Batch script to run distributed inference using inference.py on Lassen.
#
# - Loads UCG simulation data
# - Uses trained model checkpoint to generate CG structures
# - Saves generated CG structures to output directory

####################### LSF Job Directives #######################

#BSUB -nnodes 1            # Number of nodes
#BSUB -W 1:00              # Walltime
#BSUB -G cancer            # Account
#BSUB -J v2cg
#BSUB -o v2cg-%J.log
#BSUB -q pbatch
#BSUB -alloc_flags ipisolate

####################### Environment Setup #######################

source /usr/workspace/ipe1/anaconda/bin/activate
module load cuda/11.8.0
conda activate opence-1.9.1

####################### Distributed Setup #######################

# Record which nodes are used
lrun -T1 hostname

# Set MASTER_ADDR to the hostname of the first node
firsthost=$(lrun -N1 -n1 hostname)
echo "First host: $firsthost"

export MASTER_ADDR=$firsthost
export MASTER_PORT=23456 # Arbitrary unused port

####################### Launch Multi-GPU Inference #######################

lrun -T4 --gpubind=off python ./lit_ras/inference.py \
    --ucg-file /p/gpfs1/splash/hmc_project/ucg_npz_data_ucg_40site_aligned_to_gdom_and_crd_membrane_alignment/pfpatch_000005132579_ucg.npz \
    --out-dir /p/gpfs1/ipe1/LLNLMLBackmapping \
    --cg-generator /p/gpfs1/ipe1/LLNLMLBackmapping/lit_logs/ras-raf-test/version_4/checkpoints/epoch=1800-step=585325.ckpt
