#!/bin/bash
# analysis_job.sh
# 
# Batch script to run distributed RMSD analysis on Lassen using model_analysis_distributed.py.
# 
# - Requests compute resources
# - Loads environment and modules
# - Sets distributed environment variables
# - Launches model analysis across GPUs

####################### LSF Job Directives #######################

#BSUB -nnodes 1             # Number of nodes
#BSUB -W 12:00              # Walltime
#BSUB -G cancer             # Account
#BSUB -J distributed-analysis
#BSUB -o distributed-analysis-%J.log
#BSUB -q pbatch             # Queue to use
#BSUB -alloc_flags ipisolate

####################### System Setup #######################

# Raise the number of maxium open files (to be memory-mapped)
# ulimit -n 20000

# Load Anaconda environment
source /usr/workspace/ipe1/anaconda/bin/activate
module load cuda/11.8.0
conda activate opence-1.9.1

####################### Distributed Setup #######################

# Record each node used
lrun -T1 hostname

# Set distributed training variables (master address and port)
firsthost=$(lrun -N1 -n1 hostname)
echo "First host: $firsthost"

# Set MASTER_ADDR to hostname of first compute node in allocation
export MASTER_ADDR=$firsthost

# Set MASTER_PORT to any used port number
export MASTER_PORT=23456

####################### Launch Multi-GPU Distributed Analysis #######################

lrun -T4 --gpubind=off python ./lit_ras/model_analysis_distributed.py \
    --out-filename /p/gpfs1/ipe1/LLNLMLBackmapping/model_exp2_rmsd_analysis.png \
    --cg-generator /p/gpfs1/ipe1/LLNLMLBackmapping/lit_logs/ras-raf-test/version_4/checkpoints/epoch=1800-step=585325.ckpt