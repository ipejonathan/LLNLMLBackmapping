#!/bin/bash
#BSUB -nnodes 1             # Number of nodes
#BSUB -W 12:00              # Walltime
#BSUB -G cancer             # Account
#BSUB -J distributed-analysis
#BSUB -o distributed-analysis-%J.log
#BSUB -q pbatch             # Queue to use
#BSUB -alloc_flags ipisolate

# Raise the number of maxium open files (to be memory-mapped)
# ulimit -n 20000

# Load Python environment
source /usr/workspace/ipe1/anaconda/bin/activate
module load cuda/11.8.0
conda activate opence-1.9.1

# Just to record each node we're using in the job output log
lrun -T1 hostname

# Get hostname of the rank-0 node
firsthost=$(lrun -N1 -n1 hostname)
echo "First host: $firsthost"

# Set MASTER_ADDR to hostname of first compute node in allocation
# Set MASTER_PORT to any used port number
export MASTER_ADDR=$firsthost
export MASTER_PORT=23456

## Launch multi-GPU generation
lrun -T4 --gpubind=off python ./lit_ras/model_analysis_distributed.py \
    --out-filename /p/gpfs1/ipe1/LLNLMLBackmapping/distributed_analysis_test.png \
    --cg-generator /p/gpfs1/ipe1/LLNLMLBackmapping/lit_logs/ras-raf-test/version_4/checkpoints/epoch=1800-step=585325.ckpt