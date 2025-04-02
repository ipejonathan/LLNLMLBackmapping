#!/bin/bash
#BSUB -nnodes 8             # Number of nodes
#BSUB -W 12:00              # Walltime
#BSUB -G cancer             # Account
#BSUB -J ucg2cg
#BSUB -o ucg2cg-%J.log
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

# Train
lrun -T4 --gpubind=off python ./lit_ras/train_lassen.py
