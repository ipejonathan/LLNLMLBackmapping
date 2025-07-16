#!/bin/bash
# split_inputs_jobb.sh
#
# Batch script to launch distributed training on Lassen using train_lassen.py.
#
# - Requests resources for distributed multi-node, multi-GPU training
# - Loads Python environment
# - Configures distributed setup variables
# - Launches PyTorch Lightning training

####################### LSF Job Directives #######################

#BSUB -nnodes 1             # Number of nodes
#BSUB -W 1:00              # Walltime
#BSUB -G cancer             # Account
#BSUB -J ucg2cg-experiment5
#BSUB -o ucg2cg-%J.log
#BSUB -q pbatch             # Queue to use
#BSUB -alloc_flags ipisolate

####################### System Setup #######################

# Raise the number of maxium open files (to be memory-mapped)
# ulimit -n 20000

# Load Anaconda environment
source /usr/workspace/ipe1/anaconda/bin/activate # Remember to change 'ipe1' to your own username
module load cuda/11.8.0
conda activate opence-1.9.1

####################### Distributed Setup #######################

# Record node hostnames
lrun -T1 hostname

# Get hostname of the rank-0 node
firsthost=$(lrun -N1 -n1 hostname)
echo "First host: $firsthost"

# Set MASTER_ADDR to hostname of first compute node in allocation
export MASTER_ADDR=$firsthost

# Set MASTER_PORT to any used port number
export MASTER_PORT=23456

####################### Launch Distributed Training #######################

lrun -T4 --gpubind=off python /p/gpfs1/ipe1/LLNLMLBackmapping/lit_ras/split_inputs.py
