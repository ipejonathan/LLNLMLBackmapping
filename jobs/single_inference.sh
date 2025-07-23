#!/bin/bash
#BSUB -nnodes 1
#BSUB -W 00:10
#BSUB -G cancer
#BSUB -q pbatch
#BSUB -J infer_single
#BSUB -o infer_single-%J.log
#BSUB -alloc_flags ipisolate

source /usr/workspace/ipe1/anaconda/bin/activate
module load cuda/11.8.0
conda activate opence-1.9.1

UCG_FILE=$1
FNAME=$(basename "$UCG_FILE" .npz)
OUT_FILE="/p/gpfs1/ipe1/LLNLMLBackmapping/cg_outputs/${FNAME}_cg.npy"
SCRIPT="/p/gpfs1/ipe1/LLNLMLBackmapping/lit_ras/inference.py"
CKPT="/p/gpfs1/ipe1/LLNLMLBackmapping/lit_logs/ras-raf-test/version_4/checkpoints/epoch=1800-step=585325.ckpt"


####################### Distributed Setup #######################

# Record which nodes are used
lrun -T1 hostname

# Set MASTER_ADDR to the hostname of the first node
firsthost=$(lrun -N1 -n1 hostname)
echo "First host: $firsthost"

export MASTER_ADDR=$firsthost
export MASTER_PORT=23456 # Arbitrary unused port


echo "Running inference on: $UCG_FILE"

lrun -T4 --gpubind=off python "$SCRIPT" \
    --ucg-file "$UCG_FILE" \
    --out-dir "$OUT_FILE" \
    --cg-generator "$CKPT"
