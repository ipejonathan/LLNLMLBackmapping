#!/bin/bash
#BSUB -W 12:00
#BSUB -G cancer
#BSUB -q pbatch
#BSUB -nnodes 1
#BSUB -J inf_chunk
#BSUB -o inf_chunk-%J.log
#BSUB -alloc_flags ipisolate

CHUNK_ID=$1  # Passed as an argument
CHUNK_FILE="/p/gpfs1/ipe1/LLNLMLBackmapping/inference_chunks/chunk_${CHUNK_ID}.txt"
CG_GENERATOR="/p/gpfs1/ipe1/LLNLMLBackmapping/lit_logs/ras-raf-test/version_4/checkpoints/epoch=1800-step=585325.ckpt"
SCRIPT="/p/gpfs1/ipe1/LLNLMLBackmapping/lit_ras/inference.py"
OUTPUT_DIR="/p/gpfs1/ipe1/LLNLMLBackmapping/cg_outputs"

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

while IFS= read -r ucg_file || [[ -n "$ucg_file" ]]; do
    fname=$(basename "$ucg_file" .npz)
    out_file="${OUTPUT_DIR}/${fname}_cg.npy"
    echo "Processing $ucg_file"

    lrun -T4 --gpubind=off python "$SCRIPT" \
        --ucg-file "$ucg_file" \
        --out-dir "$out_file" \
        --cg-generator "$CG_GENERATOR"

    if [[ $? -ne 0 ]]; then
        echo "❌ Error processing $ucg_file" >&2
    else
        echo "✅ Successfully processed $ucg_file"
    fi

    echo
    
done < "$CHUNK_FILE"
