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

while IFS= read -r ucg_file; do
    fname=$(basename "$ucg_file" .npz)
    out_file="${OUTPUT_DIR}/${fname}_cg.npy"
    echo "Processing $ucg_file"
    lrun -T4 --gpubind=off python "$SCRIPT" \
        --ucg-file "$ucg_file" \
        --out-file "$out_file" \
        --cg-generator "$CG_GENERATOR"
done < "$CHUNK_FILE"
