CHUNK_ID=$1  # Passed as an argument
CHUNK_FILE="/p/gpfs1/ipe1/LLNLMLBackmapping/inference_chunks/chunk_${CHUNK_ID}.txt"


####################### Launch Multi-GPU Inference #######################

while IFS= read -r ucg_file; do
    # fname=$(basename "$ucg_file" .npz)
    # out_file="${OUTPUT_DIR}/${fname}_cg.npy"
    # echo "Processing $ucg_file"

    bsub < single_inference.sh "$ucg_file"

    status=$?
    if [[ $status -ne 0 ]]; then
        echo "Failed: $ucg_file"
    else
        echo "Done: $ucg_file"
    fi

    echo

done < "$CHUNK_FILE"
