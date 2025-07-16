for i in {0..7}; do
    bsub < run_chunk.sh $i
done
