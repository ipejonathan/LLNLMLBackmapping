import os
import glob
import numpy as np

input_dir = "/p/gpfs1/splash/hmc_project/ucg_npz_data_ucg_40site_aligned_to_gdom_and_crd_membrane_alignment/"
output_dir = "/p/gpfs1/ipe1/LLNLMLBackmapping/inference_chunks/"
os.makedirs(output_dir, exist_ok=True)

all_files = sorted(glob.glob(os.path.join(input_dir, "pfpatch_*_ucg.npz")))
num_chunks = 8  # Adjust based on available nodes
chunks = np.array_split(all_files, num_chunks)

for i, chunk in enumerate(chunks):
    with open(os.path.join(output_dir, f"chunk_{i}.txt"), "w") as f:
        f.write("\n".join(chunk))
