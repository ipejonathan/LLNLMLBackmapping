from numpy import load
import numpy as np

# data = load('./sample-data/cg/all_indices_per_cluster.npz', allow_pickle=True)
data = load("/Users/jonathan/Documents/LLNLMLBackmapping/sample-data/ucg/pfpatch_000000000138_ucg.npz", allow_pickle=True)
lst = data.files
print(lst)
# print(data["aligned_anchors_BB_index"])
# print(len(data["indices_per_cluster"]))
print(data["positions_ucg"][0])



# print(data[lst[0]].shape)
# for item in lst:
#     print(item)
#     print(data[item])