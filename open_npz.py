from numpy import load
import numpy as np

data = load('./sample-data/cg/all_indices_per_cluster.npz', allow_pickle=True)
# data = load("/Users/jonathan/Documents/LLNLMLBackmapping/sample-data/ucg/pfpatch_000000000138_ucg.npz", allow_pickle=True)
lst = data.files
print(lst)
# print(data["aligned_anchors_BB_index"])
print(data["indices_per_cluster"].shape)
print(data['indices_per_cluster'])
# print(data["positions_ucg"][0])
sum = 0
max = 0
for item in data['indices_per_cluster']:
    if len(item) > max:
        max = len(item)
    sum += len(item)
print(max)
print(sum)


# print(data[lst[0]].shape)
# for item in lst:
#     print(item)
#     print(data[item])