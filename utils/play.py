import torch


file = "/home/hli31/S2024_MLSYS/dlrm_datasets/fbgemm_t856_bs65536_0.pt"
indices, offsets, lengths = torch.load(file)

print(indices.shape)
print(lengths[0].shape)
sum = 0
for i in range(len(lengths[0])):
    sum += lengths[0][i]

print(sum)