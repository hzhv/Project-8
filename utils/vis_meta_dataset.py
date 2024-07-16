import os.path as osp

import torch
import matplotlib.pyplot as plt


dlrm_dataset = "/home/hli31/S2024_MLSYS/dlrm_datasets/2021/fbgemm_t856_bs65536_0.pt"
name = osp.basename(dlrm_dataset).split('.')[0]

indices, offsets, lengths = torch.load(dlrm_dataset)
lengths_flat = lengths.numpy().flatten()

plt.figure(figsize=(12, 6))
plt.hist(lengths_flat, bins='auto', alpha=0.5)
plt.xlabel('Pooling Factors')
plt.ylabel('Frequency')
plt.title(f"{name} Lengths Distribution")

plt.savefig(f'{name}_length_density_dist.png')
print(f"indices shape: {indices.shape}")
print(f"Offsets shape:, {offsets.shape}")
print(f"Lengths shape:, {lengths.shape}\n")

