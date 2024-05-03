import gzip
import typing

import torch


file = "/home/hli31/S2024_MLSYS/dlrm_datasets/fbgemm_t856_bs65536_0.pt"

try:
    with gzip.open(file) as f:
        indices, offsets, lengths = torch.load(f)
except:
    indices, offsets, lengths = torch.load(file)
    print(lengths[1, :])
    print(type(indices[1]))
    print("yeah!!!")

t = torch.load("demo_traces.pt")

print(t[0])

if __name__ == "__main__":
    # getQueryID(file, (1, 999))
    print("Hello, world!")