import gzip

import numpy as np
import click

import torch


@click.command()
@click.option(
    "--file",
    type=str,
    default="/home/hli31/S2024_MLSYS/dlrm_datasets/fbgemm_t856_bs65536_0.pt",
    help="Embedding bag data file",
)
@click.option(
    "--factor",
    type=float,
    default=1.0,
    help="Random sampling the indices",
)
def preprocess(file, factor):
    if file.endswith(".gz"):
        with gzip.open(file) as _:
            indices, offsets, lengths = torch.load(_)
    else:
        indices, offsets, lengths = torch.load(file)

    print(f"indices shape: {indices.shape}")
    print(f"Offsets shape:, {offsets.shape}")
    print(f"Lengths shape:, {lengths.shape}")

    # items = np.unique(indices)
    # print(f"Number of unique indices: {len(items)}")
    # print(f"Number of unique offsets: {len(np.unique(offsets))}")
    # print(f"Number of unique lengths: {len(np.unique(lengths))}")

    if factor < 1:
        items = np.random.choice(indices, int(len(items) * factor), replace=False)

        print(f"Number of unique indices after random sampling: {len(items)}")

        indices = file[0:file.rfind(".pt")] + "_cached.csv"

        np.savetxt(indices, items.reshape(1, -1), delimiter=",", fmt="%d")

if __name__ == "__main__":
    preprocess()