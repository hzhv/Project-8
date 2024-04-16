import gzip

import numpy as np
import click

import torch


@click.command()
@click.option(
    "--file",
    type=str,
    default="../dlrm_datasets/embedding_bag/2021/fbgemm_t856_bs65536.pt.gz",
    help="Embedding bag data file",
)
@click.option(
    "--factor",
    type=float,
    default=1.0,
    help="Random sampling the indices",
)
def preprocess(file, factor):
    with gzip.open(file) as f:
        indices, offsets, lengths = torch.load(f)

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