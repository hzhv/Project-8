import os
import gzip
import typing


import numpy as np
import click

import torch


# @click.command()
# @click.option(
#     "--file",
#     type=str,
#     default="/home/hli31/S2024_MLSYS/dlrm_datasets/fbgemm_t856_bs65536_0.pt",
#     help="Embedding bag data file",
# )
# @click.option(
#     "--scope",
#     nargs=2,
#     type=int,
#     default=(0, 0),
#     help="the Trace range, e.g. (1, 856)",
# )
# @click.option(
#     "--bs",
#     nargs=2,
#     type=int,
#     default=(0, 0),
#     help="the query range, e.g. (1, 65536)",
# )
# def main(file, scope, bs):
#     getDataset(file)
    
    
# def getDataset(file, factor=1):
#     try:
#         with gzip.open(file) as f:
#             indices, offsets, lengths = torch.load(f)
#     except:
#         indices, offsets, lengths = torch.load(file)
#     print(f"indices shape: {indices.shape}")
#     print(f"Offsets shape:, {offsets.shape}")
#     print(f"Lengths shape:, {lengths.shape}")
#     print()
    
#     print(f"Indices range: min {indices.min()}, max {indices.max()}")
#     print(f"Lengths range: min {lengths.min()}, max {lengths.max()}")

#     assert not torch.isinf(indices).any(), "Infinity found in indices"
#     assert not torch.isinf(lengths).any(), "Infinity found in lengths"
def calculate_pair_match(predicted, ground_truth):
    """
    Calculate the number of pairs in the predicted sequence 
    that appear in the ground truth sequence.
    """
    print("\nPAIR MATCH CALCULATION")
    print("predicted shape:", predicted.shape)
    print("ground_truth shape:", ground_truth.shape)

    match_count = 0
    total_count = len(predicted)

    
    for pair in predicted:
        print(pair, pair.shape)
        # if tuple(pair.tolist()) in [tuple(gt_pair.tolist()) for gt_pair in ground_truth]:
        #     match_count += 1

    print("Match count:", match_count)
    print("Total count:", total_count, "\n")

    accuracy = match_count / total_count if total_count > 0 else 0.0
    return accuracy

# 测试用例
def test_calculate_pair_match():
    predicted = torch.tensor([
        [100, 200],
        [300, 400],
        [500, 600],
        [700, 800]
    ])
    ground_truth = torch.tensor([
        [100, 200],
        [300, 400],
        [900, 1000],
        [500, 600],
        [700, 800],
        [1100, 1200]
    ])
    expected_accuracy = 1.0  # 因为所有的预测对都在 ground truth 中出现过
    accuracy = calculate_pair_match(predicted, ground_truth)
    assert accuracy == expected_accuracy, f"Expected {expected_accuracy}, but got {accuracy}"
    print(f"Test passed. Accuracy: {accuracy}")

if __name__ == "__main__":
    test_calculate_pair_match()
