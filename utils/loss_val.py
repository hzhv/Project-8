import torch
import numpy


def calculate_pair_match(predicted, ground_truth):
    """
    Calculate the number of pairs in the predicted sequence 
    that appear in the ground truth sequence.
    """
    print("\nPAIR MATCH CALCULATION")
    # print("predicted shape:", predicted.shape)
    # print("ground_truth shape:", ground_truth.shape)
    print(predicted)
    print("\n######################################################################################################################\n")
    gt_set = {tuple(gt_pair.tolist()) for gt_pair in ground_truth}
    total_count = len(predicted)
    match_count = 0
    for pair in predicted:
        if tuple(pair.tolist()) in gt_set:
            match_count += 1

    print("Match count:", match_count)
    print("Total count:", total_count, "\n")

    accuracy = match_count / total_count if total_count > 0 else 0.0
    return accuracy