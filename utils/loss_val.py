import torch
import torch.nn as nn
import torch.nn.functional as F

class JaccardLoss(nn.Module):
    def __init__(self):
        super(JaccardLoss, self).__init__()
    
    def forward(self, predicted, ground_truth):
        predicted_set = set(predicted)
        ground_truth_set = set(ground_truth)
        intersection = predicted_set.intersection(ground_truth_set)
        union = predicted_set.union(ground_truth_set)
        jaccard_index = len(intersection) / len(union)
        loss = 1 - jaccard_index
        return loss

def calculate_pair_match(predicted, ground_truth):
    """
    Calculate the number of pairs in the predicted sequence 
    that appear in the ground truth sequence.
    """
    print("\nPAIR MATCH CALCULATION")
    print("predicted shape:", predicted.shape)
    print("ground_truth shape:", ground_truth.shape)
    print("######################################################################################################################\n")
    # Convert each pair to tuple to make it hashable
    gt_set = {tuple(pair) for pair in ground_truth.reshape(-1, ground_truth.shape[-1]).tolist()}
    total_count = predicted.shape[0] * predicted.shape[1]
    match_count = 0

    # Compare each pair in predicted with the ground truth set
    for pair in predicted.reshape(-1, predicted.shape[-1]).tolist():
        if tuple(pair) in gt_set:
            match_count += 1

    print("Match count:", match_count)
    print("Total count:", total_count, "\n")

    accuracy = match_count / total_count if total_count > 0 else 0.0
    return accuracy


# class NegativeSamplingLoss(nn.Module):
#     def __init__(self, num_neg_samples):
#         super(NegativeSamplingLoss, self).__init__()
#         self.num_neg_samples = num_neg_samples

#     def forward(self, pos_output, pos_target, neg_output, neg_targets):
#         """
#         pos_output: (batch_size, seq_len, embed_dim)
#         pos_target: (batch_size, seq_len, embed_dim)
#         """
#         print("\nDEBUGGING NEGATIVE SAMPLING LOSS")
#         print("pos_output shape:", pos_output.shape)
#         print("pos_target shape:", pos_target.shape)
#         print("neg_output shape:", neg_output.shape)
#         print("neg_targets shape:", neg_targets.shape)
        
#         pos_loss = F.logsigmoid(torch.sum(pos_output * pos_target, dim=-1))
#         print("Positive loss shape",pos_loss.shape)
#         neg_loss = F.logsigmoid(-torch.sum(neg_output * neg_targets, dim=-1))
#         neg_loss = neg_loss.mean(dim=-1)
#         print("Negative loss shape",neg_loss.shape)
#         return - (pos_loss + neg_loss).mean()




