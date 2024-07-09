import torch
import torch.nn as nn
import torch.nn.functional as F


class NegativeSamplingLoss(nn.Module):
    def __init__(self, num_neg_samples):
        super(NegativeSamplingLoss, self).__init__()
        self.num_neg_samples = num_neg_samples

    def forward(self, pos_output, pos_target, neg_output, neg_targets):
        """
        pos_output: (batch_size, seq_len, embed_dim)
        pos_target: (batch_size, seq_len, embed_dim)
        """
        print("\nDEBUGGING NEGATIVE SAMPLING LOSS")
        print("pos_output shape:", pos_output.shape)
        print("pos_target shape:", pos_target.shape)
        print("neg_output shape:", neg_output.shape)
        print("neg_targets shape:", neg_targets.shape)
        
        pos_loss = F.logsigmoid(torch.sum(pos_output * pos_target, dim=-1))
        print("Positive loss shape",pos_loss.shape)
        neg_loss = F.logsigmoid(-torch.sum(neg_output * neg_targets, dim=-1))
        neg_loss = neg_loss.mean(dim=-1)
        print("Negative loss shape",neg_loss.shape)
        return - (pos_loss + neg_loss).mean()




def calculate_pair_match(predicted, ground_truth):
    """
    Calculate the number of pairs in the predicted sequence 
    that appear in the ground truth sequence.
    """
    print("\nPAIR MATCH CALCULATION")
    print("predicted shape:", predicted.shape)
    print("ground_truth shape:", ground_truth.shape)
    print(predicted)
    print("######################################################################################################################\n")
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