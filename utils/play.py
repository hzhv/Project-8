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
# def calculate_pair_match(predicted, ground_truth):
#     """
#     Calculate the number of pairs in the predicted sequence 
#     that appear in the ground truth sequence.
#     """
#     print("\nPAIR MATCH CALCULATION")
#     print("predicted shape:", predicted.shape)
#     print("ground_truth shape:", ground_truth.shape)

#     match_count = 0
#     total_count = len(predicted)

    
#     for pair in predicted:
#         print(pair, pair.shape)
#         # if tuple(pair.tolist()) in [tuple(gt_pair.tolist()) for gt_pair in ground_truth]:
#         #     match_count += 1

#     print("Match count:", match_count)
#     print("Total count:", total_count, "\n")

#     accuracy = match_count / total_count if total_count > 0 else 0.0
#     return accuracy

# def test_calculate_pair_match():
#     predicted = torch.tensor([
#         [100, 200],
#         [300, 400],
#         [500, 600],
#         [700, 800]
#     ])
#     ground_truth = torch.tensor([
#         [100, 200],
#         [300, 400],
#         [900, 1000],
#         [500, 600],
#         [700, 800],
#         [1100, 1200]
#     ])
#     expected_accuracy = 1.0  
#     accuracy = calculate_pair_match(predicted, ground_truth)
#     assert accuracy == expected_accuracy, f"Expected {expected_accuracy}, but got {accuracy}"
#     print(f"Test passed. Accuracy: {accuracy}")


import torch
import torch.nn as nn
import torch.optim as optim

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim, n_layers, n_heads, max_seq_len):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_dim)
        encoder_layers = nn.TransformerEncoderLayer(hidden_dim, n_heads)
        self.encoder = nn.TransformerEncoder(encoder_layers, n_layers)
        self.linear = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        seq_len, batch_size = x.size()
        pos = torch.arange(0, seq_len).unsqueeze(1).expand(seq_len, batch_size).to(x.device)
        x = self.embedding(x) + self.pos_embedding(pos)
        x = self.encoder(x)
        x = self.linear(x)
        return x

# 假设词汇表大小为5000，隐藏单元数为512，6层编码器，8头注意力机制，最大序列长度为2400
vocab_size = 5000
hidden_dim = 512
n_layers = 6
n_heads = 8
max_seq_len = 2400
model = TransformerEncoder(vocab_size, hidden_dim, n_layers, n_heads, max_seq_len)

# 设置模型到GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 假设输入序列已经是词索引，长度为2000
input_ids = torch.randint(0, vocab_size, (2000, 1)).to(device)

# 模型预测
model.eval()
predicted_ids = input_ids.clone()

with torch.no_grad():
    for _ in range(400):
        outputs = model(predicted_ids)
        next_token_logits = outputs[-1, 0, :]  # 获取最后一个时间步的预测结果
        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0).unsqueeze(0)
        predicted_ids = torch.cat([predicted_ids, next_token_id], dim=0)  # 将新预测的词加入输入序列

# 假设有一个词汇表将索引转换回单词
# 这里只是示例，实际需要根据具体词汇表进行转换
vocab = {i: f'word{i}' for i in range(vocab_size)}
predicted_text = ' '.join([vocab[idx.item()] for idx in predicted_ids.squeeze()])

print(predicted_text)
