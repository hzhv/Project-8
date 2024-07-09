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
from torch.utils.data import DataLoader, Dataset

class CustomDataset(Dataset):
    def __init__(self, input_sequences, target_sequences, max_len):
        self.input_sequences = input_sequences
        self.target_sequences = target_sequences
        self.max_len = max_len

    def __len__(self):
        return len(self.input_sequences)

    def __getitem__(self, idx):
        input_seq = self.tokenizer.encode(self.input_sequences[idx], max_length=self.max_len, padding='max_length', truncation=True)
        target_seq = self.tokenizer.encode(self.target_sequences[idx], max_length=self.max_len, padding='max_length', truncation=True)
        return torch.tensor(input_seq), torch.tensor(target_seq)

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, dim_feedforward, max_len, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position_encoding = nn.Parameter(torch.zeros(1, max_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        seq_len = x.size(1)
        x = self.embedding(x) + self.position_encoding[:, :seq_len, :]
        x = self.transformer_encoder(x)
        x = self.fc(x)
        return x

# Hyperparameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vocab_size = 30522  # 词汇表大小，取决于tokenizer
d_model = 32  # 词嵌入维度
nhead = 8  # 注意力头数
num_encoder_layers = 6  # 编码器层数
dim_feedforward = 2048  # 前馈神经网络维度
max_len = 2400  # 序列最大长度
dropout = 0.1  # Dropout概率
lr = 1e-4  # 学习率
batch_size = 4  # 批次大小
epochs = 3 # 训练轮数

# 准备数据集
# tokenizer = ...  # 初始化你的tokenizer
input_sequences = torch.rand(0, 20, [100])  # 你的输入序列列表
target_sequences = torch.rand(0, 20, [100])  # 你的目标序列列表
dataset = CustomDataset(input_sequences, target_sequences,  max_len)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 初始化模型
model = TransformerEncoder(vocab_size, d_model, nhead, num_encoder_layers, dim_feedforward, max_len, dropout).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# 训练模型
for epoch in range(epochs):
    model.train()
    for input_seq, target_seq in dataloader:
        input_seq, target_seq = input_seq.to(device), target_seq.to(device)
        optimizer.zero_grad()
        output = model(input_seq)
        loss = criterion(output.view(-1, vocab_size), target_seq.view(-1))
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

# 模型预测
model.eval()

with torch.no_grad():
    input_text = "your input sequence of 2000 words here"
    input_ids = tokenizer.encode(input_text, max_length=2000, padding='max_length', truncation=True)
    input_seq = torch.tensor(input_ids).unsqueeze(0).to(device)

    generated_seq = input_seq
    for _ in range(400):  # 生成400个词
        output = model(generated_seq)
        next_token = torch.argmax(output[:, -1, :], dim=-1).unsqueeze(0)
        generated_seq = torch.cat((generated_seq, next_token), dim=1)
        if generated_seq.size(1) > 2400:
            generated_seq = generated_seq[:, 1:]  # 确保序列长度不超过2400

    predicted_seq = generated_seq.squeeze(0).tolist()
    predicted_text = tokenizer.decode(predicted_seq, skip_special_tokens=True)
    print(predicted_text)

