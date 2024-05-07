import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1)]

class PrefetcherTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, 
                 dim_feedforward, max_seq_length, num_classes_table, num_classes_idx):
        super(PrefetcherTransformer, self).__init__()
        self.embedding = nn.Embedding(num_classes_table + num_classes_idx, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length)
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward)
        self.decoder_table = nn.Linear(d_model, num_classes_table)
        self.decoder_idx = nn.Linear(d_model, num_classes_idx)
        self.d_model = d_model

    def forward(self, src, tgt):
        src = self.embedding(src) * torch.sqrt(torch.tensor(self.d_model))
        tgt = self.embedding(tgt) * torch.sqrt(torch.tensor(self.d_model))
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)
        output = self.transformer(src, tgt)
        output_table = self.decoder_table(output)
        output_idx = self.decoder_idx(output)
        return output_table, output_idx

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

