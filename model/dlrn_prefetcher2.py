import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DLRMPrefetcher(nn.Module):
    def __init__(self, table_id_vocab, idx_id_vocab, embed_dim, output_length, block_size=2048, n_heads=8, n_layers=2):
        super(DLRMPrefetcher, self).__init__()
        self.table_id_vocab = table_id_vocab
        self.idx_id_vocab = idx_id_vocab
        self.embed_dim = embed_dim
        self.hidden_dim = 512
        self.output_length = output_length
        self.block_size = block_size
        self.n_idx_segments = idx_id_vocab // self.block_size + 1

        self.table_id_embed = nn.Embedding(table_id_vocab, embed_dim)
        self.idx_id_embed = nn.ModuleList([
            nn.Embedding(self.block_size, embed_dim) for _ in range(self.n_idx_segments)
        ])

        encoder = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads)
        self.table_id_transformer = nn.TransformerEncoder(encoder, num_layers=n_layers)
        self.idx_id_transformer = nn.TransformerEncoder(encoder, num_layers=n_layers)
        
        self.linear = nn.Linear(embed_dim * 2, self.hidden_dim)
        self.table_output_projection = nn.Linear(self.hidden_dim, table_id_vocab)
        self.idx_output_projection = nn.Linear(self.hidden_dim, idx_id_vocab)

        self.position_encoding = self.create_positional_encoding(output_length, self.hidden_dim)
        
    def create_positional_encoding(self, length, dimension):
        pe = torch.zeros(length, dimension)
        position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dimension, 2).float() * (-math.log(10000.0) / dimension))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension
        return pe
    
    def forward(self, table_id_seq, idx_id_seq):
        print("table_id and idx_id input shape: ", table_id_seq.shape, idx_id_seq.shape)
        
        table_id_embeds = self.table_id_embed(table_id_seq)  # (batch_size, seq_length, embed_dim)

        idx_id_embeds_tmp = []
        for seg in range(self.n_idx_segments):
            mask = (idx_id_seq >= seg * self.block_size) & (idx_id_seq < (seg + 1) * self.block_size)
            if mask.any():
                segment_indices = idx_id_seq[mask] - seg * self.block_size
                idx_id_embeds_tmp.append((mask, self.idx_id_embed[seg](segment_indices)))

        idx_id_embeds = torch.zeros(idx_id_seq.size(0), idx_id_seq.size(1), self.embed_dim, device=idx_id_seq.device)
        for mask, embed in idx_id_embeds_tmp:
            idx_id_embeds[mask] = embed

        table_id_embeds = table_id_embeds.permute(1, 0, 2)  # (seq_length, batch_size, embed_dim)
        idx_id_embeds = idx_id_embeds.permute(1, 0, 2)      # (seq_length, batch_size, embed_dim)
        
        print("Table ID Embeds shape:", table_id_embeds.shape)
        print("Idx ID Embeds shape:", idx_id_embeds.shape)

        table_id_ec = self.table_id_transformer(table_id_embeds) # encoded table_id sequence
        idx_id_ec = self.idx_id_transformer(idx_id_embeds)
        
        print("Encoded Table ID shape after Transformer:", table_id_ec.shape)
        print("Encoded Idx ID shape after Transformer:", idx_id_ec.shape)
        
        table_id_ec_mean = table_id_ec.mean(dim=0)  # (batch_size, embed_dim)
        idx_id_ec_mean = idx_id_ec.mean(dim=0)      # (batch_size, embed_dim)
        
        print("Mean Table ID shape:", table_id_ec_mean.shape)
        print("Mean Idx ID shape:", idx_id_ec_mean.shape)
        
        h_combined = torch.cat((table_id_ec_mean, idx_id_ec_mean), dim=-1)  # (batch_size, embed_dim * 2)
        print("Concatenated output shape (input to linear layer):", h_combined.shape)
        
        hidden = F.relu(self.linear(h_combined))  # (batch_size, hidden_dim)
        print("Output of linear layer shape:", hidden.shape)

        table_output_list = []
        idx_output_list = []
        positional_encoding = self.position_encoding.to(hidden.device)
        for t in range(self.output_length):
            hidden_with_pos = hidden + positional_encoding[:, t, :]
            table_output = self.table_output_projection(hidden_with_pos)
            idx_output = self.idx_output_projection(hidden_with_pos)
            table_output_list.append(table_output)
            idx_output_list.append(idx_output)

        table_outputs = torch.stack(table_output_list, dim=1)  # (batch_size, output_length, table_id_vocab)
        idx_outputs = torch.stack(idx_output_list, dim=1)      # (batch_size, output_length, idx_id_vocab)
        
        table_outputs = table_outputs.view(-1, self.table_id_vocab)
        idx_outputs = idx_outputs.view(-1, self.idx_id_vocab)
        return table_outputs, idx_outputs