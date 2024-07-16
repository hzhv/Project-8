import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DLRMPrefetcher(nn.Module):
    def __init__(self, table_id_vocab, idx_id_vocab, table_embed_dim, idx_embed_dim, output_length=1, n_heads=8, n_layers=6):
        super(DLRMPrefetcher, self).__init__()
        self.table_id_vocab = table_id_vocab
        self.idx_id_vocab = idx_id_vocab
        self.table_embed_dim = table_embed_dim
        self.idx_embed_dim = idx_embed_dim
        self.hidden_dim = 512
        self.output_length = output_length

        self.table_id_embed = nn.Embedding(table_id_vocab, self.table_embed_dim)
        self.idx_id_embed = nn.Embedding(idx_vocab_size, self.idx_embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, dim_feedforward=hidden_dim)
        self.table_id_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.idx_id_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # self.linear = nn.Linear(embed_dim * 2, self.hidden_dim)
        # self.table_output_projection = nn.Linear(self.hidden_dim, table_id_vocab)
        # self.idx_output_projection = nn.Linear(self.hidden_dim, idx_id_vocab)

        self.linear = nn.Linear(embed_dim * 2, self.hidden_dim)
        self.table_output_layer = nn.Linear(self.hidden_dim, table_id_vocab)
        self.idx_output_layer = nn.Linear(self.hidden_dim, idx_id_vocab)

    def forward(self, table_seq, idx_seq):
        table_emb = self.table_embedding(table_seq)
        idx_emb = self.idx_embedding(idx_seq)
        print("Table ID Embeds shape:", table_emb.shape)
        print("Idx ID Embeds shape:", idx_emb.shape)

        table_id_ec = self.table_id_encoder(table_emb)
        idx_id_ec = self.idx_id_encoder(idx_emb)
        print("Encoded Table ID shape after Transformer:", table_id_ec.shape)
        print("Encoded Idx ID shape after Transformer:", idx_id_ec.shape)
        
        table_id_ec_last = table_id_ec[-1]  # pooling by last time step 
        idx_id_ec_last = idx_id_ec[-1]      # (batch_size, seq_length, embed_dim)
        print("Last Table ID shape:", table_id_ec_last.shape)
        print("Last Idx ID shape:", idx_id_ec_last.shape)

        combined = torch.cat((table_id_ec_last, idx_id_ec_last), dim=-1)
        combined = combined.permute(1, 0, 2)  
        print("Concatenated output shape (input to linear layer):", combined.shape)

        hidden = F.relu(self.linear(combined)) # (output_lenth, batch_size, hidden_dim) 
        print("Linear layer shape:", hidden.shape)

        table_out = self.fc_table(transformer_out)
        idx_out = self.fc_idx(transformer_out)
        return table_out, idx_out
