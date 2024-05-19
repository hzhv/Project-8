import torch
import torch.nn as nn
import torch.nn.functional as F

class DLRMPrefetcher(nn.Module):
    def __init__(self, input_size, embed_dim, hidden_dim=512, n_heads=8, n_layers=2):
        super(DLRMPrefetcher, self).__init__()
        
        self.table_id_embed = nn.Embedding(input_size, embed_dim)
        self.idx_id_embed = nn.Embedding(input_size, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads)
        self.table_id_transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.idx_id_transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.linear = nn.Linear(embed_dim * 2, hidden_dim)
        self.table_output_layer = nn.Linear(hidden_dim, input_size)
        self.idx_output_layer = nn.Linear(hidden_dim, input_size)
        
    def forward(self, table_id_seq, idx_id_seq):
        # table_id_seq = table_id_seq.float()
        # idx_id_seq = idx_id_seq.float()
      
        table_id_seq = self.table_id_embed(table_id_seq)
        idx_id_seq = self.idx_id_embed(idx_id_seq)
        
        table_id_seq = table_id_seq.permute(1, 0, 2)  # (seq_length, batch_size, embed_dim)
        idx_id_seq = idx_id_seq.permute(1, 0, 2)      # (seq_length, batch_size, embed_dim)
        
        table_id_out = self.table_id_transformer(table_id_seq)
        idx_id_out = self.idx_id_transformer(idx_id_seq)
        
        table_id_out = table_id_out[-1]
        idx_id_out = idx_id_out[-1]
        
        h_combined = torch.cat((table_id_out, idx_id_out), dim=-1)
     
        hidden = F.relu(self.linear(h_combined))
 
        table_output = self.table_output_layer(hidden)
        idx_output = self.idx_output_layer(hidden)
     
        return table_output, idx_output

if __name__ == "__main__":
    model = DLRMPrefetcher(1630, embed_dim=256)
    table_id_seq = torch.randint(0, 10, (32, 10, 855))  # (batch_size, seq_length, table_id_size)
    idx_id_seq = torch.randint(0, 10, (32, 10, 65536))  # (batch_size, seq_length, idx_id_size)
    table_output, idx_output = model(table_id_seq, idx_id_seq)
