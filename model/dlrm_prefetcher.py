import torch
import torch.nn as nn
import torch.nn.functional as F


### Hierarchical Embedding Method
class DLRMPrefetcher(nn.Module):
    def __init__(self, table_id_vocab, idx_id_vocab, embed_dim, output_length, block_size=2048, n_heads=8, n_layers=2):
        f'''
        Segement embedding layer into n_parts partitions for large vocab size, i.e., idx_id

        '''
        super(DLRMPrefetcher, self).__init__()
        self.table_id_vocab = table_id_vocab
        self.idx_id_vocab = idx_id_vocab
        self.embed_dim = embed_dim
        self.hidden_dim = embed_dim * 4
        self.output_length = output_length
        self.block_size = block_size
        self.n_idx_segments = idx_id_vocab // self.block_size + 1
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # print("table_id_vocab:", table_id_vocab, "idx_id_vocab", idx_id_vocab,"output length:", output_length, "n_idx_segments:", self.n_idx_segments)
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        
        self.table_id_embed = nn.Embedding(table_id_vocab, embed_dim)
        self.idx_id_embed = nn.ModuleList([
            nn.Embedding(self.block_size, embed_dim) for _ in range(self.n_idx_segments)
        ])

        encoder = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads)
        self.table_id_transformer = nn.TransformerEncoder(encoder, num_layers=n_layers)
        self.idx_id_transformer = nn.TransformerEncoder(encoder, num_layers=n_layers)
        
        self.linear = nn.Linear(embed_dim * 2, self.hidden_dim)
        # self.table_output_layer = nn.Linear(self.hidden_dim, table_id_vocab * output_length)
        # self.idx_output_layer = nn.Linear(self.hidden_dim, idx_id_vocab * output_length)
        self.table_output_projection = nn.Linear(self.hidden_dim + 1, table_id_vocab)
        self.idx_output_projection = nn.Linear(self.hidden_dim + 1, idx_id_vocab)
        
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
        for t in range(self.output_length):
            time_step_info = torch.full((hidden.size(0), 1), t, device=hidden.device).float()  # (batch_size, 1)
            hidden_with_time = torch.cat((hidden, time_step_info), dim=-1)  # (batch_size, hidden_dim + 1)
            table_output = self.table_output_projection(hidden_with_time)
            
            idx_output = self.idx_output_projection(hidden_with_time)
            table_output_list.append(table_output)
            idx_output_list.append(idx_output)

        table_outputs = torch.stack(table_output_list, dim=1)  # (batch_size, output_length, table_id_vocab)
        idx_outputs = torch.stack(idx_output_list, dim=1)      # (batch_size, output_length, idx_id_vocab)
        
        table_outputs = table_outputs.view(-1, self.table_id_vocab)
        idx_outputs = idx_outputs.view(-1, self.idx_id_vocab)
        return table_outputs, idx_outputs
    
    def forward_bug(self, table_id_seq, idx_id_seq):
        print("table_id and idx_id input shape: ",table_id_seq.shape, idx_id_seq.shape)
        
        # # project table_id_seq to [0, table_id_vocab)
        # unique_table_ids = torch.unique(table_id_seq)
        # table_id_map = {id.item(): idx for idx, id in enumerate(unique_table_ids)}
        # table_id_seq_mapped = torch.tensor([table_id_map[id.item()] for id in table_id_seq.view(-1)]).view(table_id_seq.size()).to(table_id_seq.device)
        
        # # project idx_id_seq to [0, idx_id_vocab)
        # unique_idx_ids = torch.unique(idx_id_seq)
        # idx_id_map = {id.item(): idx for idx, id in enumerate(unique_idx_ids)}
        # idx_id_seq_mapped = torch.tensor([idx_id_map[id.item()] for id in idx_id_seq.view(-1)]).view(idx_id_seq.size()).to(idx_id_seq.device)

        table_id_embeds = self.table_id_embed(table_id_seq)  # (batch_size, seq_length, embed_dim)

        idx_id_embeds_tmp = []
        for seg in range(self.n_idx_segments):
            mask = (idx_id_seq >= seg * self.block_size) & (idx_id_seq < (seg + 1) * self.block_size)
            if mask.any():
                segment_indices = idx_id_seq[mask] - seg * self.block_size
                idx_id_embeds_tmp.append((mask, self.idx_id_embed[seg](segment_indices)))

        # Combine the embedded segments into a full idx_id embedding
        idx_id_embeds = torch.zeros(idx_id_seq.size(0), idx_id_seq.size(1), self.embed_dim, device=idx_id_seq.device)
        for mask, embed in idx_id_embeds_tmp:
            idx_id_embeds[mask] = embed

        print("##################Debugging##################")
        print("embedding shape:", table_id_embeds.shape, idx_id_embeds.shape)
        table_id_embeds = table_id_embeds.permute(1, 0, 2)  # (seq_length, batch_size, embed_dim)
        idx_id_embeds = idx_id_embeds.permute(1, 0, 2)      # (seq_length, batch_size, embed_dim)
        
        table_id_ec = self.table_id_transformer(table_id_embeds) # encoded table_id sequence
        idx_id_ec = self.idx_id_transformer(idx_id_embeds)
        print("encoder shape after Transformer:", table_id_ec.shape, idx_id_ec.shape)
        # table_id_ec = table_id_ec[-1] # use last time step represent the encoded result of the entire sequence
        # idx_id_ec = idx_id_ec[-1]
        table_id_ec = table_id_ec.mean(dim=0)
        idx_id_ec = idx_id_ec.mean(dim=0)
        
        print(table_id_ec.shape, idx_id_ec.shape)
        print("encoder shape mean:", table_id_ec.shape, idx_id_ec.shape)
        
        h_combined = torch.cat((table_id_ec, idx_id_ec), dim=-1)
        
        print("h_combined shape:", h_combined.shape)
        hidden = F.relu(self.linear(h_combined))
        print("hidden shape:", hidden.shape)
        
        # table_output = self.table_output_layer(hidden).view(-1, self.output_length, self.table_id_vocab)
        # idx_output = self.idx_output_layer(hidden).view(-1, self.output_length, self.idx_id_vocab)
        table_output_list = []
        idx_output_list = []
        for _ in range(self.output_length):
            table_output = self.table_output_projection(hidden)
            idx_output = self.idx_output_projection(hidden)
            table_output_list.append(table_output)
            idx_output_list.append(idx_output)

        table_outputs = torch.stack(table_output_list, dim=1)  # (batch_size, output_length, table_id_vocab)
        idx_outputs = torch.stack(idx_output_list, dim=1)
        print("output shape before flatten:", table_outputs.shape, idx_outputs.shape)
        
        table_outputs = table_outputs.view(-1, self.table_id_vocab)
        idx_outputs = idx_outputs.view(-1, self.idx_id_vocab)
        return table_outputs, idx_outputs
    

def neg_sampling_loss(positive_samples, negative_samples, model, device):
    """
    Compute the negative sampling loss.
    
    Args:
    - positive_samples (torch.Tensor): Embeddings for the positive samples.
    - negative_samples (torch.Tensor): Embeddings for the negative samples.
    - model (nn.Module): The model to be trained.
    - device (torch.device): Device to run the computations.
    
    Returns:
    - loss (torch.Tensor): Negative sampling loss.
    """

    pos_score = torch.bmm(positive_samples.unsqueeze(1), model.idx_output_projection.weight[positive_samples.long()].unsqueeze(2)).squeeze()   
    neg_score = torch.bmm(negative_samples, model.idx_output_projection.weight[negative_samples.long()].unsqueeze(2)).squeeze()
    
    pos_loss = -F.logsigmoid(pos_score)
    neg_loss = -F.logsigmoid(-neg_score)
    
    loss = pos_loss + neg_loss
    return loss.mean()










if __name__ == "__main__":
    table_id_seq = torch.randint(0, 10, (32, 2048)).cuda()  # (batch_size, seq_length, table_id_size)
    idx_id_seq = torch.randint(0, 10, (32, 2048)).cuda()  # (batch_size, seq_length, idx_id_size)
    print(table_id_seq.shape, idx_id_seq.shape)
    model = DLRMPrefetcher(855, 9000000, 128, output_length=409, block_size=2048, n_heads=8, n_layers=2).cuda()
    table_out, idx_out = model(table_id_seq, idx_id_seq)
    print("________________Output shape________________:")
    print(table_out.shape, idx_out.shape)

