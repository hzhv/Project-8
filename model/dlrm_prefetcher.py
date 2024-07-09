import torch
import torch.nn as nn
import torch.nn.functional as F


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
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # print("table_id_vocab:", table_id_vocab, "idx_id_vocab", idx_id_vocab,"output length:", output_length, "n_idx_segments:", self.n_idx_segments)
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        self.table_id_embed = nn.Embedding(table_id_vocab, embed_dim)
        self.idx_id_embed = nn.Embedding(100000, embed_dim)

        encoder = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads)
        self.table_id_transformer = nn.TransformerEncoder(encoder, num_layers=n_layers)
        self.idx_id_transformer = nn.TransformerEncoder(encoder, num_layers=n_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim * 2, nhead=n_heads)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        self.linear = nn.Linear(embed_dim * 2, self.hidden_dim)
        self.table_output_layer = nn.Linear(self.hidden_dim, table_id_vocab)
        self.idx_output_layer = nn.Linear(self.hidden_dim, idx_id_vocab)
        # self.table_output_projection = nn.Linear(self.hidden_dim, table_id_vocab)
        # self.idx_output_projection = nn.Linear(self.hidden_dim, idx_id_vocab)
        
    def forward(self, table_id_seq, idx_id_seq, tgt_table_seq, tgt_idx_seq):
        print("table_id and idx_id input shape: ", table_id_seq.shape, idx_id_seq.shape)
        
        table_id_embeds = self.table_id_embed(table_id_seq)  # (batch_size, seq_length, embed_dim)
        idx_id_embeds = self.idx_id_embed(idx_id_seq)

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

        
        ## input_embeds, input to the decoder， 
        combined_memory = torch.cat((table_id_ec_mean, idx_id_ec_mean), dim=-1) # (batch_size, embed_dim * 2)
        memory = combined_memory.unsqueeze(0).expand(tgt_table_seq.size(1), -1, -1)  # (seq_len, batch_size, embed_dim * 2)
        

        tgt_table_embeds = self.table_id_embed(tgt_table_seq).permute(1, 0, 2)
        tgt_idx_embeds = self.idx_id_embed(tgt_idx_seq).permute(1, 0, 2)
        print(tgt_table_seq.shape, tgt_idx_seq.shape)
        print(tgt_table_embeds.shape, tgt_idx_embeds.shape, "\n")
        tgt_embeds = torch.cat((tgt_table_embeds, tgt_idx_embeds), dim=-1) # (batch_size, embed_dim * 2)
        
        print("Concatenated output shape (input to linear layer):", tgt_embeds.shape)
        print("\nmemory shape:", combined_memory.shape, memory.shape, "\n")
        hidden_states = self.decoder(tgt_embeds, memory)

        print("Decoder output shape:", hidden_states.shape)

        hidden = F.relu(self.linear(hidden_states)) # (batch_size, hidden_dim)       
        print("Linear layer shape:", hidden.shape)

        # hidden_tmp = hidden
        # table_output_list = []
        # idx_output_list = []
        # for t in range(self.output_length):
        #     table_output = self.table_output_layer(hidden_tmp)  # (batch_size, table_id_vocab)
        #     idx_output = self.idx_output_layer(hidden_tmp)

        #     table_output_list.append(table_output)
        #     idx_output_list.append(idx_output)

        #     # use the output as the input to the next time step
        #     table_output_id = torch.argmax(table_output, dim=-1)
        #     idx_output_id = torch.argmax(idx_output, dim=-1)

        #     next_input = torch.cat((table_output_embed, idx_output_embed), dim=-1)
        #     hidden_tmp = F.relu(self.linear(next_input))

        # table_outputs = torch.stack(table_output_list, dim=1)  # (batch_size, output_length, table_id_vocab)
        # idx_outputs = torch.stack(idx_output_list, dim=1)       # (batch_size, output_length, idx_id_vocab)
        
        table_outputs = self.table_output_layer(hidden)
        idx_outputs = self.idx_output_layer(hidden)
        print("Output shape:", table_outputs.shape, idx_outputs.shape)

        return table_outputs.permute(1, 0, 2), idx_outputs.permute(1, 0, 2)
    
        
    def generate(self, table_id_seq, idx_id_seq):
        table_id_embeds = self.table_id_embed(table_id_seq)
        idx_id_embeds = self.idx_id_embed(idx_id_seq)

        table_id_embeds = table_id_embeds.permute(1, 0, 2)
        idx_id_embeds = idx_id_embeds.permute(1, 0, 2)

        table_id_ec = self.table_id_transformer(table_id_embeds)
        idx_id_ec = self.idx_id_transformer(idx_id_embeds)

        table_id_ec_mean = table_id_ec.mean(dim=0)
        idx_id_ec_mean = idx_id_ec.mean(dim=0)

        table_id_ec_mean = table_id_ec.mean(dim=0, keepdim=True).expand(self.output_length, -1, -1)
        idx_id_ec_mean = idx_id_ec.mean(dim=0, keepdim=True).expand(self.output_length, -1, -1)

        combined_memory = torch.cat((table_id_ec_mean, idx_id_ec_mean), dim=-1) # (batch_size, embed_dim * 2)
        memory = combined_memory.unsqueeze(0).expand(self.output_length, -1, -1)  # (seq_len, batch_size, embed_dim * 2)

        generated_table_ids = []
        generated_idx_ids = []

        next_table_id = torch.tensor([[self.table_id_vocab - 1]] * table_id_seq.size(0)).to(table_id_seq.device)
        next_idx_id = torch.tensor([[self.idx_id_vocab - 1]] * idx_id_seq.size(0)).to(idx_id_seq.device)

        for _ in range(self.output_length):
            tgt_table_embeds = self.table_id_embed(next_table_id).permute(1, 0, 2)
            tgt_idx_embeds = self.idx_id_embed(next_idx_id).permute(1, 0, 2)

            tgt_embeds = torch.cat((tgt_table_embeds, tgt_idx_embeds), dim=-1)

            hidden_states = self.decoder(tgt_embeds, memory)

            hidden = F.relu(self.linear(hidden_states[-1]))

            table_output = self.table_output_layer(hidden)
            idx_output = self.idx_output_layer(hidden)

            next_table_id = torch.argmax(table_output, dim=-1)
            next_idx_id = torch.argmax(idx_output, dim=-1)

            generated_table_ids.append(next_table_id)
            generated_idx_ids.append(next_idx_id)

        generated_table_ids = torch.cat(generated_table_ids, dim=1)
        generated_idx_ids = torch.cat(generated_idx_ids, dim=1)

        return generated_table_ids, generated_idx_ids





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

