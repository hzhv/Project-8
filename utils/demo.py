import torch
import torch.optim as optim



class TraceDataset(Dataset):
    def __init__(self, file_path, sequence_length, prediction_ratio, idx_tokenizer_path=""):
        self.file_path = file_path
        self.data = torch.load(self.file_path)
        self.data_length = len(self.data)
        self.sequence_length = sequence_length
        
        unique_table_ids = torch.unique(self.data[:, 0])
        self.table_id_map = {v.item(): i for i, v in enumerate(unique_table_ids)}
        self.reverse_table_id_map = {i: v.item() for i, v in enumerate(unique_table_ids)}
        
        self.data[:, 0] = torch.tensor([self.table_id_map[v.item()] for v in self.data[:, 0]])

    def __len__(self):
        return self.data_length - self.sequence_length

    def __getitem__(self, index):
        start_idx = index
        end_idx = start_idx + self.sequence_length

        x = self.data[start_idx:end_idx]
        table_id_seq, idx_id_seq = x[:, 0], x[:, 1]
        gt_table = self.data[end_idx, 0]
        gt_idx = self.data[end_idx, 1]

        return table_id_seq, idx_id_seq, gt_table, gt_idx

    def get_maps(self):
        return self.table_id_map, self.reverse_table_id_map



def train_model(train_loader, table_vocab_size, idx_vocab_size, embedding_dim, hidden_dim, num_layers, num_epochs):
    model = SequencePredictor(table_vocab_size, idx_vocab_size, embedding_dim, hidden_dim, num_layers)
    criterion_table = nn.CrossEntropyLoss()
    criterion_idx = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        for table_seq, idx_seq, gt_table, gt_idx in train_loader:
            optimizer.zero_grad()
            table_out, idx_out = model(table_seq, idx_seq)
            loss_table = criterion_table(table_out, gt_table)
            loss_idx = criterion_idx(idx_out, gt_idx)
            loss = loss_table + loss_idx
            loss.backward()
            optimizer.step()
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")


file_path = 'path_to_your_data.pt'
sequence_length = 128
prediction_ratio = 0.2
batch_size = 32
num_epochs = 10
embedding_dim = 128
hidden_dim = 256
num_layers = 4

train_loader, output_length, table_unq, table_id_map, reverse_table_id_map = load_dataset(file_path, sequence_length, prediction_ratio, batch_size)
train_model(train_loader, table_unq, 10000000, embedding_dim, hidden_dim, num_layers, num_epochs)
