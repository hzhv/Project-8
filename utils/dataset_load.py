import logging
import torch
from torch.utils.data import Dataset, DataLoader
import yaml

# def generate_training_data(trace_file, sequence_length, prediction_steps):
#     """
#     Generate training samples and corresponding GT from trace data.

#     Args:
#     - trace_file (str): Path to the trace file containing (table_id, idx_id) pairs.
#     - sequence_length (int): Number of (table_id, idx_id) pairs to use as input features.
#     - prediction_steps (int): Number of (table_id, idx_id) pairs to predict.

#     Returns:
#     - X (torch.Tensor): Tensor of input features of shape [num_samples, sequence_length, 2].
#     - Y (torch.Tensor): Tensor of target outputs of shape [num_samples, prediction_steps, 2].
#     """
 
#     tc = torch.load(trace_file)
#     X, Y = [], []

#     # Generate samples where we have enough data to form a complete sequence + prediction
#     num_samples = tc.size(0) - sequence_length - prediction_steps + 1
#     for i in range(num_samples):
#         X.append(tc[i:i+sequence_length])
#         Y.append(tc[i+sequence_length:i+sequence_length+prediction_steps])

#     X = torch.stack(X)
#     Y = torch.stack(Y)

#     return X, Y

class TraceDataset(Dataset):
    def __init__(self, data_file, sequence_length, prediction_steps):
        self.data = torch.load(data_file)
        self.sequence_length = sequence_length
        self.prediction_steps = prediction_steps

    def __len__(self):
        return len(self.data) - self.sequence_length - self.prediction_steps + 1

    def __getitem__(self, idx):
        start = idx
        end = idx + self.sequence_length
        x = self.data[start:end]
        target_start = end
        target_end = target_start + self.prediction_steps
        Y = self.data[target_start:target_end]
        tgt_table = Y[:, 0]
        tgt_idx = Y[:, 1]
        return x, (tgt_table, tgt_idx)

    def get_num_tables(self, data_file):
        num_classes_table = len(torch.unique(data_file[:, 0]))
        print(f"Number of unique table IDs: {num_classes_table}")
        return num_classes_table

    def get_num_indices(self, data_file):
        num_classes_idx = len(torch.unique(data_file[:, 1]))
        print(f"Number of unique idx IDs: {num_classes_idx}")
        return num_classes_idx


def load_dataset(data_file, sequence_length, prediction_steps, batch_size, shuffle=True):
    """
    Args:
    - data_file (str): Path to the trace data file.
    - sequence_length (int): Number of (table_id, idx_id) pairs to use as input features.
    - prediction_steps (int): Number of (table_id, idx_id) pairs to predict.
    - batch_size (int): Batch size for DataLoader.

    Return:
    Torch DataLoader for the TraceDataset
    """
    tcDataset = TraceDataset(data_file, sequence_length, prediction_steps)
    num_classes_table = tcDataset.get_num_tables(tcDataset.data)
    num_classes_idx = tcDataset.get_num_indices(tcDataset.data)
    return DataLoader(tcDataset, batch_size=batch_size, shuffle=shuffle, num_workers=4), num_classes_table, num_classes_idx

def load_config(config_path):
    with open(config_path, "r") as f:
        try:
            configs = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            logging.warning(exc)
    return configs

