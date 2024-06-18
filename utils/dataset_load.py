import logging

import torch
from torch.utils.data import Dataset, DataLoader
import yaml


class TraceDataset(Dataset): # load transformed data
    def __init__(self, file_path, sequence_length, prediction_steps):
        self.file_path = file_path
        self.data = torch.load(self.file_path)
        self.data_length = len(self.data)
        self.sequence_length = sequence_length
        self.prediction_steps = prediction_steps
        # with h5py.File(file_path, 'r') as f:
        #     self.data = f['data']
        #     self.data_length = len(self.data)

    def __len__(self):
        return len(self.data) - self.sequence_length - self.prediction_steps + 1

    def __getitem__(self, index):
        tgt_start = index + self.sequence_length
        tgt_end = tgt_start + self.prediction_steps
        x = self.data[index : index + self.sequence_length]
        table_id_seq = x[:, 0]
        idx_id_seq = x[:, 1]
        gt_table = self.data[tgt_start : tgt_end, 0]
        gt_idx = self.data[tgt_start : tgt_end, 1]

        return table_id_seq, idx_id_seq, gt_table, gt_idx

def get_num_classes(data, index):
    return len(torch.unique(data[:, index]))

def load_dataset(file_path, sequence_length, prediction_steps, batch_size, shuffle=True):
    """
    Args:
    - data_file (str): Path to the trace data file.
    - sequence_length (int): Number of (table_id, idx_id) pairs to use as input features.
    - prediction_steps (int): Number of (table_id, idx_id) pairs to predict.
    - batch_size (int): Batch size for DataLoader.

    Return:
    Torch DataLoader for the TraceDataset
    """

    tcDataset = TraceDataset(file_path, sequence_length, prediction_steps)
    data_loader = DataLoader(tcDataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True)
    input_length = tcDataset.data_length
    print(f"Input datset has {input_length} pairs of (table_id, idx_id).")

    table_unq = get_num_classes(tcDataset.data, 0)  
    idx_unq = get_num_classes(tcDataset.data, 1)   

    return data_loader, input_length, table_unq, idx_unq

def load_config(config_path):
    with open(config_path, "r") as f:
        try:
            configs = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            logging.error("Error in loading configuration", exc)
            return {}
    return configs

# def load_dataset(data_file, sequence_length, prediction_steps, batch_size, shuffle=True):
#     """
#     Args:
#     - data_file (str): Path to the TRANSFORMED trace data file.
#     - sequence_length (int): Number of (table_id, idx_id) pairs to use as input features.
#     - prediction_steps (int): Number of (table_id, idx_id) pairs to predict.
#     - batch_size (int): Batch size for DataLoader.

#     Return:
#     Torch DataLoader for the TraceDataset
#     """
#     transformed_data = transform_dataset(data_file)
#     tcDataset = TraceDataset(transformed_data, sequence_length, prediction_steps)
#     data_loader = DataLoader(tcDataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
#     num_classes_table = get_num_classes(tcDataset.data, 0)
#     num_classes_idx = get_num_classes(tcDataset.data, 1)
#     return data_loader, num_classes_table, num_classes_idx

