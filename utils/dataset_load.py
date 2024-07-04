import os
import logging

import torch
from torch.utils.data import Dataset, DataLoader
import yaml


class TraceDataset(Dataset):
    def __init__(self, file_path, sequence_length, prediction_ratio):
        self.file_path = file_path
        self.data = torch.load(self.file_path)
        self.data_length = len(self.data)
        self.sequence_length = sequence_length
        self.prediction_steps = int(sequence_length * prediction_ratio)
        
        unique_table_ids = torch.unique(self.data[:, 0])
        unique_idx_ids = torch.unique(self.data[:, 1])
        self.table_id_map = {v.item(): i for i, v in enumerate(unique_table_ids)}
        self.idx_id_map = {v.item(): i for i, v in enumerate(unique_idx_ids)}

        # logging.info(f"Unique table ids: {unique_table_ids}")
        # logging.info(f"Unique idx ids: {unique_idx_ids}")
        logging.info(f"Number of unique table ids: {len(unique_table_ids)}")
        logging.info(f"Number of unique idx ids: {len(unique_idx_ids)}")
        # logging.info(f"Table ID map: {self.table_id_map}")
        # logging.info(f"Idx ID map: {self.idx_id_map}")

        self.data[:, 0] = torch.tensor([self.table_id_map[v.item()] for v in self.data[:, 0]])
        self.data[:, 1] = torch.tensor([self.idx_id_map[v.item()] for v in self.data[:, 1]])

    def __len__(self):
        return (self.data_length - self.sequence_length - self.prediction_steps + 1) // self.sequence_length

    def __getitem__(self, index):
        start_idx = index * self.sequence_length
        end_idx = start_idx + self.sequence_length
        tgt_start = end_idx
        tgt_end = tgt_start + self.prediction_steps

        x = self.data[start_idx:end_idx]
        table_id_seq, idx_id_seq = x[:, 0], x[:, 1]
        gt_table = self.data[tgt_start:tgt_end, 0]
        gt_idx = self.data[tgt_start:tgt_end, 1]

        return table_id_seq, idx_id_seq, gt_table, gt_idx

    def get_maps(self):
        return self.table_id_map, self.idx_id_map
    

def load_dataset(file_path, sequence_length, prediction_ratio, batch_size, shuffle=True):
    """
    Args:
    - file_path (str): Path to the trace data file.
    - sequence_length (int): Number of (table_id, idx_id) pairs to use as input features.
    - prediction_ratio(float): Percentage of (table_id, idx_id) pairs to predict.
    - batch_size (int): Batch size for DataLoader.

    Return:
    Torch DataLoader for the TraceDataset
    """
    log_dir = "/home/hli31/S2024_MLSYS/Project-8-exp/exp_logs/"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, file_path[file_path.rfind("fbgemm"):file_path.rfind(".pt")] + ".log")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_file),
                        logging.StreamHandler()
                    ])

    tcDataset = TraceDataset(file_path, sequence_length, prediction_ratio)
    input_length = tcDataset.data_length
    output_length = tcDataset.prediction_steps
    data_loader = DataLoader(tcDataset, batch_size, shuffle=shuffle, num_workers=4, pin_memory=True)
    logging.info(f"Input dataset has {input_length} pairs of (table_id, idx_id).")
    table_unq = get_unqs(tcDataset.data, 0)
    idx_unq = get_unqs(tcDataset.data, 1)
    n_table_unq, n_idx_unq = len(table_unq), len(idx_unq)
    table_id_map, idx_id_map = tcDataset.get_maps()
    
    return data_loader, output_length, n_table_unq, n_idx_unq, table_id_map, idx_id_map

def load_config(config_path):
    with open(config_path, "r") as f:
        try:
            configs = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            logging.error("Error in loading configuration", exc)
            return {}
    return configs

def get_unqs(data, index):
    """
    return: unique elements of input trace
    """
    return torch.unique(data[:, index])