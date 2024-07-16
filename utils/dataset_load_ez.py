import os
import logging

import yaml
import torch
from torch.utils.data import Dataset, DataLoader


class TraceDataset(Dataset):
    def __init__(self, file_path, sequence_length, prediction_ratio, idx_tokenizer_path=""):
        self.file_path = file_path
        self.data = torch.load(self.file_path)
        self.data_length = len(self.data)
        self.sequence_length = sequence_length
        self.prediction_steps = int(sequence_length * prediction_ratio)
        

        unique_table_ids = torch.unique(self.data[:, 0])
        self.table_id_map = {v.item(): i for i, v in enumerate(unique_table_ids)}
        self.reverse_table_id_map = {i: v.item() for i, v in enumerate(unique_table_ids)}
        
        self.data[:, 0] = torch.tensor([self.table_id_map[v.item()] for v in self.data[:, 0]])

    def __len__(self):
        # return self.data_length - self.sequence_length
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
        return self.table_id_map, self.reverse_table_id_map

        
def load_dataset(file_path, sequence_length, prediction_ratio, batch_size, shuffle=True):
    log_dir = "../Project-8-exp/exp_logs/"
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

    table_id_map, reverse_table_id_map = tcDataset.get_maps()
    table_unq = len(table_id_map)

    return data_loader, output_length, table_unq, table_id_map, reverse_table_id_map


def load_config(config_path):
    with open(config_path, "r") as f:
        try:
            configs = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            logging.error("Error in loading configuration", exc)
            return {}
    return configs

