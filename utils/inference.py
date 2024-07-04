import sys
import os

current_path = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_path)
print(parent_dir)
sys.path.append(parent_dir)

import torch
from torch.utils.data import DataLoader
import click

from model.dlrm_prefetcher import DLRMPrefetcher
from utils.dataset_load import TraceDataset, load_dataset


def load_model(path, table_id_vocab, idx_id_vocab, embed_dim, block_size=2048, n_heads=8, n_layers=2):
    model = DLRMPrefetcher(table_id_vocab, idx_id_vocab, embed_dim, block_size, n_heads, n_layers).cuda()
    model.load_state_dict(torch.load(path))
    return model


if __name__ == "__main__":
    trace_path = '/home/hli31/S2024_MLSYS/Trace/fbgemm_t856_bs65536_0_trace_551_555.pt'
    sequence_length = 2048
    prediction_ratio = 0.2
    batch_size = 64
    model_path = '/home/hli31/S2024_MLSYS/Project-8/configs/prefetcher_transformer_64_5_2000.pt'

    data_loader, output_length, n_table_unq, n_idx_unq, table_id_map, idx_id_map = load_dataset(
        trace_path, sequence_length, prediction_ratio, batch_size
    )

    model = load_model(model_path, n_table_unq, n_idx_unq, embed_dim=128, block_size=2048, n_heads=8, n_layers=2)

    reverse_table_id_map = {v: k for k, v in table_id_map.items()}
    reverse_idx_id_map = {v: k for k, v in idx_id_map.items()}

    for batch in data_loader:
        table_id_seq, idx_id_seq, gt_table, gt_idx = batch

        with torch.no_grad():
            table_outputs, idx_outputs = model(table_id_seq, idx_id_seq)
        _, predicted_table = torch.max(table_outputs, dim=-1)
        _, predicted_idx = torch.max(idx_outputs, dim=-1)

        predicted_table_original = torch.tensor([reverse_table_id_map[v.item()] for v in predicted_table.view(-1)]).view(predicted_table.size())
        predicted_idx_original = torch.tensor([reverse_idx_id_map[v.item()] for v in predicted_idx.view(-1)]).view(predicted_idx.size())

        print(f"Predicted original table values: {predicted_table_original}")
        print(f"Predicted original idx values: {predicted_idx_original}")
        break
