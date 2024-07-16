import os
import time
import logging
import typing
import gzip

import numpy as np
import click
import torch

from dataset_load import get_unqs

@click.command()
@click.option(
    "--file",
    type=str,
    default="/home/hli31/S2024_MLSYS/dlrm_datasets/2021/fbgemm_t856_bs65536_0.pt",
    help="Embedding bag data file",
)
@click.option(
    "--scope",
    nargs=2,
    type=int,
    help="the Table_id range, e.g. (1, 856)",
)
@click.option(
    "--bs",
    nargs=2,
    type=int,
    help="the Table_id range",
)
def main(file, scope, bs):
    start = time.time()
    log_directory = "/home/hli31/S2024_MLSYS/Project-8/trace_logs/"
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    log_file = os.path.join(log_directory, file[file.rfind("fbgemm"):file.rfind(".pt")] + "_traces.log")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_file),
                        logging.StreamHandler()
                    ])
    logging.info("Start to get traces...")
    
    if scope:
        getTraces_tableID(file, scope)
    elif bs:
        getTraces_tableID_queryID(file, bs)
    else:
        raise AttributeError("No valid range is given, cannot get traces,\
                            Please provide a valid range for table ID or batch size.")
    end = time.time()
    logging.info(f"Finished! Time cost: {(end - start)/60:.2f} min\n")
    
def getDataset(file, factor=1):
    try:
        with gzip.open(file) as f:
            indices, offsets, lengths = torch.load(f)
    except:
        indices, offsets, lengths = torch.load(file)
    logging.info(f"indices shape: {indices.shape}")
    logging.info(f"Offsets shape:, {offsets.shape}")
    logging.info(f"Lengths shape:, {lengths.shape}\n")

    if factor < 1:
        items = np.random.choice(indices, int(len(items) * factor), replace=False)

        print(f"Number of unique indices after random sampling: {len(items)}")

        indices = file[0:file.rfind(".pt")] + "_cached.csv"

        np.savetxt(indices, items.reshape(1, -1), delimiter=",", fmt="%d")
    return indices, offsets, lengths

def getTableInfo(file, scope): 
    '''
    Args:
    file (str): Path to the dataset file. e.g. embedding bag
    scope (tuple): start from 1 e.g. [1, 100]->[0, 99]
    
    Return: 
    embag_indices (torch.Tensor): shape [1,], the indices of the embedding bag
    embag_lengths (torch.Tensor): shape [1,], the lengths of the embedding bag
    tableIDList (list): the table id indices of given scope
    '''
    embag_indices, _, embag_lengths = getDataset(file)

    if scope == (0, 0):
        logging.info(f"Table ID scope: {1, len(embag_lengths)}")
        return embag_indices, embag_lengths, list(range(len(embag_lengths)))

    start, end = scope
    if start < 1 or end > len(embag_lengths):
        raise ValueError("The range is out of the embedding table size!")
    tableIDScope = list(range(start - 1, end))
    tableIDList = list(range(0, end))
    logging.info(f"Table ID scope ({start}, {end}): {tableIDScope}\n")
    return embag_indices, embag_lengths, tableIDList

def getQueryInfo(file, bs) -> typing.List:
    '''
    Args:
    file (str): Path to the dataset file. e.g. embedding bag
    bs (tuple): 
    
    Return:
    queryList (list): the query id indices of the given range
    '''
    embag_indices, _, embag_lengths = getDataset(file)

    if bs == (0, 0):
        return list(range(len(lengths[0])))
    
    start, end = bs
    if start < 1 or end > len(lengths[0]):
        raise ValueError("The range is out of the query size!")
    queryList = list(range(start - 1, end))
    logging.info(f"Query ID scope: ({start}, {end}): {queryList}")
    return embag_indices, embag_lengths, queryList

def getTraces_tableID(file, scope):
    '''
    Args:
    file (str): Path to the dataset file
    scope (tuple): A tuple range of table IDs, e.g. [0, 856)

    Return:
    - traces (torch.Tensor): shape [N, 2], where N is the number of samples
    Each row in the tensor represents a sample as [table_id, idx_id]
    '''
    samples = [] # (table_id, idx_id)
    start_table_id = scope[0]
    start_idx = 0
    skiped_tables = []
    exe_flag = False

    indices, lengths, tableIDList = getTableInfo(file, scope)

    for table_id in tableIDList:
        pf_sum = lengths[table_id].sum().item()
        end_idx = start_idx + pf_sum  # Two Pointers

        if scope == (0, 0):
            pass
        elif not exe_flag and table_id != start_table_id - 1:
            start_idx = end_idx
            continue
        elif table_id == start_table_id - 1:
            exe_flag = True
        
        if pf_sum == 0:
            skiped_tables.append(table_id)
            print(f"Notice: table {table_id} has 0 indices, skip to the next table, sum pf: {pf_sum}")
            continue
        idxIDs = indices[start_idx:end_idx].tolist()
        for idx_ID in idxIDs:
            sample_tensor = torch.tensor([int(table_id), int(idx_ID)], dtype=torch.int64)
            ## test codes:
            if torch.isinf(sample_tensor).any():
                print(f"Created inf in tensor for table_id = {table_id}, idx_ID = {idx_ID}")
            samples.append(sample_tensor)
   
        print(f"Table ID: {table_id}, Start Index: {start_idx}, End Index: {end_idx}, sum pf: {pf_sum}\n")
        start_idx = end_idx
    
    traces = torch.stack(samples)
    logging.info(f"Skiped Table ID in the given scope {scope}: {skiped_tables}")
    logging.info(f"Trace shape: {traces.shape}")
    ptName = "/home/hli31/S2024_MLSYS/Trace/" + file[file.rfind("fbgemm"):file.rfind(".pt")] + f"_trace_{start_table_id}_{tableIDList[-1]}.pt"
    
    ## following are test codes:
    inf_mask = torch.isinf(traces)
    if inf_mask.any():
        logging.warning(f"HUH? Detected inf {traces}")
    ## end of test codes

    torch.save(traces, ptName)
    return traces

def getTraces_tableID_queryID(file, scope):
    '''
    Args:
    file (str): Path to the dataset file
    scope (tuple): A tuple range of table IDs, e.g. [0, 856)

    Return:
    - traces (torch.Tensor): shape [N, 3], where N is the number of samples
    Each row in the tensor represents a sample as [table_id, query_id, idx_id]
    '''

    samples = [] # (table_id, query_id, idx_id)
    start_table_id = scope[0]
    start_idx = 0
    skiped = []
    exe_flag = False

    indices, lengths, tableIDList = getTableInfo(file, scope)
    queryListScope = len(lengths[0])  # default to get all queries
    
    for table_id in tableIDList:
        for query_id in range(queryListScope):
            

            pf = lengths[table_id][query_id].item()
            end_idx = start_idx + pf

            if not exe_flag and table_id != start_table_id - 1:
                start_idx = end_idx
                continue
            elif table_id == start_table_id - 1:
                exe_flag = True

            if pf == 0:
                skiped.append([table_id,query_id])
                continue

            idxIDs = indices[start_idx:end_idx].tolist()
            for idx_ID in idxIDs:
                sample_tensor = torch.tensor([int(table_id), int(query_id), int(idx_ID)], dtype=torch.int64)
            ## test codes:
                if torch.isinf(sample_tensor).any():
                    print(f"Created inf in tensor for table_id = {table_id}, idx_ID = {idx_ID}")
                samples.append(sample_tensor)

            start_idx = end_idx
        print(f"Table ID: {table_id}, Query ID: {query_id} Start Index: {start_idx}, End Index: {end_idx}\n")

    traces = torch.stack(samples)
    # logging.info(f"Skiped samples in the given scope {scope}: {skiped}")
    logging.info(f"Trace shape: {traces.shape}")
    ptName = "/home/hli31/S2024_MLSYS/new_Trace/" + file[file.rfind("fbgemm"):file.rfind(".pt")] + f"_trace_{start_table_id}_{tableIDList[-1]}.pt"
    
    ## following are test codes:
    inf_mask = torch.isinf(traces)
    if inf_mask.any():
        logging.warning(f"HUH? Detected inf {traces}")
    ## end of test codes

    torch.save(traces, ptName)
    
    return traces


if __name__ == "__main__":
    main()

