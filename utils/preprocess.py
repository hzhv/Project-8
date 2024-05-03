import gzip
import typing

import numpy as np
import click

import torch


@click.command()
@click.option(
    "--file",
    type=str,
    default="/home/hli31/S2024_MLSYS/dlrm_datasets/fbgemm_t856_bs65536_0.pt",
    help="Embedding bag data file",
)
@click.option(
    "--scope",
    nargs=2,
    type=int,
    default=[66, 69],
    help="the Trace range, e.g. (1, 856)",
)
@click.option(
    "--bs",
    nargs=2,
    type=int,
    default=(0, 0),
    help="the query range, e.g. (1, 65536)",
)
def main(file, scope, bs):
    if scope:
        getTraces_tableID(file, scope)
    elif bs:
        getTraces_queryID(file, bs)
    else:
        print("No valid range is given, cannot get traces,")
        print("Please provide a valid range for table ID or batch size.")
    

def getDataset(file, factor=1):
    try:
        with gzip.open(file) as f:
            indices, offsets, lengths = torch.load(f)
    except:
        indices, offsets, lengths = torch.load(file)
    print(f"indices shape: {indices.shape}")
    print(f"Offsets shape:, {offsets.shape}")
    print(f"Lengths shape:, {lengths.shape}")
    print()

    if factor < 1:
        items = np.random.choice(indices, int(len(items) * factor), replace=False)

        print(f"Number of unique indices after random sampling: {len(items)}")

        indices = file[0:file.rfind(".pt")] + "_cached.csv"

        np.savetxt(indices, items.reshape(1, -1), delimiter=",", fmt="%d")
    return indices, offsets, lengths

def getTableID(file, scope) -> typing.List: 
    '''
    Args:
    file (str): Path to the dataset file. e.g. embedding bag
    scope (tuple): start from 1 e.g. [1, 100]->[0, 99]
    
    Return: 
    tableIDList (list): the table id indices of given range
    '''
    _, _, lengths = getDataset(file)

    if scope == (0, 0):
        print(f"Table ID indices: {0, len(lengths)}")
        return list(range(len(lengths)))

    start, end = scope
    if start < 1 or end > len(lengths):
        raise ValueError("The range is out of the embedding table size!")
    tableIDScope = list(range(start - 1, end))
    tableIDList = list(range(0, end))
    print(f"Table ID scope: ({start}, {end}): {tableIDScope}")
    return tableIDList

def getQueryID(file, bs) -> typing.List:
    '''
    Args:
    file (str): Path to the dataset file. e.g. embedding bag
    bs (tuple): 
    
    Return:
    queryList (list): the query id indices of the given range
    '''
    _, _, lengths = getDataset(file)

    if bs == (0, 0):
        return list(range(len(lengths[0])))
    
    start, end = bs
    if start < 1 or end > len(lengths[0]):
        raise ValueError("The range is out of the query size!")
    queryList = list(range(start - 1, end))
    print(f"Query ID scope: ({start}, {end}): {queryList}")
    return queryList

def getTraces_tableID(file, scope):
    '''
    Args:
    file (str): Path to the dataset file
    scope (tuple): A tuple range of table IDs, e.g. [0, 856)

    Return:
    - traces (torch.Tensor): shape [N, 2], where N is the number of samples
    Each row in the tensor represents a sample as [table_id, idx_id]
    '''
    start_table_id = scope[0]
    samples = [] # (table_id, idx_id)
    start_idx = 0
    exe_flag = False

    indices, _, lengths = getDataset(file)
    tableIDList = getTableID(file, scope)

    for table_id in tableIDList:
        pf_sum = lengths[table_id].sum().item()
        end_idx = start_idx + pf_sum
        # if pf_sum == 0:
        #     end_idx = start_idx # will save a [] in idxIDs
        # else:
        #     end_idx = start_idx + end_idx
        if not exe_flag and table_id != start_table_id - 1:
            start_idx = end_idx
            continue
        elif table_id == start_table_id - 1:
            exe_flag = True
        # BUG, add 0 to idxIDS
        idxIDs = indices[start_idx:end_idx].tolist()
        for idx_ID in idxIDs:
            sample_tensor = torch.tensor([table_id, idx_ID], dtype=torch.int32)
            samples.append(sample_tensor)
        print(f"table ID: {table_id}, start_idx: {start_idx}, end_idx: {end_idx}, sum pf: {pf_sum}\n")
        start_idx = end_idx
    
    traces = torch.stack(samples)
    print(f"traces shape: {traces.shape}")
    
    traces_pt = torch.save(traces, "demo_traces.pt")

    ## following are test codes:
    t = torch.load("demo_traces.pt")
    print(t[0])
    return traces

def getTraces_queryID(file, bs):
    '''
    Args:
    file (str): Path to the dataset file
    bs (tuple): A tuple range of batch size, query original samples [0, 65535)

    Return:
    - traces (torch.Tensor): shape [N, 2], where N is the number of samples
    Each row in the tensor represents a sample as [table_id, idx_id]
    '''
    indices, _, lengths = getDataset(file)
    queryIDList = getQueryID(file, bs)

    samples = [] # (table_id, idx_id)
    start_idx = 0
    for query_id in queryIDList:
        lengths[table_id].sum().item()
        pf_sum = lengths[:, query_id].sum().item()
        end_idx = start_idx + pf_sum

        idxIDs = indices[start_idx:end_idx].tolist()

    # TODO: Hanzhao WIP
    # np.sum(lengths[:row, :]) + np.sum(lengths[row, :col])
    return 0

if __name__ == "__main__":
    import time
    start = time.time()
    main()
    end = time.time()
    print(f"Time: {(end - start)/60:.2f} min")

