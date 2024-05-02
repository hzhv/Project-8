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
    default=(0, 0),
    help="the Trace range, e.g. 1 100",
)
@click.option(
    "--bs",
    nargs=2,
    type=int,
    default=(0, 0),
    help="the Trace range, e.g. 1 100",
)
def main(file, scope, bs):
    getIdxID(file, scope)


def getDataset(file, factor=1):
    if file.endswith(".gz"):
        with gzip.open(file) as _:
            indices, offsets, lengths = torch.load(_)
    else:
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
    file: the dataset file, e.g. embedding bag
    scope: start from 0 e.g. [0, 100], type Tuple
    return: the table ID in the given range, type List
    '''
    indices, offsets, lengths = getDataset(file)

    if scope == (0, 0):
        return list(range(len(lengths)))

    start, end = scope
    if start < 1 or end > len(lengths):
        raise ValueError("The range is not valid")
    tableIDList = list(range(start - 1, end))
    print(f"Table ID scope: ({start}, {end}): {tableIDList}")
    return tableIDList

def getIdxID(file, scope):
    # idx_id_0 = indices[sum(lengths[0][0] + … + lengths[table_id][col - 1])]
    tableID = getTableID(file, scope)
    indices, _, lengths = getDataset(file)
    idxIDList = []
    
    start_idx = 0
   
    for table_id in tableID:
        # for col in range(3):#(len(lengths[0])):
        #     pf = lengths[table_id][col].item()
        #     buffer += pf
        #     print(f"table ID: {table_id}, pf: {pf}, buffer: {buffer}")

        #     for i in range(pf, buffer):
        #         idxIDList.append(indices[i].item())
                
        #     print(f"idx_id: {idxIDList}\n")
        # pf = lengths[table_id].sum().item()
        # print(f"table ID: {table_id}, pf: {pf}")
    
        # Calculate the starting index for the next table ID

        
        pf = lengths[table_id].sum().item()
        
        if pf == 0:
            end_idx = start_idx
        else:
            end_idx = start_idx + pf - 1

        idxIDs = indices[start_idx:end_idx].tolist()
        idxIDList.append(idxIDs)
        print(f"table ID: {table_id}, pf: {pf}, start_idx: {start_idx}, end_idx: {end_idx}\n")
        # print(f"idx_id range: {idxIDList[start_idx:end_idx]}")
        start_idx = end_idx + 1
        # break

    return 0

def getTrace():
    return


if __name__ == "__main__":
    main()
