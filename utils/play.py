import gzip
import typing

import torch


file = "/home/hli31/S2024_MLSYS/dlrm_datasets/fbgemm_t856_bs65536_0.pt"

try:
    with gzip.open(file) as f:
        indices, offsets, lengths = torch.load(f)
except:
    indices, offsets, lengths = torch.load(file)
    print("yeah!!!")

def getQueryID(file, bs) -> typing.List:
    '''
    Args:
    file (str): Path to the dataset file. e.g. embedding bag
    bs (tuple): 
    
    Return:
    queryList (list): the batch sizes in the given range
    '''
    indices, offsets, lengths = torch.load(file)

    if bs == (0, 0):
        return list(range(len(lengths[0])))
    
    start, end = bs
    if start < 1 or end > len(lengths[0]):
        raise ValueError("The range is out of the query size!")
    queryList = list(range(start - 1, end))
    print(f"Query ID scope: ({start}, {end}): {queryList}")
    return queryList

if __name__ == "__main__":
    getQueryID(file, (1, 999))
