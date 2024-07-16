import torch
# import numpy as np
import matplotlib.pyplot as plt


data = torch.load('../new_Trace/fbgemm_t856_bs65536_0_trace_201_250.pt')

print(len(torch.unique(data[:, 0])))
print(len(torch.unique(data[:, 2])))
if isinstance(data, dict):
    for key, value in data.items():
        print(f"Key: {key}, Type: {type(value)}, Shape: {value.shape if isinstance(value, torch.Tensor) else 'N/A'}")
        
elif isinstance(data, torch.Tensor):
    print(f"Type: {type(data)}, Shape: {data.shape}")
    table_id = data[:, 0]
    idx_id = data[:, 1]

    plt.figure(figsize=(12, 6))
    plt.hist(table_id.numpy(), bins='auto', alpha=0.5, label='table_id')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('table_id Distribution')
    plt.legend()
    plt.savefig('table_id_distribution.png')


    plt.figure(figsize=(12, 6))
    plt.hist(idx_id.numpy(), bins='auto', alpha=0.5, label='idx_id')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('idx_id Distribution')
    plt.legend()
    plt.savefig('idx_id_distribution.png')
