import os.path as osp

import torch
import matplotlib.pyplot as plt

tc = "../new_Trace/fbgemm_t856_bs65536_0_trace_201_250.pt"
file = torch.load(tc)
file_name = osp.basename(tc).split('.')[0]

unq_table_id = torch.unique(file[:, 0])

data_dict = {
    table_id.item(): (unq_idx_id_each_table := torch.unique(file[file[:, 0] == table_id][:, 2]), len(unq_idx_id_each_table))
    for table_id in unq_table_id
}

table_ids = list(data_dict.keys())
unique_idx_counts = [data_dict[table_id][1] for table_id in table_ids]

plt.figure(figsize=(12, 6))
bars = plt.bar(table_ids, unique_idx_counts, color='skyblue')
plt.xlabel('Table ID')
plt.ylabel('Unique Idx_ID Count')
plt.title(f'Distribution of Unique Idx_ID Count for Each Table_ID, {file_name}')
plt.xticks(rotation=90)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), ha='center', va='bottom')

plt.tight_layout()
plt.savefig(f'{file_name}_dist.png')