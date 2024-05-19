04/23/2024

lengths.size = [856，65536]
[i, j] = embedding table i的第j个query下的pooling factor (PF) 
PF: number of embedding rows that need to be looked up per query

Trace: sequence of samples
Each sample: [T_id, Idx_id]; 
e.g. Length[0][0] = 10, generate 10 samples from embedding table_0
Then
Process the same query in embedding table_1,
If  Length[1][0] = 3, (PF)

from torch loaded indices to extract Idx_id,
idx_id_0 = indices[sum(lengths[0][0] + … + lengths[table_id][col - 1])]
idx_id_1 = indices[sum(lengths[0][0] + … + lengths[table_id][col - 1]) + 1]
idx_id_2 = indices[sum(lengths[0][0] + … + lengths[table_id][col - 1]) + 2]


TODO: develop the preprocessing python script with following new feature:
- Could Specify batch size range = query original range [0, 65535]
	e.g. Input a list (2 , 3, 4, 5), return list corresponding pair
- Specify embedding table, range = [0, 855]
       e.g. Input a list (2 , 3, 4) return list corresponding pair



假设输入序列为 [(table_id 1, idx_id 2), (table_id 3, idx_id 2), (table_id 5, idx_id 6)]，
下一个序列为   [(table_id 3, idx_id 2), (table_id 5, idx_id 6), (table_id 7, idx_id 8)]，
那么 
             (table_id 1, idx_id 2) 的 GT 为 0， (table_id 7, idx_id 8) 的 GT 为 0
             (table_id 3, idx_id 2) 和 (table_id 5, idx_id 6) 的 GT 为 1

