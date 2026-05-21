import os

from common.inconsistents_parser import inconsistents_graph_parser
from torch_geometric.loader import DataLoader

filename = input('Filename: ')
incon_path = input('Inconsistent path: ')

data = inconsistents_graph_parser(os.path.abspath(filename), os.path.abspath(incon_path))
print(f'Number of data: {len(data)}')

loader = DataLoader(data, batch_size=32, follow_batch=['x_1', 'x_2'])
batch = next(iter(loader))

print(f'Batch info: {batch}')
print(f'Batch x1: {batch.x_1}')
print(f'Batch x2: {batch.x_2}')
print(f'Batch edge_index_1: {batch.edge_index_1}')
print(f'Batch edge_index_2: {batch.edge_index_2}')
print(f'Batch x1 batch: {batch.x_1_batch}')
print(f'Batch x2 batch: {batch.x_2_batch}')
print(f'Batch y: {batch.y}')