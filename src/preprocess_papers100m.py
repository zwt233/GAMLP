import argparse
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected, dropout_adj
import os
import os.path as osp
import numpy as np
from ogb.nodeproppred import PygNodePropPredDataset
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
import dgl.function as fn
parser = argparse.ArgumentParser()
parser.add_argument('--num_hops', type=int, default=20)
args = parser.parse_args()
print(args)

dataset = PygNodePropPredDataset('ogbn-papers100M', root='../../')
split_idx = dataset.get_idx_split()
train_idx, valid_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']
data = dataset[0]
num_classes = (data.y.data.to(torch.long).max()+1).item()
x = data.x.numpy()
y = np.zeros(shape=(data.y.shape[0],num_classes))
y[train_idx]=data.y.numpy()[train_idx]
N = data.num_nodes
path = './adj_gcn.pt'
if osp.exists(path):
    adj = torch.load(path)
else:
    print('Making the graph undirected.')
    # Randomly drop some edges to save computation
    data.edge_index, _ = dropout_adj(
        data.edge_index, p=0., num_nodes=data.num_nodes)
    data.edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)
    print(data)
    row, col = data.edge_index
    print('Computing adj...')
    adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N))
    adj = adj.set_diag()
    deg = adj.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    adj = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)
    torch.save(adj, path)
adj = adj.to_scipy(layout='csr')

print('Start processing')

for i in tqdm(range(args.label_num_hops)):
    y = adj @ y
torch.save(torch.from_numpy(y[train_idx]).to(
        torch.float), f'./y_train.pt')
torch.save(torch.from_numpy(y[valid_idx]).to(
        torch.float), f'./y_valid.pt')
torch.save(torch.from_numpy(y[valid_idx]).to(
        torch.float), f'./y_test.pt')
exit()        
for i in tqdm(range(args.num_hops)):
    x = adj @ x
    torch.save(torch.from_numpy(x[train_idx]).to(
        torch.float), f'./x_train_{i+1}.pt')
    torch.save(torch.from_numpy(x[valid_idx]).to(
        torch.float), f'./x_valid_{i+1}.pt')
    torch.save(torch.from_numpy(x[test_idx]).to(
        torch.float), f'./x_test_{i+1}.pt')
