from itertools import repeat

import torch
import pdb
import random
import numpy as np
import scipy.sparse as sp
import os.path as osp
from torch_sparse import coalesce
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops, to_undirected

try:
    import cPickle as pickle
except ImportError:
    import pickle

def read_dblp_data(folder, prefix):
    file_path = osp.join(folder,  '{}.{}'.format(prefix.lower(), 'npz'))

    with np.load(file_path) as loader:
        loader = dict(loader)

        x = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                           loader['attr_indptr']), shape=loader['attr_shape'])
        x = torch.Tensor(x.todense()).to(torch.float)
        y = torch.Tensor(loader.get('labels')).long()

        # obtain the train data
        counter = []
        start_node = 0
        for i in range(len(set(loader.get('labels')))):
            counter.append(20)
        y_list = loader.get('labels')
        train_list = []
        for i in range(len(y_list)):
            start_node += 1
            if counter[y_list[i]] != 0:
                train_list.append(i)
                counter[y_list[i]] = counter[y_list[i]] - 1
            if max(counter) == 0:
                break

        train_index = torch.Tensor(train_list).long()
        val_index = torch.Tensor(random.sample(range(start_node, y.size(0)-1), 500)).long()
        test_index = torch.Tensor(random.sample(range(start_node, y.size(0)-1), 1000)).long()
        train_mask = sample_mask(train_index, num_nodes=y.size(0))
        val_mask = sample_mask(val_index, num_nodes=y.size(0))
        test_mask = sample_mask(test_index, num_nodes=y.size(0))

        # obtain the adjacency matrix
        # edge_index = create_edge_index(loader, y.size(0))
        adj = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                             loader['adj_indptr']), loader['adj_shape']).tocoo()
        edge_index = torch.tensor([adj.row, adj.col], dtype=torch.long)
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = to_undirected(edge_index, x.size(0))  # Internal coalesce.

        data = Data(x=x, edge_index=edge_index, y=y)
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

        return data


def create_edge_index(loader, num_nodes=None):
    adj_indptr_source = loader['adj_indptr']
    adj_indices_source = loader['adj_indices']

    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.int)
    for i in range(1, len(adj_indptr_source)):
        left = adj_indptr_source[i - 1]
        right = adj_indptr_source[i]
        if left == right:
            continue
        indices = adj_indices_source[left:right]

        adj_matrix[i - 1, indices] = 1
        adj_matrix[indices, i - 1] = 1

    row, col = [], []

    for i in range(len(adj_matrix)):
        current_row = adj_matrix[i]
        for j in range(len(current_row)):
            if current_row[j] == 1:
                row.append(i)
                col.append(j)

    edge_index = torch.stack([torch.tensor(row), torch.tensor(col)], dim=0)  # shape(2, 2*num_edges)

    # NOTE: There are duplicated edges and self loops in the datasets. Other
    # implementations do not remove them!
    edge_index, _ = remove_self_loops(edge_index)
    edge_index, _ = coalesce(edge_index, None, num_nodes, num_nodes)
    return edge_index


def sample_mask(index, num_nodes):
    mask = torch.zeros((num_nodes, ), dtype=torch.uint8)
    mask[index] = 1
    return mask