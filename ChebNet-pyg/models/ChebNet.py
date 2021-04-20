import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv

import torch_cluster


class ChebNet(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_classes, num_hops, dropout):
        super(ChebNet, self).__init__()
        self.dropout = dropout
        self.layer_1 = ChebConv(input_dim, hidden_dim, num_hops)
        self.layer_2 = ChebConv(hidden_dim, num_classes, num_hops)

    def reset_parameters(self):
        self.layer_1.reset_parameters()
        self.layer_2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layer_1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layer_2(x, edge_index)
        return F.log_softmax(x, dim=1)