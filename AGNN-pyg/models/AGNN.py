import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import AGNNConv


class AGNN(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_classes, dropout):
        super(AGNN, self).__init__()
        self.dropout = dropout
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        self.layer_2 = AGNNConv(requires_grad=True)
        self.layer_3 = AGNNConv(requires_grad=True)
        self.layer_4 = nn.Linear(hidden_dim, num_classes)

        self.layer_5 = AGNNConv(requires_grad=True)
        self.layer_6 = AGNNConv(requires_grad=True)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.layer_2(x, edge_index)
        x = self.layer_3(x, edge_index)
        x = self.layer_5(x, edge_index)
        x = self.layer_6(x, edge_index)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layer_4(x)
        return F.log_softmax(x, dim=1)


class AGNN_PPI(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_classes, dropout):
        super(AGNN_PPI, self).__init__()
        self.dropout = dropout
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        self.layer_2 = AGNNConv(requires_grad=True)
        self.layer_3 = AGNNConv(requires_grad=True)
        self.layer_4 = nn.Linear(hidden_dim, num_classes)
        self.layer_5 = AGNNConv(requires_grad=True)
        self.layer_6 = AGNNConv(requires_grad=True)

        self.linear_1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear_3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear_4 = nn.Linear(hidden_dim, hidden_dim)


    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layer_1(x)
        x = F.elu(self.layer_2(x, edge_index) + self.linear_1(x))
        x = F.elu(self.layer_3(x, edge_index) + self.linear_2(x))
        x = F.elu(self.layer_5(x, edge_index) + self.linear_3(x))
        x = F.elu(self.layer_6(x, edge_index) + self.linear_4(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layer_4(x)
        return x
