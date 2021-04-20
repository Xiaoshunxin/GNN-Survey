import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GATConv


class GAT(nn.Module):

    def __init__(self, input_dim, hidden_dim, heads, dropout, num_classes):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.layer_1 = GATConv(input_dim, hidden_dim, heads, dropout=dropout)
        self.layer_2 = GATConv(hidden_dim * heads, num_classes, 8, dropout=dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layer_1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layer_2(x, edge_index)
        return F.log_softmax(x, dim=1)


class GAT_PPI(nn.Module):

    def __init__(self, num_features, num_classes):
        super(GAT_PPI, self).__init__()
        self.conv1 = GATConv(num_features, 256, heads=4)
        self.lin1 = torch.nn.Linear(num_features, 4 * 256)
        self.conv2 = GATConv(4 * 256, 256, heads=4)
        self.lin2 = torch.nn.Linear(4 * 256, 4 * 256)
        self.conv3 = GATConv(4 * 256, num_classes, heads=6, concat=False)
        self.lin3 = torch.nn.Linear(4 * 256, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.elu(self.conv1(x, edge_index) + self.lin1(x))
        x = F.elu(self.conv2(x, edge_index) + self.lin2(x))
        x = self.conv3(x, edge_index) + self.lin3(x)
        return x