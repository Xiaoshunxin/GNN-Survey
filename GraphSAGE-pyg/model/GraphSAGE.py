import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import SAGEConv


class GraphSAGE(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_classes, normalize):
        super(GraphSAGE, self).__init__()
        self.layer_1 = SAGEConv(input_dim, hidden_dim, normalize)
        self.layer_2 = SAGEConv(hidden_dim, num_classes, normalize)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, training=self.training)
        x = self.layer_1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.layer_2(x, edge_index)
        return F.log_softmax(x, dim=1)


class GraphSAGE_PPI(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_classes, normalize):
        super(GraphSAGE_PPI, self).__init__()
        self.layer_1 = SAGEConv(input_dim, hidden_dim, normalize)
        self.linear_1 = torch.nn.Linear(input_dim, hidden_dim)
        self.layer_2 = SAGEConv(hidden_dim, num_classes, normalize)
        self.linear_2 = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.elu(self.layer_1(x, edge_index) + self.linear_1(x))
        x = self.layer_2(x, edge_index) + self.linear_2(x)
        return x
