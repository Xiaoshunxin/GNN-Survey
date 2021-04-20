import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class Encoder(nn.Module):

    def __init__(self, input_dim, output_dim, model_type):
        super(Encoder, self).__init__()
        self.model_type = model_type
        self.conv_1 = GCNConv(input_dim, 2 * output_dim, cached=True)
        if model_type == "GAE":
            self.conv_2 = GCNConv(2 * output_dim, output_dim, cached=True)
        elif model_type == "VGAE":
            self.conv_mu = GCNConv(2 * output_dim, output_dim, cached=True)
            self.conv_logvar = GCNConv(2 * output_dim, output_dim, cached=True)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv_1(x, edge_index)
        x = F.relu(x)
        if self.model_type == "GAE":
            return self.conv_2(x, edge_index)
        elif self.model_type == "VGAE":
            return self.conv_mu(x, edge_index), self.conv_logvar(x, edge_index)
