from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import SGConv


class SGC(nn.Module):

    def __init__(self, input_dim, num_classes, K):
        super(SGC, self).__init__()
        self.conv1 = SGConv(input_dim, num_classes, K=K, cached=True)

    def reset_parameters(self):
        self.conv1.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        return F.log_softmax(x, dim=1)
