from torch import nn
import torch.nn.functional as F

class MyLinear(nn.Module):

    def __init__(self, hidden_dim, num_classes):
        super(MyLinear, self).__init__()
        self.linear = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.linear(x)
        return F.log_softmax(x, dim=1)