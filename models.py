import torch
import torch.nn as nn

class MyLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        nn.init.kaiming_uniform_(self.weight, a=0.01)
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / torch.sqrt(torch.tensor(fan_in, dtype=torch.float32))
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, X):
        if self.bias is not None:
            return torch.matmul(X, self.weight) + self.bias
        else:
            return torch.matmul(X, self.weight)