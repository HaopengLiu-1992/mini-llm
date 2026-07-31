import torch
from torch import nn

class FFN(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim)
        self.gelu = nn.GELU(approximate="tanh")
        self.w2 = nn.Linear(hidden_dim, dim)
    
    def forward(self, x: torch.Tensor):
        return self.w2(self.gelu(self.w1(x)))