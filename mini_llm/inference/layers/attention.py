import torch
import torch.nn as nn

class MHA(nn.Module):
    def __init__(self, dim, n_head):
        super().__init__()
        self.dim = dim
        self.n_head = n_head
        assert dim % n_head == 0, "dim must be divisible by n_head"
        self.head_dim = dim // n_head
        self.wq = nn.Linear(dim, dim)
        self.wk = nn.Linear(dim, dim)
        self.wv = nn.Linear(dim, dim)
        self.wo = nn.Linear(dim, dim)
    
    def scaled_dot_product(self, q, k, v, mask = None):
        attn_score = q @ k.transpose(-2, -1) / (self.head_dim ** 0.5)
        if mask is not None:
            attn_score = attn_score.masked_fill(mask, float("-inf"))
        attn_prob = torch.softmax(attn_score, dim = -1)
        return attn_prob @ v
    
    def forward(self, q, k, v, mask = None):
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        b, q_len, _ = q.shape
        _, k_len, _ = k.shape
        _, v_len, _ = v.shape
        q = q.view(b, q_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(b, k_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(b, v_len, self.n_head, self.head_dim).transpose(1, 2)
        attn = self.scaled_dot_product(q, k, v, mask)
        attn = attn.transpose(1, 2).contiguous().view(b, q_len, self.dim)
        return self.wo(attn)