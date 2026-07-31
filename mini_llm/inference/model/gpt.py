import torch
from torch import nn

from ..layers.attention import MHA
from ..layers.ffn import FFN

class GPT2TransformerLayer(nn.Module):
    def __init__(self, n_heads, dim, hidden_dim):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.mha = MHA(dim, n_heads)
        self.ffn = FFN(dim, hidden_dim)
        self.layer_norm1 = nn.LayerNorm(dim)
        self.layer_norm2 = nn.LayerNorm(dim)
        
    
    def forward(self, x, mask = None):
        residual = x
        x = self.layer_norm1(x)
        attn = self.mha(x, x, x, mask)
        x = residual + attn

        residual = x
        x = self.layer_norm2(x)
        ffn = self.ffn(x)
        x = residual + ffn
        return x
    
class GPT2(nn.Module):
    def __init__(self, vocab_size, max_seq_len, n_heads, dim, hidden_dim, n_layers):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.n_heads = n_heads
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.transformer_layers = nn.ModuleList([
            GPT2TransformerLayer(n_heads, dim, hidden_dim)
            for _ in range(n_layers)
        ])
        self.layer_norm = nn.LayerNorm(dim)
        self.lm = nn.Linear(dim, vocab_size)
        self.wte = nn.Embedding(num_embeddings=vocab_size, embedding_dim=dim)
        self.wpe = nn.Embedding(num_embeddings=max_seq_len, embedding_dim=dim)
    

    # padding_mask bool = [batch_size, seq_len] True -> needs padding
    def forward(self, input_token_ids, padding_mask = None):
        # input_token_ids: [batch_size, seq_len]
        # padding_mask: [batch_size, seq_len]
        batch_size, seq_len = input_token_ids.shape

        device = input_token_ids.device
        position_ids = torch.arange(seq_len, device=device)

        # positional encoding
        x = self.wte(input_token_ids)   
        x = x + self.wpe(position_ids)
        
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
            diagonal=1,
        )

        if padding_mask is not None:
            mask =  causal_mask[None, None, :, :]| padding_mask[:, None, None, :]
        else:
            mask = causal_mask
            

        for layer in self.transformer_layers:
            x = layer(x, mask)
        
        x = self.layer_norm(x)
        logits = self.lm(x)
        return logits
