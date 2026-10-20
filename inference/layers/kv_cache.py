import torch

class LayerKVCache:
    def __init__(self, max_seq_len, dim):
        self.max_seq_len = max_seq_len
        self.dim = dim
        self.k = None
        self.v = None

    # k size: batch_size, seq_len, dim
    # v size: batch_size, seq_len, dim
    def append_kv(self, k, v):
        if self.k is None:
            self.k = k
            self.v = v
        else:
            self.k = torch.cat([self.k, k], dim = 1)
            self.v = torch.cat([self.v, v], dim = 1)
        return self.k, self.v
    
    def get_kv(self, seq_len):
        return self.k[:, :seq_len], self.v[:, :seq_len]
    
    def clear(self):
        self.k = None
        self.v = None
        return self.k, self.v
    
    def __len__(self):
        return 0 if self.k is None else self.k.shape[1]

class KVCache:
    def __init__(self, n_layers, max_seq_len, dim):
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.dim = dim
        self.kv_caches = [LayerKVCache(max_seq_len, dim) for _ in range(n_layers)]

    def append_kv(self, layer_idx, k, v):
        self.kv_caches[layer_idx].append_kv(k, v)
        return self.kv_caches[layer_idx].k, self.kv_caches[layer_idx].v
    
    def get_kv(self, layer_idx, seq_len):
        return self.kv_caches[layer_idx].get_kv(seq_len)
    
    def clear(self, layer_idx):
        self.kv_caches[layer_idx].clear()
    
    def clear_all(self):
        for i in range(self.n_layers):
            self.kv_caches[i].clear()
    
    def __len__(self):
        return [len(layer) for layer in self.kv_caches]

