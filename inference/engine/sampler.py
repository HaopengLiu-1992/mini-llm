import torch 

class Sampler:
    def __init__(self):
        pass

    def sample(self, logits):
        next_token_ids = torch.argmax(logits, dim=-1)
        return next_token_ids.tolist()

class TopKSamplerWithTemperature(Sampler):
    def __init__(self, k, temperature):
        self.k = k
        self.temperature = temperature

    def sample(self, logits):
        logits = logits / self.temperature
        top_k_logits, top_k_indices = torch.topk(logits, self.k)
        probs = torch.softmax(top_k_logits, dim=-1)
        idx_in_topk = torch.multinomial(probs, num_samples=1)
        next_token = top_k_indices.gather(-1, idx_in_topk).squeeze(-1)
        return next_token.tolist()