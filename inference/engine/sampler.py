import torch 

class Sampler:
    def __init__(self):
        pass

    def sample(self, logits):
        next_token_ids = torch.argmax(logits, dim=-1)
        return next_token_ids.tolist()