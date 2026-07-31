import torch
from torch.nn.utils.rnn import pad_sequence

class ModelRunner:
    def __init__(self, model, pad_token_id=0):
        self.model = model
        self.pad_token_id = pad_token_id
        self.device = next(
            model.parameters()
        ).device
        self.model.eval()

    @torch.no_grad()
    def run(self, batch_token_ids):
        if len(batch_token_ids) == 0:
            return []
        
        if any(len(token_ids) == 0 for token_ids in batch_token_ids):
            raise ValueError("Sequence ca nnot be empty")

        token_tensors = [torch.tensor(token_ids,  dtype=torch.long) for token_ids in batch_token_ids]

        input_ids = pad_sequence(
            token_tensors,
            batch_first=True,
            padding_value=self.pad_token_id,
        ).to(self.device)
        
        batch_size, max_seq_len = input_ids.shape
        
        lengths = torch.tensor([len(token_ids) for token_ids in batch_token_ids],
            dtype=torch.long,
            device=self.device,
        )

        positions = torch.arange(max_seq_len, device=self.device)

        padding_mask = (positions[None, :] >= lengths[:, None])

        logits = self.model(input_ids, padding_mask=padding_mask)
        batch_indices = torch.arange(batch_size, device=self.device)
        last_logits = logits[batch_indices, lengths - 1, :]
        return last_logits