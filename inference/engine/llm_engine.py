from .model_runner import ModelRunner
from .sampler import Sampler
from .scheduler import Scheduler
from .sequence import Sequence, Status

class LLMEngine:
    def __init__(self, model, max_seq_len, batch_size, end_token_id):
        self.model_runner = ModelRunner(model)
        self.sampler = Sampler()
        self.scheduler = Scheduler(batch_size)
        self.max_seq_len = max_seq_len
        self.end_token_id = end_token_id
    
    def add_request(self, request_id, prompt_token_ids):
        seq = Sequence(request_id)
        seq.token_ids += prompt_token_ids
        if seq.length >= self.max_seq_len or seq.length == 0:
            raise ValueError('Sequence length must be between 1 and max_seq_len')
        
        self.scheduler.wq.append(seq)
    
    def step(self):
        batch = self.scheduler.schedule()
        if not batch:
            return []
        batch_token_ids = [seq.token_ids for seq in batch]
        logits = self.model_runner.run(batch_token_ids)
        next_token_ids = self.sampler.sample(logits)
        outputs = []

        for seq, token_id in zip(batch, next_token_ids):
            seq.add_token(token_id)
            if token_id == self.end_token_id:
                seq.status = Status.Completed
            if seq.length >= self.max_seq_len:
                seq.status = Status.Completed
            outputs.append({
                'request_id': seq.request_id,
                'token_id': token_id,
                'done': seq.status == Status.Completed,
            })
        return outputs

    def generate(self):
        all_outputs = []
        while self.scheduler.has_unfinished_requests():
            outputs = self.step()
            all_outputs += outputs
        return all_outputs
        