from .sequence import Status

class Scheduler:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.wq = []
        self.rq = []
        self.cq = []
    
    def schedule(self):
        self.cq += [seq for seq in self.rq if seq.status == Status.Completed]
        self.rq = [seq for seq in self.rq if seq.status == Status.Running]

        while self.wq and len(self.rq) < self.batch_size:
            new_seq = self.wq.pop(0)
            new_seq.status = Status.Running
            self.rq.append(new_seq)
        
        return self.rq
    
    def has_unfinished_requests(self):
        return len(self.wq) > 0 or len(self.rq) > 0

        
        
        
        
