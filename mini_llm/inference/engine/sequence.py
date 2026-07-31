from enum import Enum


class Status(Enum):
    Waiting = "pending"
    Running = "running"
    Completed = "completed"
    
class Sequence:
    def __init__(self, request_id):
        self.request_id = request_id
        self.token_ids = []
        self.status = Status.Waiting

    def add_token(self, token_id):
        self.token_ids.append(token_id)
    
    @property
    def length(self):
        return len(self.token_ids)
    