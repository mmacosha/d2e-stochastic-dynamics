import random
import torch


class ReplayBuffer:
    def __init__(self, size, update_fraction=1.0):
        self.size = size
        self.update_fraction = update_fraction
        self.buffer = []

    @torch.no_grad()
    def sample(self, batch_size, *args, **kwargs):
        return torch.cat(random.choices(self.buffer, k=batch_size), dim=0)

    @torch.no_grad()
    def update(self, batch):
        if self.update_fraction < 1.0:
            mask = torch.rand(batch.size(0), 
                              device=batch.device) < self.update_fraction
            batch = batch[mask]

        self.buffer = self.buffer[-self.size + batch.size(0):]
        self.buffer.extend(batch.split(1, 0))

    def clear(self):
        self.buffer.clear()

    def is_empty(self) -> bool:
        return not self.buffer
