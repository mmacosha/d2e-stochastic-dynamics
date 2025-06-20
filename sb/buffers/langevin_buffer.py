import math

import torch
from . import simple_buffer


class LangevinReplayBuffer(simple_buffer.ReplayBuffer):
    def __init__(self, size, grad_log_density, step_size, n_update_steps: float = 1e-3):
        super().__init__(size)
        self.step_size = step_size
        self.grad_log_density = grad_log_density
        self.n_update_steps = n_update_steps

    def run_langevin(self):
        dt = self.step_size
        buffer = torch.cat(self.buffer, dim=0)
        
        for _ in range(self.n_update_steps):
            z = torch.randn_like(buffer)
            grad = self.grad_log_density(buffer)
            buffer = buffer + grad * dt + z * math.sqrt(2 * dt)
        
        # if buffer.is_cuda:
        #     torch.cuda.empty_cache()

        self.buffer = list(buffer.split(1, 0))

    def sample(self, batch_size):
        self.run_langevin()
        sample =  super().sample(batch_size)
        return sample
