import math

import torch
from . import simple_buffer


class LangevinReplayBuffer(simple_buffer.ReplayBuffer):
    def __init__(self, size, grad_energy, step_size, n_update_steps: float = 1e-3):
        super().__init__(size)
        self.step_size = step_size
        self.grad_energy = grad_energy
        self.n_update_steps = n_update_steps

    def run_langevin(self):
        dt = self.step_size
        buffer = torch.cat(self.buffer, dim=0)
        
        for _ in range(self.n_update_steps):
            z = torch.randn_like(buffer)
            buffer += self.grad_energy(buffer) * dt + z * math.sqrt(2 * dt)
        
        self.buffer = list(buffer.split(1, 0))

    def sample(self, batch_size):
        self.run_langevin()
        sample =  super().sample(batch_size)
        return sample
