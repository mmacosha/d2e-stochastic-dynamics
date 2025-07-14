import torch

from .langevin_buffer import LangevinReplayBuffer


class DecoupledLangevinBuffer(LangevinReplayBuffer):
    def __init__(self, langevin_freq: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.langevin_freq = langevin_freq
        self._counter = 0
        self.langevin_buffer = []

        self.buffer_reward_probas = None

    def sample(self, batch_size, *args, **kwargs):
        if self.sampler == "legacy":
            raise ValueError(
                "DecoupledLangevinBuffer does not support legacy sampling."
            )

        chosen_idx = torch.multinomial(
            self.buffer_reward_probas, num_samples=batch_size, 
            replacement=False
            )
        return self.langevin_buffer[chosen_idx]


    def run_langevin(self, dim=None):
        if self._counter % self.langevin_freq != 0:
            self._counter += 1
            return
        
        self._counter += 1
        if self.is_empty():
            shape = (self.size, *dim) \
                if isinstance(dim, (tuple, list)) else (self.size, dim)
            x = torch.randn(shape, device=self.device)
        else:
            x = torch.cat(self.buffer, dim=0)
            self.buffer.clear()
            
            if self.noise_start_ration > 0:
                noise_size = int(self.size * self.noise_start_ration)
                x[:noise_size] = torch.randn(noise_size, *x.shape[1:], device=x.device)
        
        x = self.run_sampler(x)
        
        self.langevin_buffer = x
        
        self.buffer_reward_probas = \
            torch.ones(self.size, device=self.device) / self.size
