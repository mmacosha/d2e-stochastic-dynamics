from typing import Callable
import math

import torch
import hamiltorch

from . import simple_buffer


class LangevinReplayBuffer(simple_buffer.ReplayBuffer):
    _sampler = {'legacy', 'ula', 'ula2', 'mala', 'hmc'}
    def __init__(
            self, 
            size: int, 
            p1, 
            init_step_size: float, 
            num_steps: int,
            sampler: str = 'ula',
            noise_start_ration: float = 0.0,
            anneal_value: float = 1.0,
            hmc_freq: int = 4,
            reward_threshold: float = 0.8,
        ):
        super().__init__(size)
        self.anneal_value = anneal_value
        self.hmc_freq = hmc_freq
        self.noise_start_ration = noise_start_ration
        self.num_steps = num_steps
        self.reward_threshold = reward_threshold
        
        self.sampler = sampler
        self.step_size = init_step_size
        
        self.log_density = p1.log_density
        self.reward = p1.reward

    def run_sampler(self, x):
        if self.sampler == 'hmc':
            x0 = x[0]
            size = x.size(0)

            def _log_gensity(x):
                x = x.reshape(1, -1)
                density = self.log_density(x)
                return density[0]

            xt = hamiltorch.sample(
                log_prob_func=_log_gensity, 
                params_init=x0,  
                num_samples=(size * self.hmc_freq), 
                step_size=self.init_step_size, 
                num_steps_per_sample=15,
                sampler=hamiltorch.Sampler.HMC, 
                integrator=hamiltorch.Integrator.IMPLICIT,
            )
            return torch.stack(xt[::self.hmc_freq])
 
        prev_z = None
        dt = self.step_size
        anneal_alpha = math.exp(math.log(self.anneal_value) / self.num_steps)
        for _ in range(self.num_steps):
            if self.sampler in {'legacy', 'ula', 'ula2'}:
                x, prev_z = ula_step(self.log_density, x, dt, prev_z)
                prev_z = prev_z if self.sampler == 'ula2' else None
            
            elif self.sampler == 'mala':
                x = mala_step(self.log_density, x, dt)

            else:
                raise ValueError('Unknown method')

            dt *= anneal_alpha
        
        return x
    
    @torch.no_grad()
    def update(self, batch):
        if self.reward_threshold > 0.0:
            rewards = self.reward(batch)
            batch = batch[rewards > self.reward_threshold]

        self.buffer = self.buffer[-self.size + batch.size(0):]
        self.buffer.extend(batch.split(1, 0))

    def sample(self, batch_size):
        if self.sampler == "legacy":
            buffer = torch.cat(self.buffer, dim=0)
            buffer = self.run_sampler(buffer)
            self.buffer = list(buffer.split(1, 0))
            x = super().sample(batch_size)
        
        else:
            x = super().sample(batch_size)
            noise_size = int(batch_size * self.noise_start_ration)
            if noise_size > 0:
                x[-noise_size:] = torch.randn(noise_size, x.size(1), device=x.device)
            x = self.run_sampler(x)
                
        return x


def compute_grad(fn, x):
    x_ = x.detach().clone().requires_grad_(True)
    o = fn(x_)
    o.sum().backward()
    return o, x_.grad


def mala_correction(x, logp_x, grad_logp_x, y, logp_y, grad_logp_y, dt):
    adj = logp_y - logp_x + \
          (y - x - dt * grad_logp_x).pow(2).sum(1) / (4 * dt) - \
          (x - y - dt * grad_logp_y).pow(2).sum(1) / (4 * dt)

    adj = torch.minimum(torch.ones_like(adj), adj.exp())
    return torch.rand_like(adj) < adj


def ula_step(fn, x, dt, prev_z=None):
    _, grad = compute_grad(fn, x)

    curr_z = torch.randn_like(x)
    z = curr_z if prev_z is None else (curr_z + prev_z) / 2
    prev_z = curr_z
    
    return x + grad * dt + math.sqrt(2 * dt) * z, prev_z


def mala_step(fn, x, dt):
    logp_x, grad_logp_x = compute_grad(fn, x)

    z = torch.randn_like(x)
    y = x + grad_logp_x * dt + math.sqrt(2 * dt) * z

    logp_y, grad_logp_y = compute_grad(fn, y)

    adj = mala_correction(x, logp_x, grad_logp_x, 
                            y, logp_y, grad_logp_y, 
                            dt)
    x[adj] = y[adj]
    return x
