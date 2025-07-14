from typing import Callable

import sys
import math
import wandb
import torch
import functools

from tqdm.auto import trange
import matplotlib.pyplot  as plt

from . import simple_buffer

sys.path.append('external/hamiltorch')
import hamiltorch


def compute_grad(fn, x):
    x_ = x.detach().clone().requires_grad_(True)
    o = fn(x_)
    o.sum().backward()
    grad = x_.grad
    
    if grad.isnan().any():
        raise ValueError("Gradient contains NaN values.")

    if grad.isinf().any():
        raise ValueError("Gradient contains Inf values.")
    
    return o, grad


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


class LangevinReplayBuffer(simple_buffer.ReplayBuffer):
    _sampler = {'legacy', 'ula', 'ula2', 'mala', 'hmc'}
    def __init__(
            self, 
            buffer_size: int, 
            p1, 
            init_step_size: float, 
            num_langevin_steps: int,
            ema_lambda: float = 0,
            sampler: str = 'ula',
            noise_start_ration: float = 0.0,
            anneal_value: float = 1.0,
            hmc_freq: int = 4,
            device = 'cpu',
            beta_fn: Callable = None,
            log_hist: bool = False,
            *args, **kwargs
        ):
        super().__init__(buffer_size)
        self.log_hist = log_hist
        self.device = device
        self.lmbda = ema_lambda
        self.beta_fn = eval(beta_fn) if beta_fn else None

        self.anneal_value = anneal_value
        self.hmc_freq = hmc_freq
        self.noise_start_ration = noise_start_ration
        self.num_steps = num_langevin_steps
        
        self.sampler = sampler
        self.step_size = init_step_size
        
        self.log_density = p1.log_density
        self.reward = p1.reward

        self.langevin_step_counter = 0

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
        for i in trange(self.num_steps, desc='Langevin', leave=False):
            if self.sampler in {'legacy', 'ula', 'ula2'}:
                beta = self.beta_fn(i) if self.beta_fn else None
                density_fn = functools.partial(self.log_density, anneal_beta=beta)
                
                y, prev_z = ula_step(density_fn, x, dt, prev_z)
                x = self.lmbda * x + (1 - self.lmbda) * y
                prev_z = prev_z if self.sampler == 'ula2' else None
            
            elif self.sampler == 'mala':
                x = mala_step(self.log_density, x, dt)

            else:
                raise ValueError('Unknown method')
            
            dt *= anneal_alpha
            
            if wandb.run is not None:
                with torch.no_grad():
                    outputs = self.reward(x)
                    rwd, prc = self.reward.get_reward_and_precision(outputs=outputs)
                    
                    probas = outputs['all_probas'].mean(dim=0).cpu()
                    fig = plt.figure(figsize=(12, 8))
                    plt.xticks([i for i in range(10)])
                    plt.stem(probas)
                    plt.show()

                    log_dict = {
                        "metrics/langevin_precision": prc,
                        "metrics/langevin_mean_log_reward": rwd.log().mean(),
                        "langevin_step_counter": self.langevin_step_counter
                    }
                    
                    if self.log_hist:
                        log_dict.update({
                             "metrics/hist": wandb.Image(fig),
                        })
                    
                    wandb.log(log_dict)
                    plt.close('all')
            
            self.langevin_step_counter += 1
        
        return x
    
    def resample_propto_reward(self, x):
        with torch.no_grad():
            rwd = self.reward.reward(x)
            probas = rwd / rwd.sum()
            chosen_idx = torch.multinomial(
                probas, num_samples=probas.size(0), 
                replacement=False
            )
        return x[chosen_idx]

    def sample(self, batch_size, dim=None):
        if self.sampler == "legacy":
            if self.is_empty():
                raise ValueError(
                    "Buffer is empty. Cannot sample from ",
                    "an empty buffer in legacy mode."
            )
            buffer = torch.cat(self.buffer, dim=0)
            buffer = self.run_sampler(buffer)
            self.buffer = list(buffer.split(1, 0))
            x = super().sample(batch_size)
        
        else:
            if self.is_empty():
                shape = (batch_size, *dim) \
                    if isinstance(dim, (tuple, list)) else (batch_size, dim)
                x = torch.randn(*shape, device=self.device)
            else:
                x = super().sample(batch_size)
            
            if not self.is_empty() and self.noise_start_ration > 0:
                noise_size = int(batch_size * self.noise_start_ration)
                x[-noise_size:] = torch.randn(noise_size, *x.shape[1:], device=x.device)
            
            x = self.run_sampler(x)
            x = self.resample_propto_reward(x)
                
        return x
