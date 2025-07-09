import math
import torch
from torch import nn

from sb.nn.utils import ModelOutput


def vp_beta_fn(t, beta_min, beta_max):
    return beta_min + (beta_max - beta_min) * t

def ve_sigma_fn(t, sigma_min, sigma_max):
    return sigma_min * (sigma_max / sigma_min) ** t


class DSFixedBackward(nn.Module):
    def __init__(self, bwd_process_type, betas, sigmas):
        super().__init__()
        self.type = bwd_process_type
        self.beta_min, self.beta_max = betas
        self.sigma_min, self.sigma_max = sigmas

    def drift(self, x, t):
        if self.type == 'vp':
            beta_t = vp_beta_fn(t, self.beta_min, self.beta_max)
            return -0.5 * beta_t[..., None] * x
        elif self.type == 've':
            return 0
        
    def log_var(self, x, t):
        if self.type == 'vp':
            beta_t = vp_beta_fn(t, self.beta_min, self.beta_max)
            return torch.log(beta_t[..., None])
        elif self.type == 've':
            sigma_t = ve_sigma_fn(t, self.sigma_min, self.sigma_max)
            sigma_t = sigma_t * 2 * torch.log(self.sigma_max / self.sigma_min)
            return 2 * torch.log(sigma_t[..., None])

    def forward(self, x, t):
        drift = self.drift(x, t)
        sigma = self.sigma(x, t)
        log_var = torch.log(sigma ** 2)
        return ModelOutput(drift=drift, log_var=log_var)
