import torch
from torch import nn


class Diffusion:
    def __init__(self, betas: tuple[float, float], n_steps: int):
        
        self.betas = betas
        
    def sample_forward(self, x_0, t):
        beta_min, beta_max = self.betas
        mean = x_0 * torch.exp(- 0.25 * t**2 * (beta_max - beta_min) - 0.5 * t * beta_min)
        std = torch.sqrt(1 - torch.exp(- 0.5 * t**2 * (beta_max - beta_min) - t * beta_min))
        return mean + torch.randn_like(x_0) * std
    
    def sample_backward(self, ):
        
