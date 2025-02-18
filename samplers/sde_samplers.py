import math

import torch
from torch import nn
from torch import Tensor

import utils


class VPSDE:
    t_max = 1.0
    
    def __init__(self, beta_min, beta_max, n_steps):
        self.beta_min = beta_min #* 1000 / n_steps
        self.beta_max = beta_max #* 1000 / n_steps
        self.n_steps = n_steps

    def get_beta_t(self, t):
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def get_mean_coeff_and_std(self, t):
        log_coeff = - 0.25 * t**2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        mean = torch.exp(log_coeff)
        std = torch.sqrt(1 - torch.exp(2.0 * log_coeff))
        return mean, std

    def sample_x_t_from_x_0(self, x_0, t, noise: Tensor | None = None):
        if noise is None:
            noise = torch.randn_like(x_0)
        
        mean, std = self.get_mean_coeff_and_std(t)
        return utils.extract_into_tensor(mean, x_0.shape) * x_0 + noise * utils.extract_into_tensor(std, x_0.shape)

    def get_sde_params(self, x_t, t):
        '''
            Compute coefficient for SDE:
            dx_t = f_t * dt + g_t * dW_t
        '''
        beta_t = utils.extract_into_tensor(self.get_beta_t(t), x_t.shape)
        drift = - 0.5 * beta_t * x_t
        
        return drift, beta_t.sqrt()
    
    def get_reverse_sde_params(self, score, x_t, t, ode_flow: bool = False):
        drift, diff = self.get_sde_params(x_t, t)
        rev_drift = drift - diff**2 * score * (0.5 if ode_flow else 1.0)
        
        return rev_drift, diff

    def compute_loss(self, model, x_0, t, learn_score: bool = True):
        noise = torch.randn_like(x_0)
        
        time = torch.maximum(t, torch.as_tensor(1e-3))
        x_t = self.sample_x_t_from_x_0(x_0, time, noise)
        
        if learn_score:
            mean_coeff, std = self.get_mean_coeff_and_std(time)
            mean_coeff = utils.extract_into_tensor(mean_coeff, x_0.shape)
            std = utils.extract_into_tensor(std, x_0.shape)
            
            score = - (x_t - mean_coeff * x_0) / std**2
            loss = nn.functional.mse_loss(model(x_t, t), score)
        else:
            loss = nn.functional.mse_loss(model(x_t, t), noise)

        return loss

    def score_from_noise(self, model, x_t, t):
        eps_pred = model(x_t, t)
        _, std = self.get_mean_coeff_and_std(t)
        return - eps_pred / utils.extract_into_tensor(std, eps_pred.shape)

    def make_fwd_step(self, x_t, t, dt: float | None = None):
        '''
            Noise for forward process.
        '''
        z = torch.randn_like(x_t)
        drift, sigma = self.get_sde_params(x_t, t)
        if dt is None:
            dt = self.t_max / self.n_steps
        
        return drift * dt + sigma * math.sqrt(dt) * z

    @torch.no_grad
    def make_bwd_step(self, score, x_t, t, dt: float | None = None, ode_flow: bool = False):
        '''
            Denoising backward process.
        '''
        z = torch.randn_like(x_t)
        drift, diff = self.get_reverse_sde_params(score, x_t, t, ode_flow)
        if dt is None:
            dt = - self.t_max / self.n_steps
        
        dx = drift * dt
        if not ode_flow:
            dx = dx + diff * math.sqrt(-dt) * z
        
        return dx
    

class VESDE:
    t_max = 1.0
    
    def __init__(self, beta_min, beta_max, n_steps):
        self.beta_min = beta_min #* 1000 / n_steps
        self.beta_max = beta_max #* 1000 / n_steps
        self.n_steps = n_steps

    def get_beta_t(self, t):
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def get_mean_coeff_and_std(self, t):
        log_coeff = - 0.25 * t**2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        mean = torch.exp(log_coeff)
        std = torch.sqrt(1 - torch.exp(2.0 * log_coeff))
        return mean, std

    def sample_x_t_from_x_0(self, x_0, t, noise: Tensor | None = None):
        if noise is None:
            noise = torch.randn_like(x_0)
        
        mean, std = self.get_mean_coeff_and_std(t)
        return utils.extract_into_tensor(mean, x_0.shape) * x_0 + noise * utils.extract_into_tensor(std, x_0.shape)

    def get_sde_params(self, x_t, t):
        '''
            Compute coefficient for SDE:
            dx_t = f_t * dt + g_t * dW_t
        '''
        beta_t = utils.extract_into_tensor(self.get_beta_t(t), x_t.shape)
        drift = - 0.5 * beta_t * x_t
        
        return drift, beta_t.sqrt()
    
    def get_reverse_sde_params(self, score, x_t, t, ode_flow: bool = False):
        drift, diff = self.get_sde_params(x_t, t)
        rev_drift = drift - diff**2 * score * (0.5 if ode_flow else 1.0)
        
        return rev_drift, diff

    def compute_loss(self, model, x_0, t, learn_score: bool = True):
        noise = torch.randn_like(x_0)
        
        time = torch.maximum(t, torch.as_tensor(1e-3))
        x_t = self.sample_x_t_from_x_0(x_0, time, noise)
        
        if learn_score:
            mean_coeff, std = self.get_mean_coeff_and_std(time)
            mean_coeff = utils.extract_into_tensor(mean_coeff, x_0.shape)
            std = utils.extract_into_tensor(std, x_0.shape)
            
            score = - (x_t - mean_coeff * x_0) / std**2
            loss = nn.functional.mse_loss(model(x_t, t), score)
        else:
            loss = nn.functional.mse_loss(model(x_t, t), noise)

        return loss

    def score_from_noise(self, model, x_t, t):
        eps_pred = model(x_t, t)
        _, std = self.get_mean_coeff_and_std(t)
        return - eps_pred / utils.extract_into_tensor(std, eps_pred.shape)

    def make_fwd_step(self, x_t, t, dt: float | None = None):
        '''
            Noise for forward process.
        '''
        z = torch.randn_like(x_t)
        drift, sigma = self.get_sde_params(x_t, t)
        if dt is None:
            dt = self.t_max / self.n_steps
        
        return drift * dt + sigma * math.sqrt(dt) * z

    @torch.no_grad
    def make_bwd_step(self, score, x_t, t, dt: float | None = None, ode_flow: bool = False):
        '''
            Denoising backward process.
        '''
        z = torch.randn_like(x_t)
        drift, diff = self.get_reverse_sde_params(score, x_t, t, ode_flow)
        if dt is None:
            dt = - self.t_max / self.n_steps
        
        dx = drift * dt
        if not ode_flow:
            dx = dx + diff * math.sqrt(-dt) * z
        
        return dx