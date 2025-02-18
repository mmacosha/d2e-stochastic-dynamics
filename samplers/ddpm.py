import torch
from torch import nn

import utils


class DDPM:
    def __init__(self, schedule, n_steps):
        self.n_steps = n_steps
        
        self.betas = schedule
        self.alphas = 1 - self.betas
        
        self.cumprod_alphas = self.alphas.cumprod(0)
        self.prev_cumprod_alphas = torch.cat([torch.ones(1), self.cumprod_alphas[:-1]])
        
        self._1m_cumprod_alphas = 1 - self.cumprod_alphas
        self._1m_prev_cumprod_alphas = 1 - self.prev_cumprod_alphas
        
        self.post_x_0_coeff = self.prev_cumprod_alphas.sqrt() * self.betas / (self._1m_cumprod_alphas)
        self.post_x_t_coeff = self.alphas.sqrt() * (self._1m_prev_cumprod_alphas) / (self._1m_cumprod_alphas)
        self.post_var = (self._1m_prev_cumprod_alphas) * self.betas / (self._1m_cumprod_alphas)
    
    def sample_x_t_from_x_0(self, x_0, t, noise=None):
        mean_coeff = utils.extract_into_tensor(self.cumprod_alphas[t].sqrt(), x_0.shape)
        std = utils.extract_into_tensor(self._1m_cumprod_alphas[t].sqrt(), x_0.shape)
        
        if noise is None:
            noise = torch.randn_like(x_0)
            
        return mean_coeff * x_0 + std * noise
    
    def predict_x_0_from_noise(self, x_t, t, pred_eps):
        std = utils.extract_into_tensor(self._1m_cumprod_alphas[t].sqrt(), x_t.shape)
        mean_coeff = utils.extract_into_tensor(self.cumprod_alphas[t].sqrt(), x_t.shape)
        
        x_0_pred = (x_t - std * pred_eps) / mean_coeff
        return x_0_pred.float()
        
    def sample_x_t_m1_from_x_t_and_x_0(self, x_t, t, pred_eps):
        x_0_coeff = utils.extract_into_tensor(self.post_x_0_coeff[t], x_t.shape)
        x_t_coeff = utils.extract_into_tensor(self.post_x_t_coeff[t], x_t.shape)
        var = utils.extract_into_tensor(self.post_var[t], x_t.shape)
        
        x_0_pred = self.predict_x_0_from_noise(x_t, t, pred_eps)
        mean = x_0_coeff * x_0_pred + x_t_coeff * x_t
        
        non_zero_mask = utils.extract_into_tensor((t != 0), x_t.shape)
        return {
            'sample': mean + non_zero_mask * torch.randn_like(x_t) * var.sqrt(),
            'mean': mean,
            'std': var.sqrt()
        }
        
    def sample_x_t_m1_simpler(self, x_t, t, pred_eps):
        alpha_t = utils.extract_into_tensor(self.alphas[t], x_t.shape)
        sqrt_1m_cumprod_alpha_t = utils.extract_into_tensor(self._1m_cumprod_alphas[t], x_t.shape).sqrt()
        beta_t = utils.extract_into_tensor(self.betas[t], x_t.shape)
        
        mean = 1 / alpha_t * (x_t - pred_eps * beta_t / sqrt_1m_cumprod_alpha_t)
        sigma = utils.extract_into_tensor(self.post_var[t].sqrt(), x_t.shape)
        
        non_zero_mask = utils.extract_into_tensor((t != 0), x_t.shape)
        return {
            'sample': mean + non_zero_mask * torch.randn_like(x_t) * sigma,
            'mean': mean,
            'std': sigma
        }
        
    def ansesterial_sampling_step(self, x_t, t, model):
        sqrt_1m_cumprod_alpha_t = utils.extract_into_tensor(self._1m_cumprod_alphas[t], x_t.shape).sqrt()
        score = - model(x_t, t) / (sqrt_1m_cumprod_alpha_t + 1e-6)
        
        beta_t = utils.extract_into_tensor(self.betas[t], x_t.shape)
        x_prev = 1 / (1 - beta_t) * (x_t + beta_t * score) + beta_t.sqrt() * torch.randn_like(x_t)
        
        return {
            'sample': x_prev
        }
        
    def make_ddim_step(self, model, x_t, t, ddim_sigma):
        pred_eps = model(x_t, t)
        x_0_pred = self.predict_x_0_from_noise(x_t, t, pred_eps)
        
        mean_coeff = utils.extract_into_tensor(self.prev_cumprod_alphas[t].sqrt(), x_t.shape)
        std = utils.extract_into_tensor((self._1m_prev_cumprod_alphas[t] - ddim_sigma**2).sqrt(), x_t.shape)
        
        x_t_prev = mean_coeff * x_0_pred + std * pred_eps + torch.randn_like(x_t) * ddim_sigma
        return x_t_prev
    
    def make_ddpm_step(self, model, x_t, t):
        pred_eps = model(x_t, t)
        x_t_prev = self.sample_x_t_m1_from_x_t_and_x_0(x_t, t, pred_eps)['sample']
        return x_t_prev
    
    def make_simple_ddpm_step(self, model, x_t, t):
        pred_eps = model(x_t, t)
        x_t_prev = self.sample_x_t_m1_simpler(x_t, t, pred_eps)['sample']
        return x_t_prev
        
    @torch.no_grad()
    def sample(self, model, shape, 
               sample_with: str = 'ddpm',
               ddim_sigma: float = 0.0
               ):
        x_t = torch.randn(shape)
        for t in range(self.n_steps -1, -1, -1):
            t = torch.ones(x_t.size(0), dtype=torch.long) * t
            if sample_with == 'ddpm':
                x_t = self.make_ddpm_step(model, x_t, t)
            elif sample_with == 'ddim':
                x_t = self.make_ddim_step(model, x_t, t, ddim_sigma)
            elif sample_with == 'simple_ddpm':
                x_t = self.make_simple_ddpm_step(model, x_t, t)
            elif sample_with == 'ansesterial':
                x_t = self.ansesterial_sampling_step(x_t, t, model)['sample']    
            else:
                raise ValueError(f"Unknown sampling method {sample_with}")
            
        return x_t

    def train_loss(self, model, x_0, t):
        noise = torch.randn_like(x_0)
        x_t = self.sample_x_t_from_x_0(x_0, t, noise)
        loss = nn.functional.mse_loss(model(x_t, t), noise)
        return loss