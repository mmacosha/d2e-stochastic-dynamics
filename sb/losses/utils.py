import ot
import numpy as np
import math
import torch

from . import utils


def get_mean_log_var(model, x, t, dt, base_var: float = 2.0,
                     return_drift: bool = False):
    log_var = torch.log(torch.ones_like(x) * base_var * dt)
    output = model(x, t)

    if output.contains('log_var'):
        log_var = log_var + output.log_var
        if log_var.isnan().any():
            raise ValueError("Log var is Nan")
    
    if output.drift.isnan().any():
        raise ValueError("Drift is Nan")
    
    if return_drift:
        return output.drift, log_var

    mean = x + output.drift * dt
    return mean, log_var


def get_model_outputs(model, x, t, dt, base_var):
    output = model(x, t)
    log_var = torch.log(torch.ones_like(x) * base_var * dt)
    
    if output.contains('log_var'):
        log_var = log_var + output.log_var
    
    return output.drift, log_var.exp().sqrt()


def log_normal_density(x, mean, log_var):
    """Compute log dencity normal distribution."""
    batch_size = x.size(0)
    x = x.view(batch_size, -1)
    mean = mean.view(batch_size, -1)
    log_var = log_var.view(batch_size, -1)
    return - 0.5 * (log_var + torch.exp(- log_var) * (mean - x).pow(2)).sum(-1)


def log_normal_density_v2(x, mean, std):
    batch_size = x.size(0)
    x, mean, std = (tensor.view(batch_size, -1) for tensor in (x, mean, std))
    return - (std.log() + (x - mean).pow(2) / (2 * std.pow(2))).sum(-1)


def compute_z_div_z(model, x, t, dt, var):
    x.requires_grad_(True)
    z, log_var = utils.get_mean_log_var(model, x, t, dt, var, return_drift=True)
    g = log_var.exp().sqrt()
    e = torch.randn_like(x)
    
    z_div = torch.autograd.grad(
        g * z, x, grad_outputs=e, create_graph=True
    )[0]
    z_div = z_div * e
    return z, z_div


def make_fwd_sde_step(z, xt, dt, alpha, std):
    drift = alpha * xt + std * z
    return xt + drift * dt + std * math.sqrt(dt) * torch.randn_like(xt)


def make_bwd_sde_step(z, xt, dt, alpha, std):
    drift = alpha * xt - std * z
    return xt - drift * dt + std * math.sqrt(dt) * torch.randn_like(xt)


def compute_ot_plan(x0, x1, g, t_max):
    a, b = ot.unif(x0.shape[0]), ot.unif(x1.shape[0])
    C = ot.dist(x0, x1).numpy()
    plan = ot.sinkhorn(
        a, b, C,
        numItermax=1000,
        stopThr=1e-6,
        reg=2 * t_max * g**2,
        verbose=False
    )
    return plan


def sample_ot_map(x0, x1, plan):
    p = plan.flatten() / plan.sum()
    choices = np.random.choice(
        x0.shape[0] * x1.shape[0],
        p=p,
        size=x0.shape[0],
        replace=True
    )

    i0, i1 = np.divmod(choices, x1.shape[0])
    return x0[i0], x1[i1]


def couple(x0, x1, g, t_max, device='cpu'):
    plan = compute_ot_plan(x0, x0, g, t_max)
    x0, x1 = sample_ot_map(x0, x1, plan)
    return x0.to(device), x1.to(device)
