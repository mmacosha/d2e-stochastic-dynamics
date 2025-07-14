import ot
import math
import numpy as np

import torch
import torch.nn.functional as F

from . import utils


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


def compute_fwd_tlm_loss_v2(fwd_model, bwd_model, x0, x1, 
                         dt, t_max, num_steps, alpha, var,
                         backward: bool = True, matching_method: str = "ll"):
    r"""Compute forward trajectory likelihood."""
    xt = x1
    traj_loss = 0

    for t_step in torch.linspace(t_max, dt, num_steps):
        t = torch.ones(xt.size(0), device=xt.device) * t_step

        if matching_method == "eot":
            with torch.no_grad():
                mean = x0 + (xt - x0) * (t - dt) / t
                std = math.sqrt(var * (t - dt) * dt / t)
                xt_m_dt = mean + std * torch.randn_like(mean)
                target = xt_m_dt + dt / (t_max - t + dt) * (x1  - xt_m_dt)
            
            fwd_mean, _ = utils.get_model_outputs(
                fwd_model, xt_m_dt, t - dt, dt, base_var=var
            )
            loss = F.mse_loss(fwd_mean, target)

        elif matching_method == "ll":
            with torch.no_grad():
                bwd_drift, bwd_std = utils.get_model_outputs(
                    bwd_model, xt, t, dt, base_var=var
                )
                xt_m_dt = xt + bwd_drift * dt + bwd_std * torch.randn_like(bwd_std)

            fwd_drift, fwd_std = utils.get_model_outputs(
                fwd_model, xt_m_dt, t - dt, dt, base_var=var
            )
            loss = - log_normal_density_v2(xt, xt_m_dt + fwd_drift * dt, fwd_std)

        elif matching_method == "mean":
            with torch.no_grad():
                bwd_mean, bwd_std = utils.get_model_outputs(
                    bwd_model, xt, t, dt, base_var=var
                )
                xt_m_dt = bwd_mean + bwd_std * torch.randn_like(bwd_std)
                bwd_mean_new, _ = utils.get_model_outputs(
                    bwd_model, xt_m_dt, t, dt, base_var=var
                )

            fwd_mean, _ = utils.get_model_outputs(
                fwd_model, xt_m_dt, t - dt, dt, var
            )
            loss = F.mse_loss(fwd_mean, xt_m_dt + bwd_mean - bwd_mean_new)

        elif matching_method == "score":
            with torch.no_grad():
                bwd_drift_dt, bwd_std = utils.get_model_outputs(
                    bwd_model, xt, t, dt, base_var=var
                )
                xt_m_dt = xt + bwd_drift_dt + bwd_std * torch.randn_like(bwd_std)
                bwd_drift_dt_new, _ = utils.get_model_outputs(
                    bwd_model, xt_m_dt, t, dt, base_var=var
                )
                target = xt + bwd_drift_dt - (xt_m_dt + bwd_drift_dt_new)
            
            fwd_drift_dt, _ = utils.get_model_outputs(
                fwd_model, xt_m_dt, t - dt, dt, var
            )
            loss = F.mse_loss(fwd_drift_dt, target)
        
        elif matching_method == "sde":
            with torch.no_grad():
                z, bwd_std = utils.get_model_outputs(
                    bwd_model, xt, t, dt, base_var=var
                )
                xt_m_dt = make_bwd_sde_step(z, xt, dt, alpha, bwd_std)

            z_hat, z_div = compute_z_div_z(fwd_model, xt_m_dt, t - dt, dt, var)
            loss = (z_hat * (0.5 * z_hat + z) + z_div).sum(1).mean()

        if backward:
            loss.mean().backward()
        
        traj_loss = traj_loss + loss.mean()
        xt = xt_m_dt
    
    return traj_loss


def compute_bwd_tlm_loss_v2(fwd_model, bwd_model,  x0, x1, 
                         dt, t_max, num_steps, alpha, var,
                         backward: bool = True, matching_method: str = "ll"):
    r"""Compute backward trajectory likelihood."""
    xt_m_dt = x0
    traj_loss = 0

    for t_step in torch.linspace(dt, t_max, num_steps):
        t = torch.ones(xt_m_dt.size(0), device=xt_m_dt.device) * t_step

        if matching_method == "eot":
            t = t - dt
            T = t_max - t
            mean = x1 if t == t_max - dt else xt_m_dt + (x1 - xt_m_dt) * dt / T
            std = 0 if t == t_max - dt else math.sqrt(var * dt * (T - dt) / T)
            xt = mean + std * torch.randn_like(mean)
            
            bwd_mean, _ = utils.get_model_outputs(
                bwd_model, xt, t + dt, dt, base_var=var
            )
            loss = F.mse_loss(bwd_mean, xt_m_dt)

        elif matching_method == "ll":
            with torch.no_grad():
                fwd_drift, fwd_std = utils.get_model_outputs(
                    fwd_model, xt_m_dt, t - dt, dt, base_var=var
                )
                xt = xt_m_dt + fwd_drift * dt + fwd_std * torch.randn_like(fwd_drift)

            bwd_drift, bwd_std = utils.get_model_outputs(
                bwd_model, xt, t, dt, base_var=var
            )
            loss = - log_normal_density_v2(xt_m_dt, xt + bwd_drift * dt, bwd_std)
        
        elif matching_method == "mean":
            with torch.no_grad():
                fwd_mean, fwd_std = utils.get_model_outputs(
                    fwd_model, xt_m_dt, t - dt, dt, base_var=var
                )
                xt = fwd_mean + fwd_std * torch.randn_like(fwd_std)
                fwd_mean_new, _ = utils.get_model_outputs(
                    fwd_model, xt, t - dt, dt, base_var=var
                )

            bwd_mean, _ = utils.get_model_outputs(
                bwd_model, xt, t, dt, base_var=var
            )
            loss = F.mse_loss(bwd_mean, xt + fwd_mean - fwd_mean_new)

        elif matching_method == "score":
            with torch.no_grad():
                fwd_drift_dt, fwd_std = utils.get_model_outputs(
                    fwd_model, xt_m_dt, t - dt, dt, base_var=var
                )
                xt = xt_m_dt + fwd_drift_dt + fwd_std * torch.randn_like(fwd_std)
                fwd_drift_dt_new, _ = utils.get_model_outputs(
                    fwd_model, xt, t - dt, dt, base_var=var
                )
                target = xt_m_dt + fwd_drift_dt - (xt + fwd_drift_dt_new)

            bwd_mean, _ = utils.get_model_outputs(
                bwd_model, xt, t, dt, base_var=var
            )
            loss = F.mse_loss(bwd_mean, target)

        elif matching_method == "sde":
            with torch.no_grad():
                z, fwd_std = utils.get_model_outputs(
                    fwd_model, xt_m_dt, t - dt, dt, base_var=var
                )
                xt = make_fwd_sde_step(z, xt_m_dt, dt, alpha, fwd_std)
            
            z_hat, z_div = compute_z_div_z(bwd_model, xt, t, dt, var)
            loss = (z_hat * (0.5 * z_hat + z) + z_div).sum(1).mean()

        if backward:
            loss.mean().backward()
        
        traj_loss = traj_loss + loss.mean()
        xt_m_dt = xt
    
    return traj_loss


def compute_fwd_tb_log_difference(fwd_model, bwd_model, log_p1, log_p0, x0, 
                                  dt, t_max, num_steps, p1_buffer = None,
                                  record_trajectory: bool = False):
    """Compute log[p_fwd p0 / p_bwd p1] for a forward trajectory. """
    xt_m_dt = x
    trajectory = []
    
    if record_trajectory:
        trajectory.append(xt_m_dt)
    
    log_p_fwd, log_p_bwd = 0, 0
    for t_step in torch.linspace(dt, t_max, num_steps):
        t = torch.ones(xt_m_dt.size(0), device=xt_m_dt.device) * t_step

        # COMPUTE FORWARD LOSS
        fwd_mean, fwd_log_var = utils.get_mean_log_var(fwd_model, xt_m_dt, t - dt, dt)

        with torch.no_grad():
            xt = fwd_mean + fwd_log_var.exp().sqrt() * torch.randn_like(fwd_mean)
        log_p_fwd = log_p_fwd + log_normal_density(xt, fwd_mean, fwd_log_var)

        # COMPUTE BACKWARD LOSS
        with torch.no_grad():
            bwd_mean, bwd_log_var = utils.get_mean_log_var(bwd_model, xt, t, dt)
            log_p_bwd = log_p_bwd + log_normal_density(xt_m_dt, bwd_mean, bwd_log_var)

        if record_trajectory:
            trajectory.append(xt)
        xt_m_dt = xt

    if p1_buffer is not None:
        p1_buffer.update(xt_m_dt)
    
    x1 = xt_m_dt
    log_diff = log_p_fwd + log_p0(x0) - (log_p_bwd + log_p1(x1))
    
    return log_diff, x1, trajectory


def compute_bwd_tb_log_difference(fwd_model, bwd_model, log_p, x, dt, t_max, 
                                  num_t_steps, p0_buffer = None,
                                  return_x: bool = False, learn_bwd: bool = True):
    fwd_tl_sum, bwd_tl_sum = 0, 0
    xt = x

    if not learn_bwd:
        bwd_tl_sum = bwd_tl_sum + log_p(xt)

    for t_step in torch.linspace(dt, t_max, num_t_steps).flip(-1):
        t = torch.ones(xt.size(0), device=xt.device) * t_step

        # COMPUTE BACKWARD LOSS
        with torch.set_grad_enabled(learn_bwd):
            bwd_mean, bwd_log_var = utils.get_mean_log_var(bwd_model, xt, t, dt)

            with torch.no_grad():
                xt_m_dt = bwd_mean + bwd_log_var.exp().sqrt() * torch.randn_like(bwd_mean)

            bwd_tl_sum = bwd_tl_sum + log_normal_density(xt_m_dt, bwd_mean, bwd_log_var)

        # COMPUTE FORWARD LOSS
        with torch.set_grad_enabled(not learn_bwd):
            fwd_mean, fwd_log_var = utils.get_mean_log_var(fwd_model, xt_m_dt, t - dt, dt)
            fwd_tl_sum = fwd_tl_sum + log_normal_density(xt, fwd_mean, fwd_log_var)

        xt = xt_m_dt
    
    if learn_bwd:
        fwd_tl_sum = fwd_tl_sum + log_p(xt)

    if p0_buffer is not None:
        p0_buffer.update(xt)
    
    if return_x:
        return fwd_tl_sum - bwd_tl_sum, xt
    
    return  fwd_tl_sum - bwd_tl_sum


def compute_fwd_tb_loss(fwd_model, bwd_model, log_p1, log_p0, x, dt, t_max, 
                        num_t_steps, p1_buffer = None):
    log = compute_fwd_tb_log_difference(fwd_model, bwd_model, log_p1, x, dt, 
                                        t_max, num_t_steps, p1_buffer=p1_buffer)
    log = log - log_p0(x)

    return (log  - log.mean(0, keepdim=True).detach()).pow(2).mean()


def compute_fwd_vargrad_loss(fwd_model, bwd_model, log_p1, x, dt, t_max, 
                             num_t_steps, p1_buffer = None, 
                             n_trajectories: int = 2, 
                             compute_var: bool = True):
    log = compute_fwd_tb_log_difference(fwd_model, bwd_model, log_p1, x, dt, 
                                        t_max, num_t_steps, p1_buffer=p1_buffer)
    
    if not compute_var:
        return log

    log = log.reshape(n_trajectories, -1)
    return (log  - log.mean(0, keepdim=True).detach()).pow(2).mean()


def compute_bwd_vargrad_loss(fwd_model, bwd_model, log_p0, x, dt, t_max,
                             num_t_steps, p0_buffer = None, n_trajectories: int = 2):
    log = compute_bwd_tb_log_difference(fwd_model, bwd_model, log_p0, x, dt, t_max,
                                        num_t_steps, p0_buffer=p0_buffer)
    log = log.reshape(n_trajectories, -1)
    return (log  - log.mean(0, keepdim=True).detach()).pow(2).mean()


def compute_fwd_tb_log_difference_reuse_traj(
        fwd_model, bwd_model, log_p1, x, dt, t_max, num_t_steps,
):
    xt = x
    fwd_tl_sum, bwd_tl_sum = 0, log_p1(xt)
    for t_step in torch.linspace(t_max, dt, num_t_steps):
        t = torch.ones(xt.size(0), device=xt.device) * t_step

        with torch.no_grad():
            bwd_mean, bwd_log_var = utils.get_mean_log_var(bwd_model, xt, t, dt)
            xt_m_dt = bwd_mean + bwd_log_var.exp().sqrt() * torch.randn_like(bwd_mean)
            bwd_tl_sum = bwd_tl_sum + log_normal_density(xt_m_dt, bwd_mean, bwd_log_var)

        fwd_mean, fwd_log_var = utils.get_mean_log_var(fwd_model, xt_m_dt, t - dt, dt)
        fwd_tl_sum = fwd_tl_sum + log_normal_density(xt, fwd_mean, fwd_log_var)

        xt = xt_m_dt

    log_diff = bwd_tl_sum - fwd_tl_sum
    return log_diff, xt
