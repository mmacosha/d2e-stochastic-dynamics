import math

import torch
import torch.nn.functional as F

from . import utils


def compute_fwd_tlm_loss(fwd_model, bwd_model, x1, dt, t_max, n_steps, 
                         backward: bool = True, method: str = "ll"):
    r"""Compute forward trajectory likelihood."""
    xt = x1
    traj_loss = 0

    for t_step in torch.linspace(dt, t_max, n_steps).flip(-1):
        t = torch.ones(xt.size(0), device=xt.device) * t_step
        
        with torch.no_grad():
            if method == "ll":
                bwd_mean, bwd_log_var = utils.get_mean_log_var(bwd_model, xt, t, dt)
                noise_std = bwd_log_var.exp().sqrt()
                xt_m_dt = bwd_mean + torch.randn_like(bwd_mean) * noise_std
            
            elif method == "mean":
                bwd_mean, bwd_log_var = utils.get_mean_log_var(bwd_model, xt, t, dt)
                noise_std = bwd_log_var.exp().sqrt()
                xt_m_dt = bwd_mean + torch.randn_like(bwd_mean) * noise_std
                
                bwd_mean_new, _ = utils.get_mean_log_var(bwd_model, xt_m_dt, t, dt)
                target = xt_m_dt + bwd_mean - bwd_mean_new
            
            elif method == "score":
                bwd_mean, bwd_log_var = utils.get_mean_log_var(bwd_model, xt, t, dt)
                noise_std = bwd_log_var.exp().sqrt()
                xt_m_dt = xt + bwd_mean + torch.randn_like(xt) * noise_std

                bwd_mean_new, _ = utils.get_mean_log_var(bwd_model, xt_m_dt, t, dt)
                target = xt_m_dt + bwd_mean - (xt_m_dt + bwd_mean_new)
            
            elif method == "sde":
                z, bwd_log_var = utils.get_mean_log_var(bwd_model, xt, t, dt)
                g = bwd_log_var.exp().sqrt()
                xt_m_dt = utils.make_bwd_sde_step(z, xt, dt, 1.42, g)

        if method == "sde":
            z_hat, z_div = utils.compute_z_div_z(fwd_model, xt_m_dt, t - dt, g)
            loss = (z_hat * (0.5 * z_hat + z) + z_div).sum(1).mean()
        elif method == "ll":
            fwd_mean, fwd_log_var = utils.get_mean_log_var(
                fwd_model, xt_m_dt, t - dt, dt
            )
            loss = - utils.log_normal_density(xt, fwd_mean, fwd_log_var)
        else:
            fwd_mean, fwd_log_var = utils.get_mean_log_var(
                fwd_model, xt_m_dt, t - dt, dt
            )
            loss = torch.nn.functional.mse_loss(xt, target)
        
        if backward:
            loss.mean().backward()
        
        traj_loss = traj_loss + loss.mean()
        xt = xt_m_dt
    
    return traj_loss


def compute_bwd_tlm_loss(fwd_model, bwd_model, x_0, dt, t_max, n_steps, 
                         backward: bool = True, method: str = "ll"):
    r"""Compute backward trajectory likelihood."""
    xt_m_dt = x_0
    traj_loss = 0

    for t_step in torch.linspace(dt, t_max, n_steps):
        t = torch.ones(xt_m_dt.size(0), device=xt_m_dt.device) * t_step

        with torch.no_grad():
            if method == "ll":
                fwd_mean, fwd_log_var = utils.get_mean_log_var(
                    fwd_model, xt_m_dt, t - dt, dt
                )
                noise_std = fwd_log_var.exp().sqrt()
                xt = fwd_mean + torch.randn_like(fwd_mean) * noise_std
            
            elif method == "mean":
                fwd_mean, fwd_log_var = utils.get_mean_log_var(
                    fwd_model, xt_m_dt, t - dt, dt
                )
                noise_std = fwd_log_var.exp().sqrt()
                xt = fwd_mean + torch.randn_like(fwd_mean) * noise_std
                fwd_mean_new, _ = utils.get_mean_log_var(
                    fwd_model, xt, t - dt, dt
                )
                target = xt + fwd_mean - fwd_mean_new

            elif method == "score":
                fwd_mean, fwd_log_var = utils.get_mean_log_var(
                    fwd_model, xt_m_dt, t - dt, dt
                )
                noise_std = fwd_log_var.exp().sqrt()
                xt = xt_m_dt + fwd_mean + torch.randn_like(fwd_mean) * noise_std
                
                fwd_mean_new, _ = utils.get_mean_log_var(
                    fwd_model, xt, t - dt, dt
                )
                target = xt_m_dt + fwd_mean - (xt + fwd_mean_new)

            elif method == "sde":
                z, fwd_log_var = utils.get_mean_log_var(fwd_model, xt_m_dt, t - dt, dt)
                g = fwd_log_var.exp().sqrt()
                xt = utils.make_fwd_sde_step(z, xt_m_dt, dt, 1.42, g)

        if method == "sde":
            z_hat, z_div = utils.compute_z_div_z(bwd_model, xt, t, g)
            loss = (z_hat * (0.5 * z_hat + z) + z_div).sum(1).mean()
        elif method == "ll":
            bwd_mean, bwd_log_var = utils.get_mean_log_var(bwd_model, xt, t, dt)
            loss = - utils.log_normal_density(xt_m_dt, bwd_mean, bwd_log_var)
        else:
            bwd_mean, bwd_log_var = utils.get_mean_log_var(bwd_model, xt, t, dt)
            loss = torch.nn.functional.mse_loss(bwd_mean, target)
        
        assert ~loss.isnan().any(), f"Loss is NaN on {t_step=}"
       
        if backward:
            loss.mean().backward()
        
        traj_loss = traj_loss + loss.mean()
        xt_m_dt = xt
    
    return traj_loss


def compute_fwd_tlm_loss_v2(fwd_model, bwd_model, x0, x1, 
                         dt, t_max, num_steps, alpha, var,
                         backward: bool = True, method: str = "ll"):
    r"""Compute forward trajectory likelihood."""
    xt = x1
    traj_loss = 0

    for t_step in torch.linspace(t_max, dt, num_steps):
        t = torch.ones(xt.size(0), device=xt.device) * t_step

        if method == "eot":
            with torch.no_grad():
                mean = x0 + (xt - x0) * (t - dt) / t
                std = math.sqrt(var * (t - dt) * dt / t)
                xt_m_dt = mean + std * torch.randn_like(mean)
                target = xt_m_dt + dt / (t_max - t + dt) * (x1  - xt_m_dt)
            
            fwd_mean, _ = utils.get_model_outputs(
                fwd_model, xt_m_dt, t - dt, dt, base_var=var
            )
            loss = F.mse_loss(fwd_mean, target)

        elif method == "ll":
            with torch.no_grad():
                bwd_drift, bwd_std = utils.get_model_outputs(
                    bwd_model, xt, t, dt, base_var=var
                )
                xt_m_dt = xt + bwd_drift * dt + bwd_std * torch.randn_like(bwd_std)

            fwd_drift, fwd_std = utils.get_model_outputs(
                fwd_model, xt_m_dt, t - dt, dt, base_var=var
            )
            loss = - utils.log_normal_density_v2(xt, xt_m_dt + fwd_drift * dt, fwd_std)

        elif method == "mean":
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

        elif method == "score":
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
        
        elif method == "sde":
            with torch.no_grad():
                z, bwd_std = utils.get_model_outputs(
                    bwd_model, xt, t, dt, base_var=var
                )
                xt_m_dt = utils.make_bwd_sde_step(z, xt, dt, alpha, bwd_std)

            z_hat, z_div = utils.compute_z_div_z(fwd_model, xt_m_dt, t - dt, dt, var)
            loss = (z_hat * (0.5 * z_hat + z) + z_div).sum(1).mean()

        if backward:
            loss.mean().backward()
        
        traj_loss = traj_loss + loss.mean()
        xt = xt_m_dt
    
    return traj_loss


def compute_bwd_tlm_loss_v2(fwd_model, bwd_model,  x0, x1, 
                            dt, t_max, num_steps, alpha, var,
                            backward: bool = True, method: str = "ll"):
    r"""Compute backward trajectory likelihood."""
    xt_m_dt = x0
    traj_loss = 0

    for t_step in torch.linspace(dt, t_max, num_steps):
        t = torch.ones(xt_m_dt.size(0), device=xt_m_dt.device) * t_step

        if method == "eot":
            t = t - dt
            T = t_max - t
            mean = x1 if t == t_max - dt else xt_m_dt + (x1 - xt_m_dt) * dt / T
            std = 0 if t == t_max - dt else math.sqrt(var * dt * (T - dt) / T)
            xt = mean + std * torch.randn_like(mean)
            
            bwd_mean, _ = utils.get_model_outputs(
                bwd_model, xt, t + dt, dt, base_var=var
            )
            loss = F.mse_loss(bwd_mean, xt_m_dt)

        elif method == "ll":
            with torch.no_grad():
                fwd_drift, fwd_std = utils.get_model_outputs(
                    fwd_model, xt_m_dt, t - dt, dt, base_var=var
                )
                xt = xt_m_dt + fwd_drift * dt + fwd_std * torch.randn_like(fwd_drift)

            bwd_drift, bwd_std = utils.get_model_outputs(
                bwd_model, xt, t, dt, base_var=var
            )
            loss = - utils.log_normal_density_v2(xt_m_dt, xt + bwd_drift * dt, bwd_std)
        
        elif method == "mean":
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

        elif method == "score":
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

        elif method == "sde":
            with torch.no_grad():
                z, fwd_std = utils.get_model_outputs(
                    fwd_model, xt_m_dt, t - dt, dt, base_var=var
                )
                xt = utils.make_fwd_sde_step(z, xt_m_dt, dt, alpha, fwd_std)
            
            z_hat, z_div = utils.compute_z_div_z(bwd_model, xt, t, dt, var)
            loss = (z_hat * (0.5 * z_hat + z) + z_div).sum(1).mean()

        if backward:
            loss.mean().backward()
        
        traj_loss = traj_loss + loss.mean()
        xt_m_dt = xt
    
    return traj_loss


# def compute_fwd_dsbm_loss(fwd_model, bwd_model, x0, x1, 
#                           dt, t_max, num_steps, alpha, var,
#                           backward: bool = True, method: str = "ll",
#                           couple: bool = False):
    
#     if method == "dsbm++":
#         x0, x1 = couple(x0, x1) 

#     t = torch.randn(x0.size(0), 1, device=x0.device) * t_max
#     xt = t / t_max * x1 + (1 - t / t_max) * x0
#     std = math.sqrt(var * t * (t_max - t) / t_max)

#     loss = F.mse_loss(
