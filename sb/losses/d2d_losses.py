import math

import torch
import torch.nn.functional as F

from . import utils


def compute_fwd_tlm_loss(fwd_model, bwd_model, x1, dt, t_max, n_steps, 
                         backward: bool = True, method: str = "ll"):
    r"""Compute forward trajectory likelihood."""
    xt = x1
    traj_loss = 0
    if method != "ll":
        raise ValueError('v1 losses are correctly implemented only for "ll" method')

    for t_step in torch.linspace(dt, t_max, n_steps).flip(-1):
        t = torch.ones(xt.size(0), device=xt.device) * t_step
        
        with torch.no_grad():
            bwd_mean, bwd_log_var = utils.get_mean_log_var(bwd_model, xt, t, dt)
            xt_m_dt = bwd_mean + torch.randn_like(bwd_mean) * bwd_log_var.exp().sqrt()
            
        fwd_mean, fwd_log_var = utils.get_mean_log_var(
            fwd_model, xt_m_dt, t - dt, dt
        )
        loss = - utils.log_normal_density(xt, fwd_mean, fwd_log_var)
        
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

    if method != "ll":
        raise ValueError('v1 losses are correctly implemented only for "ll" method')

    for t_step in torch.linspace(dt, t_max, n_steps):
        t = torch.ones(xt_m_dt.size(0), device=xt_m_dt.device) * t_step

        with torch.no_grad():
            fwd_mean, fwd_log_var = utils.get_mean_log_var(
                fwd_model, xt_m_dt, t - dt, dt
            )
            xt = fwd_mean + torch.randn_like(fwd_mean) * fwd_log_var.exp().sqrt()
            
        bwd_mean, bwd_log_var = utils.get_mean_log_var(bwd_model, xt, t, dt)
        loss = - utils.log_normal_density(xt_m_dt, bwd_mean, bwd_log_var)
        
        assert ~loss.isnan().any(), f"Loss is NaN on {t_step=}"
       
        if backward:
            loss.mean().backward()
        
        traj_loss = traj_loss + loss.mean()
        xt_m_dt = xt
    
    return traj_loss


def sf2m_loss(fwd_model, x0, x1, var):
    fwd_model.train()

    t = torch.rand(x0.size(0), 1, device=x0.device)

    z = torch.randn_like(x0)
    xt = t * x1 + (1 - t) * x0 + torch.sqrt(var * t * (1 - t)) * z    
    target_drift = (x1 - x0) + \
                   (xt - t * x1 - (1 - t) * x0) * (1 - 2 * t) / (t * (1 - t)).clip(1e-6)
    lmbda = 2 * (t * (1 - t) / var).sqrt()

    output = fwd_model(xt, t.squeeze(1))
    drift_loss = (target_drift - output.drift).pow(2).sum(1)
    score_loss = (lmbda * output.log_var + z).pow(2).sum(1)

    # score = (t*x1 + (1 - t) * x0 - xt) / (sigma^2 * t * (1 - t))
    # score =  - sigma * sqrt((1 - t)*t)/ (sigma^2 * t * (1 - t))

    return (drift_loss + score_loss).mean()


def compute_fwd_tlm_loss_v2(fwd_model, bwd_model, x0, x1, 
                            dt, t_max, num_steps, alpha, var,
                            begin: bool = False,
                            backward: bool = True, 
                            method: str = "ll"):
    r"""Compute forward trajectory likelihood."""
    if method in {"eot", "sf2m"}:
        x0, x1 = utils.couple(x0, x1, var, t_max, device=x0.device)

    if method in {"dsbm", "dsbm++"}:
        with torch.no_grad():
            x0 = x1.clone()
            for t_step in torch.linspace(t_max, dt, num_steps):
                z = torch.randn_like(x0)
                t = torch.ones(x0.size(0), device=x0.device) * t_step
                if begin:
                    x0 = x0 + alpha * x0 * dt + math.sqrt(var * dt) * z
                else:
                    x0 = x0 - (alpha * x0 + bwd_model(x0, t).drift) * dt + \
                         math.sqrt(var * dt) * z
        
        if method == "dsbm++":
            x0, x1 = utils.couple(x0, x1, var, t_max, device=x0.device)

        noise = torch.randn_like(x0)
        t = torch.rand(x0.size(0), 1, device=x0.device) * t_max
        xt = (t / t_max) * x1 + (1 - t / t_max) * x0 + \
             noise * (var * t * (1 - t / t_max)).sqrt()
        
        target_score = (x1 - xt) / (t_max - t)
        loss = (target_score - fwd_model(xt, t.squeeze(1)).drift).pow(2).mean()
        
        if backward:
            loss.mean().backward()

        return loss

    if method == "sf2m":
        loss = sf2m_loss(fwd_model, x0, x1, t_max * var)
        
        if backward:
            loss.mean().backward()
        
        return loss

    xt = x1
    traj_loss = 0

    for t_step in torch.linspace(t_max, dt, num_steps):
        t = torch.ones(xt.size(0), device=xt.device) * t_step

        if method == "eot":
            with torch.no_grad():
                t_ = t[:, None]
                mean = x0 + (xt - x0) * (t_ - dt) / t_
                std = torch.sqrt(var * (t_ - dt) * dt / t_)
                xt_m_dt = mean + std * torch.randn_like(mean)
                target = xt_m_dt + dt / (t_max - t_ + dt) * (x1  - xt_m_dt)
            
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
                z = bwd_model(xt, t).drift
                g = math.sqrt(var)
                xt_m_dt = utils.make_bwd_sde_step(z, xt, dt, alpha, g)

            z_hat, g_div_z = utils.compute_z_div_z(fwd_model, xt_m_dt, t - dt, dt, var)
            loss = (0.5 * z_hat.pow(2) + z_hat * z + g_div_z).sum(1).mean()

        if backward:
            loss.mean().backward()
        
        traj_loss = traj_loss + loss.mean()
        xt = xt_m_dt
    
    return traj_loss


def compute_bwd_tlm_loss_v2(fwd_model, bwd_model,  x0, x1, 
                            dt, t_max, num_steps, alpha, var,
                            begin: bool = False,
                            backward: bool = True, 
                            method: str = "ll"):
    r"""Compute backward trajectory likelihood."""
    if method in {"eot", "sf2m"}:
        x0, x1 = utils.couple(x0, x1, var, t_max, device=x0.device)

    if method in {"dsbm", "dsbm++"}:
        x1 = x0.clone()
        with torch.no_grad():
            for t_step in torch.linspace(0, t_max - dt, num_steps):
                z = torch.randn_like(x1)
                t = torch.ones(z.size(0), device=z.device) * t_step
                if begin:
                    x1 = x1 + (- alpha * x1) * dt + math.sqrt(var * dt) * z
                else:
                    x1 = x1 + (- alpha * x1 + fwd_model(x1, t).drift) * dt + \
                         math.sqrt(var * dt) * z
                    
        if method == "dsbm++":
            x0, x1 = utils.couple(x0, x1, var, t_max, device=x0.device)

        noise = torch.randn_like(x0)
        t = torch.rand(x0.size(0), 1, device=x0.device) * t_max
        xt = x1 * (t / t_max) + (1 - t / t_max) * x0 + \
             noise * (var * t * (1 - t / t_max)).sqrt()
        
        target_score = (xt - x0) / t.clip(1e-6)
        loss = (target_score - bwd_model(xt, t.squeeze(1)).drift).pow(2).mean()

        if backward:
            loss.mean().backward()

        return loss
    
    if method == "sf2m":
        raise ValueError(
            "Backward model should not be used for SF2M training"
        )    
    
    xt_m_dt = x0
    traj_loss = 0

    for t_step in torch.linspace(dt, t_max, num_steps):
        t = torch.ones(xt_m_dt.size(0), device=xt_m_dt.device) * t_step

        if method == "eot":
            t_ = t[:, None] - dt
            T = t_max - t_
            mean = x1 if t_step == t_max else xt_m_dt + (x1 - xt_m_dt) * dt / T
            std = 0 if t_step == t_max else torch.sqrt(var * dt * (T - dt) / T)
            xt = mean + std * torch.randn_like(mean)
            
            bwd_mean, _ = utils.get_model_outputs(
                bwd_model, xt, t, dt, base_var=var
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
                z = fwd_model(xt_m_dt, t - dt).drift
                g = math.sqrt(var)
                xt = utils.make_fwd_sde_step(z, xt_m_dt, dt, alpha, g)
            
            z_hat, g_div_z = utils.compute_z_div_z(bwd_model, xt, t, dt, var)
            loss = (0.5 * z_hat.pow(2) + z_hat * z + g_div_z).sum(1).mean()

        if backward:
            loss.mean().backward()
        
        traj_loss = traj_loss + loss.mean()
        xt_m_dt = xt
    
    return traj_loss
