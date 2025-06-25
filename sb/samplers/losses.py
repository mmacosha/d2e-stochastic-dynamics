import torch

from . import utils


def ebm_loss(energy_model, positive_samples, negative_samples, alpha=0.0, 
             reg_type: str = 'l2'):
    """Compute EBM loss."""
    positive_energy = energy_model(positive_samples).mean()
    negative_energy = energy_model(negative_samples).mean()
    
    if reg_type == 'l2':
        regularization = alpha * (positive_energy**2 + negative_energy**2)
    elif reg_type == 'l1':
        regularization = alpha * (positive_energy.abs() + negative_energy.abs())
    else:
        raise ValueError(f"Unknown regularization type: {reg_type}")
    return positive_energy - negative_energy + regularization


def log_normal_density(x, mean, log_var):
    """Compute log dencity normal distribution."""
    batch_size = x.size(0)
    x = x.view(batch_size, -1)
    mean = mean.view(batch_size, -1)
    log_var = log_var.view(batch_size, -1)
    return - 0.5 * (log_var + torch.exp(- log_var) * (mean - x).pow(2)).sum(-1)


def compute_fwd_tlm_loss(fwd_model, bwd_model, x1, dt, t_max, n_steps, 
                         backward: bool = True, matching_method: str = "ll"):
    r"""Compute forward trajectory likelihood."""
    xt = x1
    traj_loss = 0

    for t_step in torch.linspace(dt, t_max, n_steps).flip(-1):
        t = torch.ones(xt.size(0), device=xt.device) * t_step
        
        with torch.no_grad():
            if matching_method == "ll":
                bwd_mean, bwd_log_var = utils.get_mean_log_var(bwd_model, xt, t, dt)
                noise_std = bwd_log_var.exp().sqrt()
                xt_m_dt = bwd_mean + torch.randn_like(bwd_mean) *noise_std
            
            elif matching_method == "mean":
                bwd_mean, bwd_log_var = utils.get_mean_log_var(bwd_model, xt, t, dt)
                noise_std = bwd_log_var.exp().sqrt()
                xt_m_dt = bwd_mean + torch.randn_like(bwd_mean) * noise_std
                
                bwd_mean_new, _ = utils.get_mean_log_var(bwd_model, xt_m_dt, t, dt)
                target = xt_m_dt + bwd_mean - bwd_mean_new
            
            elif matching_method == "score":
                bwd_mean, bwd_log_var = utils.get_mean_log_var(bwd_model, xt, t, dt)
                noise_std = bwd_log_var.exp().sqrt()
                xt_m_dt = xt + bwd_mean + torch.randn_like(xt) * noise_std

                bwd_mean_new, _ = utils.get_mean_log_var(bwd_model, xt_m_dt, t, dt)
                target = xt_m_dt + bwd_mean - (xt_m_dt + bwd_mean_new)

        fwd_mean, fwd_log_var = utils.get_mean_log_var(fwd_model, xt_m_dt, t - dt, dt)
        
        if matching_method == "ll":
            loss = - log_normal_density(xt, fwd_mean, fwd_log_var)
        else:
            loss = torch.nn.functional.mse_loss(xt, target)
        
        if backward:
            loss.mean().backward()
        
        traj_loss = traj_loss + loss.mean()
        xt = xt_m_dt
    
    return traj_loss


def compute_bwd_tlm_loss(fwd_model, bwd_model, x_0, dt, t_max, n_steps, 
                         backward: bool = True, matching_method: str = "ll"):
    r"""Compute backward trajectory likelihood."""
    xt_m_dt = x_0
    traj_loss = 0

    for t_step in torch.linspace(dt, t_max, n_steps):
        t = torch.ones(xt_m_dt.size(0), device=xt_m_dt.device) * t_step

        with torch.no_grad():
            if matching_method == "ll":
                fwd_mean, fwd_log_var = utils.get_mean_log_var(
                    fwd_model, xt_m_dt, t - dt, dt
                )
                noise_std = fwd_log_var.exp().sqrt()
                xt = fwd_mean + torch.randn_like(fwd_mean) * noise_std
            
            elif matching_method == "mean":
                fwd_mean, fwd_log_var = utils.get_mean_log_var(
                    fwd_model, xt_m_dt, t - dt, dt
                )
                noise_std = fwd_log_var.exp().sqrt()
                xt = fwd_mean + torch.randn_like(fwd_mean) * noise_std
                fwd_mean_new, _ = utils.get_mean_log_var(
                    fwd_model, xt, t - dt, dt
                )
                target = xt + fwd_mean - fwd_mean_new

            elif matching_method == "score":
                fwd_mean, fwd_log_var = utils.get_mean_log_var(
                    fwd_model, xt_m_dt, t - dt, dt
                )
                noise_std = fwd_log_var.exp().sqrt()
                xt = xt_m_dt + fwd_mean + torch.randn_like(fwd_mean) * noise_std
                fwd_mean_new, _ = utils.get_mean_log_var(
                    fwd_model, xt, t - dt, dt
                )
                target = xt_m_dt + fwd_mean - (xt + fwd_mean_new)

        bwd_mean, bwd_log_var = utils.get_mean_log_var(bwd_model, xt, t, dt)
        
        if matching_method == "ll":
            loss = - log_normal_density(xt_m_dt, bwd_mean, bwd_log_var)
        else:
            loss = torch.nn.functional.mse_loss(bwd_mean, target)
        
        assert ~loss.isnan().any(), f"Loss is NaN on {t_step=}"
       
        if backward:
            loss.mean().backward()
        
        traj_loss = traj_loss + loss.mean()
        xt_m_dt = xt
    
    return traj_loss


def compute_fwd_tb_log_difference(fwd_model, bwd_model, log_p1, x, dt, t_max, 
                                  num_t_steps, p1_buffer = None):
    fwd_tl_sum, bwd_tl_sum = 0, 0
    xt_m_dt = x

    for t_step in torch.linspace(dt, t_max, num_t_steps):
        t = torch.ones(xt_m_dt.size(0), device=xt_m_dt.device) * t_step

        # COMPUTE FORWARD LOSS
        fwd_mean, fwd_log_var = utils.get_mean_log_var(fwd_model, xt_m_dt, t - dt, dt)

        with torch.no_grad():
            xt = fwd_mean + fwd_log_var.exp().sqrt() * torch.randn_like(fwd_mean)
        fwd_tl_sum = fwd_tl_sum + log_normal_density(xt, fwd_mean, fwd_log_var)

        # COMPUTE BACKWARD LOSS
        with torch.no_grad():
            bwd_mean, bwd_log_var = utils.get_mean_log_var(bwd_model, xt, t, dt)
            bwd_tl_sum = bwd_tl_sum + log_normal_density(xt_m_dt, bwd_mean, bwd_log_var)

        xt_m_dt = xt

    if p1_buffer is not None:
        p1_buffer.update(xt_m_dt)

    return bwd_tl_sum + log_p1(xt_m_dt) - fwd_tl_sum


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
                             num_t_steps,p1_buffer = None, n_trajectories: int = 2):
    log = compute_fwd_tb_log_difference(fwd_model, bwd_model, log_p1, x, dt, 
                                        t_max, num_t_steps, p1_buffer=p1_buffer)
    if n_trajectories == 1:
        return log.pow(2).mean()

    log = log.reshape(n_trajectories, -1)
    return (log  - log.mean(0, keepdim=True).detach()).pow(2).mean()


def compute_bwd_vargrad_loss(fwd_model, bwd_model, log_p0, x, dt, t_max,
                             num_t_steps, p0_buffer = None, n_trajectories: int = 2):
    log = compute_bwd_tb_log_difference(fwd_model, bwd_model, log_p0, x, dt, t_max,
                                        num_t_steps, p0_buffer=p0_buffer)
    log = log.reshape(n_trajectories, -1)
    return (log  - log.mean(0, keepdim=True).detach()).pow(2).mean()
