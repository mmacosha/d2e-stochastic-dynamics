import torch

from samplers import utils


def log_normal_density(x, mean, log_var):
    """Compute log dencity normal distribution."""
    return - 0.5 * (log_var + torch.exp(- log_var) * (mean - x).pow(2)).sum(-1)


def compute_fwd_tlm_loss(fwd_model, bwd_model, x_1, dt, 
                         t_max, n_steps, backward: bool = True):
    r"""Compute forward trajectory likelihood."""
    x_t = x_1
    traj_loss = 0

    for t_step in torch.linspace(dt, t_max, n_steps).flip(-1):
        t = torch.ones(512, device=x_t_m_dt.device) * t_step
        
        with torch.no_grad():
            bwd_mean, bwd_log_var = utils.get_mean_log_var(bwd_model, x_t, t, dt)
            x_t_m_dt = bwd_mean + torch.randn_like(bwd_mean) * bwd_log_var.exp().sqrt()
        
        fwd_mean, fwd_log_var = utils.get_mean_log_var(fwd_model, x_t_m_dt, t - dt, dt)
        loss = log_normal_density(x_t, fwd_mean, fwd_log_var)                        
        
        if backward:
            (-loss).mean().backward()
        
        traj_loss = traj_loss + (-loss).mean()
        x_t = x_t_m_dt
    
    return traj_loss


def compute_bwd_tlm_loss(fwd_model, bwd_model, x_0, dt, 
                         t_max, n_steps, backward: bool = True):
    r"""Compute backward trajectory likelihood."""
    x_t_m_dt = x_0
    traj_loss = 0

    for t_step in torch.linspace(dt, t_max, n_steps):
        t = torch.ones(512, device=x_t_m_dt.device) * t_step
        
        with torch.no_grad():
            bwd_mean, bwd_log_var = utils.get_mean_log_var(fwd_model, x_t_m_dt, 
                                                           t - dt, dt)
            x_t = bwd_mean + torch.randn_like(bwd_mean) * bwd_log_var.exp().sqrt()
        
        bwd_mean, bwd_log_var = utils.get_mean_log_var(bwd_model, x_t, t, dt)
        loss = log_normal_density(x_t_m_dt, bwd_mean, bwd_log_var)                        
        
        if backward:
            (-loss).mean().backward()
        
        traj_loss = traj_loss + (-loss).mean()
        x_t_m_dt = x_t
    
    return traj_loss


def compute_fwd_tb_log_difference(fwd_model, bwd_model, log_p_1, x, dt, t_max, 
                                  num_t_steps, p1_buffer = None, device: str = 'cpu'):
    fwd_tl_sum, bwd_tl_sum = 0, 0
    x_t_m_dt = x

    for t_step in torch.linspace(dt, t_max, num_t_steps):
        t = torch.ones(x_t_m_dt.size(0), device=device) * t_step

        # COMPUTE FORWARD LOSS
        fwd_mean, fwd_log_var = utils.get_mean_log_var(fwd_model, x_t_m_dt, t - dt, dt)
        with torch.no_grad():
            x_t = fwd_mean + fwd_log_var.exp().sqrt() * torch.randn_like(fwd_log_var)
        
        fwd_tl_sum = fwd_tl_sum + log_normal_density(x_t, fwd_mean, fwd_log_var)

        # COMPUTE BACKWARD LOSS
        with torch.no_grad():
            bwd_mean, bwd_log_var = utils.get_mean_log_var(bwd_model, x_t, t, dt)
            bwd_tl_sum = bwd_tl_sum + log_normal_density(x_t_m_dt, bwd_mean, bwd_log_var)

        x_t_m_dt = x_t
    bwd_tl_sum = bwd_tl_sum + log_p_1(x_t_m_dt)

    if p1_buffer is not None:
        p1_buffer.update(x_t_m_dt, fraction=0.2)

    return  fwd_tl_sum - bwd_tl_sum


def compute_bwd_tb_log_difference(fwd_model, bwd_model, log_p, x, dt, t_max, 
                                  num_t_steps, p0_buffer = None,
                                  return_x: bool = False, learn_bwd: bool = True):
    fwd_tl_sum, bwd_tl_sum = 0, 0
    x_t = x

    if not learn_bwd:
        bwd_tl_sum = bwd_tl_sum + log_p(x_t)

    for t_step in torch.linspace(dt, t_max, num_t_steps).flip(-1):
        t = torch.ones(x_t.size(0), device=x_t.device) * t_step

        # COMPUTE BACKWARD LOSS
        with torch.set_grad_enabled(learn_bwd):
            bwd_mean, bwd_log_var = utils.get_mean_log_var(bwd_model, x_t, t, dt)

            with torch.no_grad():
                x_t_m_dt = bwd_mean + bwd_log_var.exp().sqrt() * torch.randn_like(bwd_log_var)

            bwd_tl_sum = bwd_tl_sum + log_normal_density(x_t_m_dt, bwd_mean, bwd_log_var)

        # COMPUTE FORWARD LOSS
        with torch.set_grad_enabled(not learn_bwd):
            fwd_mean, fwd_log_var = utils.get_mean_log_var(fwd_model, x_t_m_dt, t - dt, dt)
            fwd_tl_sum = fwd_tl_sum + log_normal_density(x_t, fwd_mean, fwd_log_var)

        x_t = x_t_m_dt
    
    if learn_bwd:
        fwd_tl_sum = fwd_tl_sum + log_p(x_t)

    if p0_buffer is not None:
        p0_buffer.update(x_t, fraction=0.2)
    
    if return_x:
        return fwd_tl_sum - bwd_tl_sum, x_t
    
    return  fwd_tl_sum - bwd_tl_sum


def compute_fwd_ctb_loss(fwd_model, bwd_model, log_p_1, x, dt, 
                         t_max, num_t_steps, p1_buffer = None):
    log_1 = compute_fwd_tb_log_difference(fwd_model, bwd_model, log_p_1, x, dt, t_max, 
                                          num_t_steps, p1_buffer=p1_buffer)

    log_2 = compute_fwd_tb_log_difference(fwd_model, bwd_model, log_p_1, x, dt, t_max, 
                                          num_t_steps, p1_buffer=None)

    return (log_1 - log_2).pow(2).mean()


def compute_bwd_ctb_loss(fwd_model, bwd_model, log_p0, x, dt, 
                         t_max, num_t_steps, p0_buffer = None):
    log_1 = compute_bwd_tb_log_difference(fwd_model, bwd_model, log_p0, x, dt, t_max, 
                                          num_t_steps, p0_buffer=p0_buffer)

    log_2 = compute_bwd_tb_log_difference(fwd_model, bwd_model, log_p0, x, dt, t_max, 
                                          num_t_steps, p0_buffer=None)

    return (log_1 - log_2).pow(2).mean()


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def compute_fwd_ctb_loss_reuse_bwd(fwd_model, bwd_model, log_p_1, x_1, dt, 
                                   t_max, num_t_steps, p1_buffer = None):
    log_1, x_0 = compute_bwd_tb_log_difference(fwd_model, bwd_model, log_p_1, x_1, dt, 
                                               t_max, num_t_steps, p0_buffer=None,
                                               return_x=True, learn_bwd=False)
    log_2 = compute_fwd_tb_log_difference(fwd_model, bwd_model, log_p_1, x_0, dt, t_max, 
                                          num_t_steps, p1_buffer=p1_buffer)
    return (log_1 - log_2).pow(2).mean()