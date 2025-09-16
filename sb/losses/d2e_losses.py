import torch

from . import utils


def compute_fwd_tb_log_difference(fwd_model, bwd_model, log_p1, x, 
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
        log_p_fwd = log_p_fwd + utils.log_normal_density(xt, fwd_mean, fwd_log_var)

        # COMPUTE BACKWARD LOSS
        with torch.no_grad():
            bwd_mean, bwd_log_var = utils.get_mean_log_var(bwd_model, xt, t, dt)
            log_p_bwd = log_p_bwd + \
                        utils.log_normal_density(xt_m_dt, bwd_mean, bwd_log_var)

        if record_trajectory:
            trajectory.append(xt)
        xt_m_dt = xt

    if p1_buffer is not None:
        p1_buffer.update(xt_m_dt)
    
    x1 = xt_m_dt
    log_diff = log_p_fwd - (log_p_bwd + log_p1(x1))
    
    return log_diff


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

            bwd_tl_sum = bwd_tl_sum + \
                         utils.log_normal_density(xt_m_dt, bwd_mean, bwd_log_var)

        # COMPUTE FORWARD LOSS
        with torch.set_grad_enabled(not learn_bwd):
            fwd_mean, fwd_log_var = utils.get_mean_log_var(fwd_model, xt_m_dt, t - dt, dt)
            fwd_tl_sum = fwd_tl_sum + \
                         utils.log_normal_density(xt, fwd_mean, fwd_log_var)

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
    log = compute_fwd_tb_log_difference(fwd_model, bwd_model, log_p1, x, 
                                        dt, t_max, num_t_steps, 
                                        p1_buffer=p1_buffer)
    
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
            bwd_tl_sum = bwd_tl_sum + \
                         utils.log_normal_density(xt_m_dt, bwd_mean, bwd_log_var)

        fwd_mean, fwd_log_var = utils.get_mean_log_var(fwd_model, xt_m_dt, t - dt, dt)
        fwd_tl_sum = fwd_tl_sum + \
                     utils.log_normal_density(xt, fwd_mean, fwd_log_var)

        xt = xt_m_dt

    log_diff = bwd_tl_sum - fwd_tl_sum
    return log_diff, xt


def compute_relative_tb_loss(
    fwd_model, log_p1, x, dt, t_max, num_t_steps, alpha, var, 
):
    xt = x
    log_p_fwd, log_p_ref = 0, 0
    
    for t_step in torch.linspace(0, t_max - dt, num_t_steps):
        t = torch.ones(x.size(0), device=x.device) * t_step
        fwd_mean, fwd_log_var = utils.get_mean_log_var(fwd_model, xt, t, dt)
        ref_mean, ref_log_var = xt - alpha * t[:, None] * dt, torch.as_tensor(var * dt).log()
        ref_log_var = ref_log_var * torch.ones_like(xt)
        
        with torch.no_grad():
            xt = fwd_mean + fwd_log_var.exp().sqrt() * torch.randn_like(fwd_mean)
        
        log_p_fwd += log_p_fwd + utils.log_normal_density(xt, fwd_mean, fwd_log_var)
        log_p_ref += log_p_ref + utils.log_normal_density(xt, ref_mean, ref_log_var)

    if alpha > 0:
        var_div_2alpha = var / (2 * alpha)
        var_p1_ref = torch.as_tensor(
            var_div_2alpha + (- 2 * alpha * t_max).exp() * (1 - var_div_2alpha)
        )
    else:
        var_p1_ref = torch.as_tensor(1 + var * t_max)

    var_p1_ref = var_p1_ref * torch.ones_like(xt)
    log_p1_ref = - 0.5 * (var_p1_ref.log() - (xt).pow(2) / var_p1_ref).sum(-1)
    log_diff = (log_p_fwd - log_p_ref) + (log_p1_ref - log_p1(xt))

    return log_diff.pow(2).mean()


# -------------------------------- Optimised LD loss --------------------------------- #


def compute_fwd_log_difference(fwd_model, bwd_model, x0, dt, t_max, num_steps,
                               record_trajectory: bool = False):
    """Compute log[p_fwd / p_bwd] for a forward trajectory. """
    xt_m_dt = x0
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
        log_p_fwd = log_p_fwd + utils.log_normal_density(xt, fwd_mean, fwd_log_var)

        # COMPUTE BACKWARD LOSS
        with torch.no_grad():
            bwd_mean, bwd_log_var = utils.get_mean_log_var(bwd_model, xt, t, dt)
            log_p_bwd = log_p_bwd + \
                        utils.log_normal_density(xt_m_dt, bwd_mean, bwd_log_var)

        if record_trajectory:
            trajectory.append(xt)
        xt_m_dt = xt
    x1 = xt_m_dt

    trajectory = torch.stack(trajectory, dim=1) if record_trajectory else []
    return log_p_fwd, log_p_bwd, trajectory if record_trajectory else x1


def compute_log_p_fwd_loss_in_parallel(fwd_model, trajectory, dt, t_max, num_steps):
    batch_size, traj_length, dim = trajectory.shape
    xt, xt_p_dt = trajectory[:, :-1, :], trajectory[:, 1:, :]

    t = torch.linspace(0, t_max - dt, num_steps, device=trajectory.device)
    t = t.reshape(1, traj_length - 1, 1).repeat(batch_size, 1, 1)
    
    xt = xt.reshape(batch_size * (traj_length - 1), dim)
    xt_p_dt = xt_p_dt.reshape(batch_size * (traj_length - 1), dim)
    t = t.reshape(batch_size * (traj_length - 1), 1)
    
    drift, std = utils.get_model_outputs(fwd_model, xt, t, dt)
    log_p = utils.log_normal_density_v2(xt_p_dt, xt + drift * dt, std)
    return log_p.rehsape(batch_size, traj_length - 1).sum(dim=1)


def compute_log_p1(*args):
    pass


def compute_fwd_ld_optimized(
        fwd_model, bwd_model, log_p1_model, 
        x0, dt, t_max, num_steps,
        integrate_log_p1: bool = False,
    ):
    with torch.no_grad():
        log_p_fwd, log_p_bwd, trajectory = compute_fwd_log_difference(
            fwd_model, bwd_model, x0, dt, t_max, num_steps,
            record_trajectory=True
        )
        if integrate_log_p1:
            log_p1 = compute_log_p1(log_p1_model, trajectory)
        else:
            log_p1 = log_p1_model(trajectory[:, -1, :])
        weight = log_p_fwd - log_p_bwd - log_p1

    log_p_fwd_with_grad = compute_log_p_fwd_loss_in_parallel(
        fwd_model, trajectory, dt, t_max, num_steps
    )

    return torch.mean(weight * log_p_fwd_with_grad) 
