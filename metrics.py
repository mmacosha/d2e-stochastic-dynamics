import torch
from samplers import losses


@torch.no_grad()
def compute_elbo(fwd_model, bwd_model, log_p_1, x, dt, t_max, num_t_steps, n_traj):
    # x [B, 2]
    x = x.repeat(n_traj, 1) # [b * n_traj, 2]
    log_diff = -losses.compute_fwd_tb_log_difference(fwd_model, bwd_model, log_p_1, 
                                                x, dt, t_max, num_t_steps)
    elbo = log_diff.mean()
    log_diff = log_diff.reshape(-1, n_traj)
    
    iw_1 = log_diff.exp().mean(1).log().mean()
    iw_2 = log_diff.exp().mean().log()

    return elbo, iw_1, iw_2


def log_mean_exp(x, dim: int = -1):
    return torch.logsumexp() - torch.log(x.size(dim))
