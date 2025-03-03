import math
import torch
from samplers import losses


def log_mean_exp(x, dim: int | None = None):
    N = math.prod(x.size()) if dim is None else x.size(dim) 
    dim = [*range(len(x.size()))] if dim is None else dim
    return torch.logsumexp(x, dim) - math.log(N)


@torch.no_grad()
def compute_elbo(fwd_model, bwd_model, log_p_1, x, dt, t_max, num_t_steps, n_traj):
    x = x.repeat(n_traj, 1)
    log_diff = -losses.compute_fwd_tb_log_difference(fwd_model, bwd_model, log_p_1, 
                                                     x, dt, t_max, num_t_steps)
    elbo = log_diff.mean()
    log_diff = log_diff.reshape(-1, n_traj)
    
    iw_1 = log_diff.exp().mean(1).log().mean()
    iw_1 = log_mean_exp(log_diff, dim=1).mean()
    iw_2 = log_mean_exp(log_diff)

    return elbo, iw_1, iw_2
