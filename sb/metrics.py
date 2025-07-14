import ot
import math
import torch

from sb import losses
from sb.samplers import utils


def log_mean_exp(x, dim: int = None):
    N = math.prod(x.size()) if dim is None else x.size(dim) 
    dim = [*range(len(x.size()))] if dim is None else dim
    return torch.logsumexp(x, dim) - math.log(N)


@torch.no_grad()
def compute_elbo(fwd_model, bwd_model, log_p_1, x, dt, t_max, num_t_steps, n_traj):
    x = x.repeat(n_traj, 1)
    log_diff = losses.compute_fwd_tb_log_difference(fwd_model, bwd_model, log_p_1, 
                                                    x, dt, t_max, num_t_steps)
    elbo = log_diff.mean()
    log_diff = log_diff.reshape(-1, n_traj) # [Batch, NTraj]
    
    iw_1 = log_mean_exp(log_diff, dim=1).mean()
    iw_2 = log_mean_exp(log_diff)

    return elbo, iw_1, iw_2


@torch.no_grad()
def compute_w2_distance(true, pred):
    cost_matrix = torch.cdist(true, pred, p=2) ** 2
    p1 = torch.ones(true.shape[0], device=true.device) / true.shape[0]
    p2 = torch.ones(pred.shape[0], device=pred.device) / pred.shape[0]
    w2_distance = ot.emd2(p1, p2, cost_matrix)
    return w2_distance


@torch.no_grad()
def compute_path_kl(fwd_model, x0, dt, t_max, n_steps, matching_method="ll"):
    """Compute KL[p(tau|x0) || q(tau | x0)]."""
    if matching_method != "ll":
        raise NotImplementedError("Not implemented for methods other that `ll`")

    xt = x0
    path_kl = 0
    for t_step in torch.linspace(0, t_max - dt, n_steps):
        t = torch.ones(xt.size(0), device=xt.device) * t_step
        fwd_mean, fwd_log_var = utils.get_mean_log_var(
                fwd_model, xt, t, dt
            )
        noise_var = fwd_log_var.exp()
        
        xt_p_dt = fwd_mean + torch.randn_like(fwd_mean) * noise_var.sqrt()
        
        step_kl = 0.5 * (
            torch.sum(
                torch.log(2 * dt / noise_var) + 
                (xt_p_dt - xt).pow(2) / (2 * dt) -
                (xt_p_dt - fwd_mean).pow(2) / noise_var, dim=1)
        )

        path_kl += step_kl
    
    return path_kl.mean()
