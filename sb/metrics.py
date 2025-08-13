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
def compute_eubo(fwd_model, bwd_model, log_p1, x1, dt, t_max, num_steps):
    log_diff, _ = losses.compute_fwd_tb_log_difference_reuse_traj(
        fwd_model, bwd_model, log_p1, x1, dt, t_max, num_steps
    )
    eubo = log_diff.mean()
    
    return eubo


@torch.no_grad()
def compute_w2_distance(true, pred, *args, **kwargs):
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
                (xt_p_dt - fwd_mean).pow(2) / noise_var, dim=1
            )
        )

        path_kl += step_kl
    
    return path_kl.mean()


@torch.no_grad()
def compute_path_energy_discrete(fwd_model, x0, dt, t_max, n_steps, alpha, var, 
                                 method="ll"):
    x = x0.clone()
    path_kl = 0.0
    for t_step in torch.linspace(0, t_max - dt, n_steps):
        t = torch.ones(x.size(0), device=x.device) * t_step

        output = fwd_model(x, t)
        ref_mean, ref_var = x - alpha * x * dt, torch.as_tensor(var * dt)

        if method == "mean":
            fwd_mean, fwd_var = output.drift, ref_var
        elif method == "score":
            fwd_mean, fwd_var = x + output.drift, ref_var
        elif method == "ll":
            fwd_mean = x + output.drift * dt
            log_var = torch.log(torch.ones_like(x) * var * dt)
            
            if output.contains('log_var'):
                log_var = log_var + output.log_var
            
            fwd_var = log_var.exp()
        
        elif method == "eot":
            fwd_mean, fwd_var = output.drift, ref_var
        elif method == "sf2m":
            var_ = t_max * var
            t_ = t * (1 - 1 / n_steps) / (t_max - dt)
            dt_ = 1 / n_steps
            
            output = fwd_model(x, t_)
            fwd_mean = x + (output.drift + var_ / 2 * output.log_var) * dt_
            fwd_var = var_ * dt

            ref_mean = x
            ref_var = var_ * dt
        elif method in {"dsbm", "dsbm++"}:
            fwd_mean = x + (- alpha * x + output.drift) * dt
            fwd_var = ref_var
        elif method == "sde":
            g = math.sqrt(var)
            fwd_mean = x + (alpha * x + g * output.drift) * dt
            fwd_var = ref_var
        else:
            raise NotImplementedError(f"Unknown method: {method}")
        
        x = fwd_mean + torch.randn_like(fwd_mean) * fwd_var.sqrt()

        path_kl = torch.sum((x - ref_mean).pow(2) / (2 * ref_var), dim=1) - \
                  torch.sum((x - fwd_mean).pow(2) / (2 * fwd_var), dim=1)
    
    return path_kl.mean()
