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
    log_diff = losses.compute_fwd_tb_log_difference(
        fwd_model, bwd_model, log_p_1, 
        x, dt, t_max, num_t_steps
    )
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
def compute_path_kl(
        fwd_model, 
        x0, dt, t_max, n_steps, alpha, var, 
        method="ll"
    ):
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

            dt_ = 1 / n_steps
            t_ = t / t_max

            output = fwd_model(x, t_)
            fwd_mean = x + (output.drift + output.log_var) * dt_
            fwd_var = torch.as_tensor(var_ * dt_)

            ref_mean = x
            ref_var = torch.as_tensor(var_ * dt_)

        elif method in {"dsbm", "dsbm++"}:
            fwd_mean = x + (- alpha * x + output.drift) * dt
            fwd_var = ref_var

        elif method == "sde":
            g = math.sqrt(var)
            fwd_mean = x + ( - alpha * x + g * output.drift) * dt
            fwd_var = ref_var

        else:
            raise NotImplementedError(f"Unknown method: {method}")

        x = fwd_mean + torch.randn_like(fwd_mean) * fwd_var.sqrt()

        # KL(N1(fwd_mean, fwd_var), N2(ref_mean, ref_var))
        one_step_kl = 0.5 * torch.sum(
            torch.log(ref_var / fwd_var) + \
            (fwd_var + (fwd_mean - ref_mean).pow(2)) / ref_var - 1.0,
            dim=1
        )
        path_kl += one_step_kl.mean()

    return path_kl.mean()
