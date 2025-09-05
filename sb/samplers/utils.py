import math
import torch
from sb.nn.utils import ModelOutput
from sb.losses.utils import (
    get_mean_log_var,
    get_model_outputs,
    make_fwd_sde_step, 
    make_bwd_sde_step, 
)

# @torch.no_grad()
# def sample_trajectory(model, x_start, direction, dt, n_steps, t_max, 
#                       only_last: bool = False, return_timesteps: bool = False,
#                       matching_method: str = "ll"):
#     assert direction in {"forward", "backward"}
#     trajectory = [x_start]
#     timesteps = [f"{t_max if direction == 'backward' else 0}"]
    
#     if matching_method != "ll":
#         raise ValueError(
#             f"Matching method {matching_method} is not supported in v1 sampler."
#         )

#     for t_step in (
#             torch.linspace(dt, t_max, n_steps).flip(-1) \
#             if direction == 'backward' \
#             else torch.linspace(0, t_max - dt, n_steps)
#         ):
#         shift = - dt if direction == "backward" else + dt
#         timesteps.append(f"{t_step.item() + shift:.3f}")
        
#         t = torch.ones(x_start.size(0), device=x_start.device) * t_step
#         mean, log_var = get_mean_log_var(model, trajectory[-1], t, dt)
#         noise_std = log_var.exp().sqrt()

#         x_new = mean + torch.randn_like(mean) * noise_std
        
#         trajectory.append(x_new)

    
#     if return_timesteps:
#         return trajectory, timesteps
    
#     if only_last:
#         return trajectory[-1]
    
#     return trajectory


@torch.no_grad()
def sample_trajectory(
        # fwd_model, bwd_model,
        model,
        x, dt, t_max, num_steps, alpha, var, 
        direction="fwd", 
        only_last: bool = False, 
        return_timesteps: bool = False, 
        method: str = "ll"
    ):
    assert direction in {"fwd", "bwd"}
    trajectory = [x]
    
    timesteps = torch.linspace(0, t_max, num_steps + 1)
    for t_step in (timesteps if direction == "fwd" else timesteps.flip(-1))[:-1]:
        t = torch.ones(x.size(0), device=x.device) * t_step
        x = trajectory[-1]
        
        if method == 'eot':
            # model = fwd_model if direction == "fwd" else bwd_model
            model_output, std = get_model_outputs(model, x, t, dt, base_var=var)
            x = model_output + std * torch.randn_like(model_output)
        
        elif method == "ll":
            # model = fwd_model if direction == "fwd" else bwd_model
            model_output, std = get_model_outputs(model, x, t, dt, base_var=var)
            x = x + model_output * dt + std * torch.randn_like(model_output)
        
        elif method == "mean":
            # model = fwd_model if direction == "fwd" else bwd_model
            model_output, std = get_model_outputs(model, x, t, dt, base_var=var)
            x = model_output + std * torch.randn_like(model_output)
        
        elif method == "score":
            # model = fwd_model if direction == "fwd" else bwd_model
            model_output, std = get_model_outputs(model, x, t, dt, base_var=var)
            x = x + model_output + std * torch.randn_like(model_output)
        
        elif method == "sf2m":
            var_ = t_max * var
            t_ = t / t_max
            dt_ = 1 / num_steps

            output = fwd_model(x, t_)
            if direction == "fwd":
                drift = output.drift + output.log_var
                x = x + drift * dt_ + math.sqrt(var_ * dt_) * torch.randn_like(drift)
            else:   
                drift = output.drift - output.log_var
                x = x - drift * dt_ + math.sqrt(var_ * dt_) * torch.randn_like(drift)
        
        elif method in {"dsbm", "dsbm++"}:
            if direction == "fwd":
                x = x + (- alpha * x + model(x, t).drift) * dt + \
                    math.sqrt(var * dt) * torch.randn_like(x)
            else:
                x = x - (alpha * x + model(x, t).drift) * dt + \
                    math.sqrt(var * dt) * torch.randn_like(x)
        
        elif method == "sde":
            g = math.sqrt(var)
            if direction == "fwd":
                z = model(x, t).drift
                x = make_fwd_sde_step(z, x, dt, alpha, g)
            else:
                z = model(x, t).drift
                x = make_bwd_sde_step(z, x, dt, alpha, g)
        
        trajectory.append(x)

    if return_timesteps:
        timesteps = [f"{t.item():.3f}" for t in timesteps]
        if direction == "bwd":
            timesteps = timesteps[::-1]
        return trajectory, timesteps 

    if only_last:
        return trajectory[-1]

    return trajectory


class ReferenceProcess:
    def __init__(self, alpha: float, dt: float, method: str):
        self.dt = dt
        self.alpha = alpha
        self.method = method
    
    def __call__(self, x, t):
        if self.method == 'll':
            return ModelOutput(drift= - self.alpha * x)
        if self.method in {'mean', 'eot'}:
            return ModelOutput(drift=(1 - self.alpha * self.dt) *  x)
        if self.method == 'sde':
            return ModelOutput(drift=0)
        if self.method == 'score':
            return ModelOutput(drift=-self.alpha * x * self.dt)
