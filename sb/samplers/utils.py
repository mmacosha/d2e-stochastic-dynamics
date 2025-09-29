import math
import torch
from sb.nn.utils import ModelOutput
from sb.losses.utils import (
    get_mean_log_var,
    get_model_outputs,
    make_fwd_sde_step, 
    make_bwd_sde_step, 
)

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

            output = model(x, t_)
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
