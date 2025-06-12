import torch
from models import utils


def extract_into_tensor(tensor, shape):
    num_expand_dims = len(shape) - 1
    return tensor.view([-1] + [1 for _ in range(num_expand_dims)])


def get_mean_log_var(model, x, t, dt):
    log_var = torch.log(torch.ones_like(x) * 2.0 * dt)
    output = model(x, t)
    
    if output.contains('log_var'):
        log_var = log_var + output.log_var
        if log_var.isnan().any():
            print(log_var)
            assert 0, "log_var is NaN"
    
    if output.drift.isnan().any():
        print(output.drift)
        assert 0, "drift is NaN"
    

    mean = x + output.drift * dt
    return mean, log_var


def make_euler_maruyama_step(model, x, t, dt):
    mean, log_var = get_mean_log_var(model, x, t, dt)
    return mean + torch.randn_like(mean) * log_var.exp().sqrt()

@torch.no_grad()
def sample_trajectory(model, x_start, direction, dt, n_steps, t_max, 
                      only_last: bool = False, return_timesteps: bool = False):
    assert direction in {"forward", "backward"}
    trajectory = [x_start]
    timesteps = [f"{t_max if direction == 'backward' else 0}"]

    for t_step in (
            torch.linspace(dt, t_max, n_steps).flip(-1) \
            if direction == 'backward' \
            else torch.linspace(0, t_max - dt, n_steps)
        ):
        shift = - dt if direction == "backward" else + dt
        timesteps.append(f"{t_step.item() + shift:.3f}")
        
        t = torch.ones(x_start.size(0), device=x_start.device) * t_step
        trajectory.append(make_euler_maruyama_step(model, trajectory[-1], t, dt))
    
    if return_timesteps:
        return trajectory, timesteps
    
    if only_last:
        return trajectory[-1]
    
    return trajectory


class ReferenceProcess:
    def __init__(self, alpha: float, gamma: float):
        self.alpha = alpha
        self.gamma = gamma
    
    def __call__(self, x, t):
        return utils.ModelOutput(drift=-self.alpha * self.gamma * x)
    

class ReferenceProcess2:
    def __init__(self, alpha: float):
        self.alpha = alpha
    
    def __call__(self, x, t):
        return utils.ModelOutput(drift= - self.alpha * x)
