import torch
from model import ModelOutput


def extract_into_tensor(tensor, shape):
    num_expand_dims = len(shape) - 1
    return tensor.view([-1] + [1 for _ in range(num_expand_dims)])


def get_mean_log_var(model, x, t, dt):
    log_var = torch.as_tensor(2.0 * dt, device=x.device).log()
    output = model(x, t)
    
    if output.contains('log_var'):
        log_var = log_var + output.log_var
    
    mean = x + output.drift * dt
    return mean, log_var


def make_euler_maruyama_step(model, x, t, dt):
    mean, log_var = get_mean_log_var(model, x, t, dt)
    return mean + torch.randn_like(mean) * log_var.exp().sqrt()


class ReferenceProcess:
    def __init__(self, alpha: float, gamma: float):
        self.alpha = alpha
        self.gamma = gamma
    
    def __call__(self, x, t):
        return ModelOutput(drift=-self.alpha * self.gamma * x)
    

class ReferenceProcess2:
    def __init__(self, alpha: float):
        self.alpha = alpha
    
    def __call__(self, x, t):
        return ModelOutput(drift= - self.alpha * x)