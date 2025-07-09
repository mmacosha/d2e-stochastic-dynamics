import torch
from . import d2e


class DiffusionSampler(d2e.D2ESB):
    def __init__(self, fwd_model, bwd_model, p0, p1, config, buffer_config):
        super().__init__(fwd_model, bwd_model, p0, p1, config, buffer_config)
        self.config.backward_first = False 

    def train_backward_step(self, sb_iter, run):
        return None

    @torch.no_grad
    def log_backward_step(self, sb_iter, run):
        return None