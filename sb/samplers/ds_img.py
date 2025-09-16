import torch
from . import d2e_img


class DiffusionSamplerIMG(d2e_img.D2ESB_IMG):
    def __init__(self, fwd_model, bwd_model, p0, p1, config, buffer_config):
        super().__init__(fwd_model, bwd_model, p0, p1, config, buffer_config)
        self.config.backward_first = False 

    def train_backward_step(self, sb_iter, run):
        pass

    @torch.no_grad
    def log_backward_step(self, sb_iter, run):
        pass
