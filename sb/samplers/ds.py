
from . import base_class


class DiffusionSampler(base_class.SB):
    def __init__(self, config):
        super().__init__(config)
        self.fwd_model = None
        self.bwd_model = None
        self.fwd_optim = None
        self.bwd_optim = None


    def log_backward_step(self, sb_iter, run):
        pass

    def train_backward_step(self, sb_iter, run):
        pass