from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import torch
import wandb

from tqdm import trange

import utils
from samplers.utils import ReferenceProcess2


@dataclass
class SBConfig:
    name: str | None = None 
    backward_first: bool = False
    n_sb_iter: int = 10
    num_fwd_steps: int = 6000
    num_bwd_steps: int = 6000
    threshold: float = 1e-9

    fwd_optim_lr: int = 1e-3
    bwd_optim_lr: int = 1e-3

    alpha: float = 4.0
    dt: float = 0.0006
    t_max: float = 0.012
    n_steps: int = 20

    batch_size: int = 512
    save_checkpoint_freq: int = 2

    def __post__init__(self):
        assert self.dt * self.n_steps == self.t_max


class SB(ABC):
    def __init__(self, fwd_model, bwd_model, p0, p1, config):
        self.config = config
        self.fwd_model = fwd_model
        self.bwd_model = bwd_model

        self.fwd_optim = torch.optim.AdamW(self.fwd_model.parameters(), 
                                           lr=self.config.fwd_optim_lr)
        self.bwd_optim = torch.optim.AdamW(self.bwd_model.parameters(), 
                                           lr=self.config.bwd_optim_lr)

        self.p0 = p0
        self.p1 = p1

        self.fwd_ema_loss = utils.EMALoss(0.1)
        self.bwd_ema_loss = utils.EMALoss(0.1)

        self.reference_process = ReferenceProcess2(self.config.alpha)

    def train(self, experiment):
        with wandb.init(**experiment, config=self.config) as run:
            wandb.define_metric("forward_loss", step_metric="fwd_step")
            wandb.define_metric("backward_loss", step_metric="bwd_step")
            
            for sb_iter in trange(self.config.n_sb_iter, leave=False, 
                                  desc="SB training"):
                if self.config.backward_first:
                    self.train_backward_step(sb_iter, run)
                    self.log_backward_step(sb_iter, run)

                    self.train_forward_step(sb_iter, run)
                    self.log_forward_step(sb_iter, run)

                else:
                    self.train_forward_step(sb_iter, run)
                    self.log_forward_step(sb_iter, run)

                    self.train_backward_step(sb_iter, run)
                    self.log_backward_step(sb_iter, run)
                
                if sb_iter % self.config.save_checkpoint_freq == 0:
                    self.save_checkpoint(sb_iter, run)
        
            self.save_checkpoint("final", run)
    
    
    def save_checkpoint(self, sb_iter: str | int, run):
        checkpoint_path = Path(run.dir) / 'checkpoints'
        checkpoint_path.mkdir(exist_ok=True)

        checkpoint = {
            'forward': self.fwd_model.state_dict(), 
            'backward': self.bwd_model.state_dict(), 
            'forward_optim': self.fwd_optim.state_dict(), 
            'backward_optim': self.bwd_optim.state_dict(), 
        }

        torch.save(checkpoint, checkpoint_path / f'checkpoint-{sb_iter}.pth')
 
    @abstractmethod
    def train_forward_step(self, sb_iter, run):
        pass

    @abstractmethod
    def train_backward_step(self, sb_iter, run):
        pass

    @abstractmethod
    @torch.no_grad
    def log_forward_step(self, sb_iter, run):
        pass

    @abstractmethod
    @torch.no_grad
    def log_backward_step(self, sb_iter, run):
        pass
