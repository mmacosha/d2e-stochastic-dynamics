import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import torch
import wandb

from tqdm import trange

import sb.utils as utils
from sb.samplers.utils import ReferenceProcess2


@dataclass
class SBConfig:
    name: str  = None 
    device: str = "cpu"
    backward_first: bool = False
    n_sb_iter: int = 10
    num_fwd_steps: int = 6000
    num_bwd_steps: int = 6000
    threshold: float = 1e-9

    fwd_ema_decay: float = 0.999
    bwd_ema_decay: float = 0.999
    fwd_optim_lr: int = 1e-3
    bwd_optim_lr: int = 1e-3
    log_fwd_freq: int = 1
    log_bwd_freq: int = 1

    alpha: float = 4.0
    dt: float = 0.0006
    t_max: float = 0.012
    n_steps: int = 20

    batch_size: int = 512
    save_checkpoint_freq: int = 2
    restore_from: int = -1

    def __post__init__(self):
        assert self.dt * self.n_steps == self.t_max


def find_checkpoint(directory: str, checkpoint_num: int) -> str :
    """
    Get the checkpoint file from the given directory.
    If checkpoint_num is -1, return the last checkpoint.
    Otherwise, return the checkpoint corresponding to the checkpoint_num.
    
    Args:
        directory (str): The directory containing checkpoint files.
        checkpoint_num (int): The checkpoint number to retrieve (-1 for the latest checkpoint).
    
    Returns:
        str: The path to the selected checkpoint file.
    """
    directory_path = Path(directory)
    if not directory_path.is_dir():
        return None
    
    checkpoint_pattern = re.compile(r'checkpoint-(\d+).pth')
    checkpoints = []
    
    for file in directory_path.iterdir():
        match = checkpoint_pattern.match(file.name)
        if match:
            checkpoints.append((int(match.group(1)), file))
    
    if not checkpoints:
        return None
    
    checkpoints.sort(reverse=True, key=lambda x: x[0])
    
    if checkpoint_num == -1:
        return str(checkpoints[0][1])
    
    for num, file in checkpoints:
        if num == checkpoint_num:
            return str(file)
    
    raise FileNotFoundError(f"Checkpoint {checkpoint_num} not found in the directory.")



class SB(ABC):
    def __init__(self, fwd_model, bwd_model, p0, p1, config):
        self.config = config
        self.fwd_model = fwd_model
        self.bwd_model = bwd_model

        self.fwd_model_ema = utils.EMA(self.fwd_model, self.config.fwd_ema_decay)
        self.bwd_model_ema = utils.EMA(self.bwd_model, self.config.bwd_ema_decay)

        self.fwd_optim = torch.optim.AdamW(self.fwd_model.parameters(), 
                                           lr=self.config.fwd_optim_lr)
        self.bwd_optim = torch.optim.AdamW(self.bwd_model.parameters(), 
                                           lr=self.config.bwd_optim_lr)

        self.p0 = p0
        self.p1 = p1

        self.reference_process = ReferenceProcess2(self.config.alpha)

    def train(self, experiment):
        experiment.name = f"{self.config.name}-{experiment.name}"
        with wandb.init(**experiment, config=self.config) as run:
            wandb.define_metric("forward_loss", step_metric="fwd_step")
            wandb.define_metric("backward_loss", step_metric="bwd_step")

            self.resotre_from_last_checkpoint(run)
            for sb_iter in trange(self.config.n_sb_iter, leave=False, 
                                  desc="SB training"):
                if self.config.backward_first:
                    self.train_backward_step(sb_iter, run)
                    if not sb_iter % self.config.log_bwd_freq:
                        self.log_backward_step(sb_iter, run)

                    self.train_forward_step(sb_iter, run)
                    if not sb_iter % self.config.log_fwd_freq:
                        self.log_forward_step(sb_iter, run)

                else:
                    self.train_forward_step(sb_iter, run)
                    if not sb_iter % self.config.log_fwd_freq:
                        self.log_forward_step(sb_iter, run)

                    self.train_backward_step(sb_iter, run)
                    if not sb_iter % self.config.log_bwd_freq:
                        self.log_backward_step(sb_iter, run)
                
                if sb_iter % self.config.save_checkpoint_freq == 0:
                    self.save_checkpoint(sb_iter, run)
        
            self.save_checkpoint("final", run)
    
    
    def save_checkpoint(self, sb_iter: str , run):
        checkpoint_path = Path(run.dir) / 'checkpoints'
        checkpoint_path.mkdir(exist_ok=True)

        checkpoint = {
            'forward': self.fwd_model.state_dict(), 
            'backward': self.bwd_model.state_dict(), 
            'forward_optim': self.fwd_optim.state_dict(), 
            'backward_optim': self.bwd_optim.state_dict(), 
        }

        torch.save(checkpoint, checkpoint_path / f'checkpoint-{sb_iter}.pth')

    def resotre_from_last_checkpoint(self, run):
        checkpoint_path = find_checkpoint(
            Path(run.dir) / 'checkpoints', 
            self.config.restore_from
        )
        
        if checkpoint_path is None:
            print("Checkpoint not found. Train from scratch!")
            return
        
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
        
        # Handle restoration of a checkpoint
        print(f"Restoring checkpoint from {checkpoint_path.name}")

        self.fwd_model.load_state_dict(checkpoint['forward'])
        self.bwd_model.load_state_dict(checkpoint['backward'])
        self.fwd_optim.load_state_dict(checkpoint['forward_optim'])
        self.bwd_optim.load_state_dict(checkpoint['backward_optim'])

        # log that checkpoint is restored successfully
        print(f"Checkpoint successfully restored!")

 
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
