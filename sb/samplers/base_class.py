from typing import Optional
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import torch
import wandb
from omegaconf import OmegaConf

from tqdm import trange

import sb.utils as utils
from sb.samplers.utils import ReferenceProcess2


@dataclass
class SBConfig:
    name: str  = None 
    device: str = "cpu"
    backward_first: bool = False
    num_sb_steps: int = 10
    num_fwd_steps: int = 6000
    num_bwd_steps: int = 6000
    threshold: float = 5e-6

    matching_method: str = "mean"
    train_log_var: bool = True

    fwd_ema_decay: float = 0.999
    bwd_ema_decay: float = 0.999
    fwd_optim_lr: int = 1e-3
    bwd_optim_lr: int = 1e-3
    log_fwd_freq: int = 1
    log_bwd_freq: int = 1
    log_img_freq: int = 10

    alpha: float = 4.0
    dt: float = 0.0006
    t_max: float = 0.012
    n_steps: int = 20

    batch_size: int = 512
    val_batch_size: Optional[int] = None
    save_checkpoint_freq: int = 100
    restore_from: int = -1

    def __post__init__(self):
        self.val_batch_size = self.val_batch_size or self.batch_size
        assert self.dt * self.n_steps == self.t_max

        assert self.matching_method in {"mean", "score", "ll"}, \
            f"Unknown matching method: {self.matching_method}. " \
            f"Available methods: mean, score, ll."
        

def find_checkpoint(run_dir: str, checkpoint_num: int, run_id: str) -> str :
    """
    Get the checkpoint file from the given directory.
    If checkpoint_num is -1, return the last checkpoint.
    Otherwise, return the checkpoint corresponding to the checkpoint_num.
    
    Args:
        run_dir (str): The directory containing checkpoint files.
        checkpoint_num (int): The checkpoint number to retrieve (-1 for the latest checkpoint).
        run_id (str): The ID of the run to which the checkpoints belong.
    
    Returns:
        str: The path to the selected checkpoint file.
    """
    if checkpoint_num != -1:
        raise NotImplementedError("Cannot restore from a specific checkpoint.")
    
    run_dir = Path(run_dir)
    wandb_dir = run_dir.parent.parent
    previous_run_with_same_id = [*wandb_dir.glob(f"*{run_id}")]
    
    if not previous_run_with_same_id:
        return None, -1
    
    possible_checkpoints = []
    for possible_run_dir in previous_run_with_same_id:
        checkpoints_path = possible_run_dir / "files" / "checkpoints"
        for ckpt in checkpoints_path.glob("checkpoint-*.pth"):
            possible_checkpoints.append(ckpt)

    if not possible_checkpoints:
        return None, -1

    checkpoint_path = sorted(
        possible_checkpoints, 
        key=lambda x: int(re.search(r'checkpoint-(\d+).pth', x.name).group(1)),
        reverse=True
    )[0]
    sb_step = int(
        re.search(r'checkpoint-(\d+).pth', checkpoint_path.name).group(1)
    )
    return checkpoint_path, sb_step


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

        self.reference_process = ReferenceProcess2(
            alpha=self.config.alpha,
            dt=self.config.dt,
            method=self.config.matching_method,
        )

    def train(self, experiment, full_cfg=None):
        experiment.name = f"{self.config.name}-{experiment.name}"
        with wandb.init(**experiment, config=self.config) as run:
            if full_cfg is not None:
                cfg_path = f"{run.dir}/full_run_config.yaml"
                OmegaConf.save(full_cfg, cfg_path)
                artifact = wandb.Artifact("run-config", type="config")
                artifact.add_file(cfg_path)
                wandb.log_artifact(artifact)

            sb_step = self.resotre_from_last_checkpoint(run)
            for sb_iter in trange(sb_step + 1, self.config.num_sb_steps, leave=False, 
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
        wandb.save(checkpoint_path / f'checkpoint-{sb_iter}.pth')

    def resotre_from_last_checkpoint(self, run):
        checkpoint_path, sb_step = find_checkpoint(
            run.dir, self.config.restore_from, run.id
        )
        
        if checkpoint_path is None:
            print("Checkpoint not found. Train from scratch!")
            return sb_step
        
        checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
        
        # Handle restoration of a checkpoint
        print(f"Restoring checkpoint from {checkpoint_path.name}")

        self.fwd_model.load_state_dict(checkpoint['forward'])
        self.bwd_model.load_state_dict(checkpoint['backward'])
        self.fwd_optim.load_state_dict(checkpoint['forward_optim'])
        self.bwd_optim.load_state_dict(checkpoint['backward_optim'])

        # log that checkpoint is restored successfully
        print(f"Checkpoint successfully restored!")
        return sb_step

 
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
