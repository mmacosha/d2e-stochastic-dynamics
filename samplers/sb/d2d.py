from dataclasses import dataclass
from pathlib import Path

import wandb
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm, trange


import utils
from samplers import losses
from model import ModelOutput


def get_mean_log_var(model, x, t, dt):
    log_var = torch.as_tensor(2.0 * dt).log()
    output = model(x, t)
    if output.contains('log_var'):
        log_var = log_var + output.log_var
    
    mean = x + output.drift * dt
    return mean, log_var


class ReferenceProcess:
    def __init__(self, alpha: float, gamma: float):
        self.alpha = alpha
        self.gamma = gamma
    
    def __call__(self, x, t):
        return ModelOutput(drift=-self.alpha * self.gamma * x)


@dataclass
class SBConfig:
    alpha: float = 4.0
    gamma: float = 0.0006
    batch_size: int = 256
    t_max: float = 0.0012
    n_sampling_steps: int = 20
    num_sb_steps: int = 20
    threshold: float = 2e-6
    lr_forward: float = 8e-4
    lr_backward: float = 8e-4
    save_checkpoint_freq: int = 5
    max_num_iters_per_step: int = 5000


class SBTrainer:
    def __init__(
            self, 
            F, B, 
            sampler,
            config,
            wandb_config,
        ):
        self.config = config
        self.wandb_config = wandb_config

        self.data_sampler = sampler

        self.reference_process = ReferenceProcess(
            self.config.alpha, self.config.gamma
        )

        self.fwd_model = F
        self.bwd_model = B

        self.fwd_optimizer = torch.optim.Adam(
            self.fwd_model.parameters(), 
            lr=self.config.lr_forward
        )
        self.bwd_optimizer = torch.optim.Adam(
            self.bwd_model.parameters(), 
            lr=self.config.lr_backward
        )

    def train_forward(self, curr_iter: int, run):
        ema_loss = utils.EMALoss()
        var_crit = utils.VarCriterion(
            ema_loss.ema, 
            measure_size=500,
            threshold=self.config.threshold, 
            max_iter=self.config.max_num_iters_per_step, 
        )
        dt = self.config.gamma
        
        with tqdm(leave=False, desc=f'step {curr_iter + 1}, training F model') as pbar:
            while var_crit.check():
                traj_loss = 0
                self.fwd_optimizer.zero_grad(set_to_none=True)

                _, x_t = self.data_sampler.sample(self.config.batch_size)
                for t_step in torch.linspace(
                        start=self.config.gamma, end=self.config.t_max, 
                        steps=self.config.n_sampling_steps
                    ).flip(-1):
                    t = torch.ones(self.config.batch_size) * t_step

                    with torch.no_grad():
                        bwd_mean, bwd_log_var = get_mean_log_var(
                            self.reference_process if curr_iter == 0 else self.bwd_model,
                            x_t, t, self.config.gamma
                        )
                        x_t_m_dt = bwd_mean \
                            + torch.rand_like(bwd_log_var) * bwd_log_var.exp().sqrt()

                    fwd_mean, fwd_log_var = get_mean_log_var(self.fwd_model, x_t_m_dt, t - dt, dt)
                    loss = losses.log_normal_density(x_t, fwd_mean, fwd_log_var)

                    (-loss).mean().backward()
                    traj_loss += (-loss).mean().item()

                    x_t = x_t_m_dt
                run.log({f"train/forward_loss_{curr_iter}": traj_loss})

                ema_loss.update(traj_loss / self.config.n_sampling_steps)
                self.fwd_optimizer.step()

                pbar.update(1)
            pbar.close()
         
    def train_backward(self, curr_iter: int, run):
        ema_loss = utils.EMALoss()
        var_crit = utils.VarCriterion(
            ema_loss.ema, 
            measure_size=500,
            threshold=self.config.threshold, 
            max_iter=self.config.max_num_iters_per_step,
        )
        dt = self.config.gamma

        with tqdm(leave=False, desc=f'step {curr_iter + 1}, training B model') as pbar:
            while var_crit.check():
                traj_loss = 0
                self.bwd_optimizer.zero_grad(set_to_none=True)
                
                x_t_m_dt, _ = self.data_sampler.sample(self.config.batch_size)
                for t_step in torch.linspace(
                        start=self.config.gamma, end=self.config.t_max, 
                        steps=self.config.n_sampling_steps
                    ).flip(-1):
                    t = torch.ones(self.config.batch_size) * t_step
                    
                    with torch.no_grad():
                        fwd_mean, fwd_log_var = get_mean_log_var(
                            self.fwd_model, x_t_m_dt, t - dt, dt
                        )
                        x_t = fwd_mean \
                            + torch.randn_like(fwd_log_var) * fwd_log_var.exp().sqrt()
                    
                    bwd_mean, bwd_log_var = get_mean_log_var(self.bwd_model, x_t, t, dt)
                    loss = losses.log_normal_density(x_t_m_dt, bwd_mean, bwd_log_var)                        

                    (-loss).mean().backward()
                    traj_loss += (-loss).mean().item()

                    x_t_m_dt = x_t
                
                run.log({f"train/backward_loss_{curr_iter}": traj_loss})

                ema_loss.update(traj_loss / self.config.n_sampling_steps)
                self.bwd_optimizer.step()
                
                pbar.update(1)
            pbar.close()

    @torch.no_grad()
    def sample_backward(self, curr_step, run=None, return_trajectory=False):
        x_0, x_1 = self.data_sampler.sample(self.config.batch_size)
        dt = self.config.gamma

        trajectory, timesteps = [x_1], []
        for t_step in torch.linspace(
                self.config.gamma, self.config.t_max, self.config.n_sampling_steps
            ).flip(-1):
            
            timesteps.append(t_step.item())
            t = torch.ones(x_1.size(0)) * t_step

            bwd_mean, bwd_log_var = get_mean_log_var(self.bwd_model, trajectory[-1], t, dt)
            x_next = bwd_mean + torch.rand_like(bwd_log_var) * bwd_log_var.exp().sqrt()
            trajectory.append(x_next)
        
        timesteps.append(0)

        timesteps.append('target')
        trajectory.append(x_0)

        figure = utils.plot_trajectory(trajectory, timesteps)

        if run is not None:
            run.log({'images/backward_trajectory': wandb.Image(figure)}, step=curr_step)

        if return_trajectory:
            return trajectory

    @torch.no_grad()
    def sampler_forward(self, curr_step, run=None, return_trajectory=False):
        x_0, x_1 = self.data_sampler.sample(self.config.batch_size)
        dt = self.config.gamma

        trajectory, timesteps = [x_0], []
        for t_step in torch.linspace(
                0, self.config.t_max - dt, self.config.n_sampling_steps
            ):
            
            timesteps.append(t_step.item())
            t = torch.ones(x_0.size(0)) * t_step

            fwd_mean, fwd_log_var = get_mean_log_var(self.fwd_model, trajectory[-1], t, dt)
            x_next = fwd_mean + torch.randn_like(fwd_log_var) * fwd_log_var.exp().sqrt()
            trajectory.append(x_next)
        
        timesteps.append(self.config.t_max)

        timesteps.append('target')
        trajectory.append(x_1)

        figure = utils.plot_trajectory(trajectory, timesteps)
        
        if run is not None:
            run.log({'images/forward_trajectory': wandb.Image(figure)}, step=curr_step)
        
        if return_trajectory:
            return trajectory

    def save_checkpoint(self, run, curr_iter):
        checkpoint_path = Path(run.dir) / 'checkpoints'
        checkpoint_path.mkdir(exist_ok=True)

        checkpoint = {
            'forward': self.fwd_model.state_dict(), 
            'backward': self.bwd_model.state_dict(), 
            'forward_optim': self.fwd_optimizer.state_dict(), 
            'backward_optim': self.bwd_optimizer.state_dict(), 
        }

        torch.save(checkpoint, checkpoint_path / f'checkpoint-{curr_iter}.pth')

    def train(self):
        with wandb.init(**self.wandb_config, config=self.config) as run:
            for curr_iter in trange(self.config.num_sb_steps):
                self.train_forward(curr_iter, run)
                self.sampler_forward(curr_iter, run)
                
                self.train_backward(curr_iter, run)
                self.sample_backward(curr_iter, run)
            
                if curr_iter % self.config.save_checkpoint_freq == 0:
                        self.save_checkpoint(run, curr_iter)

            self.save_checkpoint(run, curr_iter)