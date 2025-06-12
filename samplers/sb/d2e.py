from dataclasses import dataclass

import torch
import wandb
from tqdm.auto import trange
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

import utils
import metrics

from samplers import losses
from samplers import utils as sutils
from buffers import ReplayBuffer, LangevinReplayBuffer

from . import base_class


@dataclass
class D2ESBConfig(base_class.SBConfig):
    drift_reg_coeff: float = 0.0
    reuse_backward_trajectory: bool = True
    n_trajectories: int = 2
    buffer_type: str = 'simple'
    val_batch_size: int = 1024
    buffer_size: int = 1024
    langevin_step_size: float = 0.001
    num_langevin_update_steps: int = 10


class D2ESB(base_class.SB):
    def __init__(self, fwd_model, bwd_model, p0, p1, config):
        super().__init__(fwd_model, bwd_model, p0, p1, config)
        
        if config.buffer_type == 'simple':
            self.p1_buffer = ReplayBuffer(config.buffer_size)
        elif config.buffer_type == "langevin":
            self.p1_buffer = LangevinReplayBuffer(
                config.buffer_size, self.p1.grad_log_density, 
                config.langevin_step_size, config.num_langevin_update_steps
            )
        else:
            raise ValueError(f"Buffer is unknow: {config.buffer_type}")

    def train_backward_step(self, sb_iter, run):
        dt = self.config.dt
        t_max = self.config.t_max
        n_steps = self.config.n_steps

        for step_iter in trange(self.config.num_bwd_steps, leave=False, 
                                desc=f'It {sb_iter} | Backward'):
            self.bwd_optim.zero_grad(set_to_none=True)

            x_0 = self.p0.sample(self.config.batch_size).to(self.config.device)
            fwd_model = self.reference_process if sb_iter == 0 else self.fwd_model
            loss = losses.compute_bwd_tlm_loss(fwd_model, self.bwd_model, 
                                               x_0, dt, t_max, n_steps)
                
            self.bwd_optim.step()
            self.bwd_ema_loss.update(loss.item() / n_steps)
            run.log({
                "train/backward_loss": loss / n_steps,
                "bwd_step": sb_iter * self.config.num_bwd_steps + step_iter
            })
                    
    def train_forward_step(self, sb_iter, run):
        dt = self.config.dt
        t_max = self.config.t_max
        n_steps = self.config.n_steps

        device = self.config.device
        batch_size = self.config.batch_size
        n_trajectories = self.config.n_trajectories

        drift_reg_coeff = self.config.drift_reg_coeff
                
        for step_iter in trange(self.config.num_fwd_steps, 
                                leave=False, desc=f'It {sb_iter} | Forward'):
            self.fwd_optim.zero_grad(set_to_none=True)

            if sb_iter == 0:
                x_0 = self.p0.sample(batch_size // n_trajectories).to(device)
                x_0 = x_0.repeat(n_trajectories, 1)
                loss = losses.compute_fwd_vargrad_loss(self.fwd_model, self.bwd_model, 
                                                       self.p1.log_density, x_0, dt, 
                                                       t_max, n_steps, 
                                                       p1_buffer=self.p1_buffer,
                                                       n_trajectories=n_trajectories,
                                                    #    reg_coeff=drift_reg_coeff
                                                       )

            elif self.config.reuse_backward_trajectory:
                raise ValueError('This part of code is under rewriting. DO NOT USE!')
                x_1 = self.p1_buffer.sample(batch_size).to(device)
                loss = losses.compute_fwd_ctb_loss_reuse_bwd(
                    self.fwd_model, self.bwd_model,
                    self.p1.log_density, x_1, dt, 
                    t_max, n_steps, self.p1_buffer
                )

            else:
                x_1 = self.p1_buffer.sample(batch_size // n_trajectories).to(device)
                x_0 = sutils.sample_trajectory(self.bwd_model, x_1,"backward", 
                                               dt, n_steps, t_max, only_last=True)
                x_0 = x_0.repeat(n_trajectories, 1)
                loss = losses.compute_fwd_vargrad_loss(self.fwd_model, self.bwd_model, 
                                                       self.p1.log_density, x_0, dt, 
                                                       t_max, n_steps, 
                                                       p1_buffer=self.p1_buffer,
                                                       n_trajectories=n_trajectories,
                                                    #    reg_coeff=drift_reg_coeff
                                                       )
            
            loss.backward()
            self.fwd_optim.step()

            self.fwd_ema_loss.update(loss.mean().item() / n_steps)
            run.log({
                "train/forward_loss": loss / n_steps,
                "fwd_step": sb_iter * self.config.num_fwd_steps + step_iter
            })

    @torch.no_grad()
    def log_forward_step(self, sb_iter, run):
        dt = self.config.dt
        t_max = self.config.t_max
        n_steps = self.config.n_steps

        # x_0 = self.p0.sample(self.config.batch_size).to(self.config.device)
        # trajectory, timesteps = sutils.sample_trajectory(self.fwd_model, x_0, "forward", 
        #                                                 dt, n_steps, t_max, 
        #                                                 only_last=True,
        #                                                 return_timesteps=False)

        # trajectory = [tensor.cpu() for tensor in trajectory]
        # figure = utils.plot_trajectory(trajectory, timesteps, 
        #                                title=f"Forward Process, step={sb_iter}")

        x_0 = self.p0.sample(self.config.val_batch_size).to(self.config.device)
        elbo, iw_1, iw_2 = metrics.compute_elbo(self.fwd_model, self.bwd_model, 
                                                self.p1.log_density, x_0, dt, t_max, 
                                                n_steps, n_traj=16)

        img_base = self.p1.reward.decoder(x_0[:36]).cpu()
        img_base = make_grid(
            img_base.view(-1, 1, 28, 28).repeat(1, 3, 1, 1), 
            nrow=6, normalize=True
        )[:1]
        
        x1_pred = sutils.sample_trajectory(self.fwd_model, x_0, 'forward', 
                                           dt, n_steps, t_max, only_last=True)

        mean_reward = self.p1.reward(x1_pred).log().mean(dim=0)
        img_pred = self.p1.reward.decoder(x1_pred[:36]).cpu()
        img_pred = make_grid(
            img_pred.view(-1, 1, 28, 28).repeat(1, 3, 1, 1), 
            nrow=6, normalize=True
        )[:1]

        # x_1_true = self.p1.sample(self.config.batch_size // 4)
        # w2_dist = metrics.compute_w2_distance(x_1_true, x_1_sampled)

        run.log({
            "images/x0_sample": wandb.Image(img_base),
            "images/x1-sample": wandb.Image(img_pred), 
            "sb_iter": sb_iter,
            "metrics/p1_elbo": elbo, 
            "metrics/mean_reward": mean_reward,
            "metrics/p1_iw_1": iw_1, 
            "metrics/p1_iw_2": iw_2,
            # "metrics/w2_dist": w2_dist,
        })
        # plt.close(figure)

    @torch.no_grad()
    def log_backward_step(self, sb_iter, run):
        return None
        dt = self.config.dt
        t_max = self.config.t_max
        n_steps = self.config.n_steps
        
        if sb_iter == 0:
            x_0  = self.p0.sample(512).to(self.config.device)
            x_1 = sutils.sample_trajectory(self.reference_process, x_0, "forward", 
                                           dt, n_steps, t_max, only_last=True)
        else:
            x_1 = self.p1_buffer.sample(512).to(self.config.device)
        
        trajectory, timesteps = sutils.sample_trajectory(self.bwd_model, x_1, "backward", 
                                                        dt, n_steps, t_max, 
                                                        return_timesteps=True)
        
        trajectory = [tensor.cpu() for tensor in trajectory]
        figure = utils.plot_trajectory(trajectory[::-1], timesteps[::-1], 
                                       title=f"Backward Process, step={sb_iter}")
        
        run.log({"images/backward_trajectory": wandb.Image(figure), "sb_iter": sb_iter})
        plt.close(figure)
