import torch
import wandb
from tqdm.auto import trange
import matplotlib.pyplot as plt

import sb.utils as utils

from . import losses
from . import utils as sutils
from . import base_class


class D2DSB(base_class.SB):
    def train_forward_step(self, sb_iter, run):        
        dt = self.config.dt
        t_max = self.config.t_max
        n_steps = self.config.n_steps

        self.bwd_model_ema.apply()
        for step_iter in trange(self.config.num_fwd_steps, leave=False, 
                               desc=f"It {sb_iter} | Forward"):
            self.fwd_optim.zero_grad(set_to_none=True)

            x_1 = self.p1.sample(self.config.batch_size).to(self.config.device)
            loss = losses.compute_fwd_tlm_loss(self.fwd_model, self.bwd_model, x_1, 
                                               dt, t_max, n_steps, backward=True)

            run.log({
                "train/forward_loss": loss / n_steps,
                "fwd_step": sb_iter * self.config.num_fwd_steps + step_iter
            })

            self.fwd_optim.step()
            self.fwd_model_ema.update()

        self.bwd_model_ema.restore()

    def train_backward_step(self, sb_iter, run):
        dt = self.config.dt
        t_max = self.config.t_max
        n_steps = self.config.n_steps

        self.fwd_model_ema.apply()
        for step_iter in trange(self.config.num_bwd_steps, leave=False,
                                desc=f"It {sb_iter} | Backward"):
            self.bwd_optim.zero_grad(set_to_none=True)
            
            x_0 = self.p0.sample(self.config.batch_size).to(self.config.device)
            loss = losses.compute_bwd_tlm_loss(self.fwd_model, self.bwd_model, x_0, 
                                               dt, t_max, n_steps, backward=True)
            
            run.log({
                "train/backward_loss": loss / n_steps,
                "bwd_step": sb_iter * self.config.num_bwd_steps + step_iter
            })

            self.bwd_optim.step()
            self.bwd_model_ema.update()

        self.fwd_model_ema.restore()

    @torch.no_grad()
    def log_forward_step(self, sb_iter, run):
        dt = self.config.dt
        t_max = self.config.t_max
        n_steps = self.config.n_steps
        
        x_0 = self.p0.sample(self.config.batch_size).to(self.config.device)
        trajectory, timesteps = sutils.sample_trajectory(self.fwd_model, x_0, "forward", 
                                                        dt, n_steps, t_max, 
                                                        return_timesteps=True)
        trajectory = [tensor.cpu() for tensor in trajectory]
        figure = utils.plot_trajectory(trajectory, timesteps, 
                                       title=f"Forward Process, step={sb_iter}")

        run.log({
            "images/forward_trajectory": wandb.Image(figure), 
            "sb_iter": sb_iter
        })
        plt.close(figure)

    @torch.no_grad()
    def log_backward_step(self, sb_iter, run):
        dt = self.config.dt
        t_max = self.config.t_max
        n_steps = self.config.n_steps

        x_1 = self.p1.sample(self.config.batch_size).to(self.config.device)
        trajectory, timesteps = sutils.sample_trajectory(self.bwd_model, x_1, "backward", 
                                                         dt, n_steps, t_max, 
                                                         return_timesteps=True)
        trajectory = [tensor.cpu() for tensor in trajectory]
        figure = utils.plot_trajectory(trajectory[::-1], timesteps[::-1], 
                                       title=f"Backward Process, step={sb_iter}")
        run.log({
            "images/backward_trajectory": wandb.Image(figure), 
            "sb_iter": sb_iter
        })
        plt.close(figure)
