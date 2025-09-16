
import functools

import math
import torch
import wandb
from tqdm.auto import trange
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

import sb.utils as utils
import sb.metrics as metrics
import sb.losses as losses

from . import d2e_2d
from . import utils as sutils


class DiffusionSampler2D(d2e_2d.D2ESB_2D):
    def __init__(self, fwd_model, bwd_model, p0, p1, config, buffer_config):
        super().__init__(fwd_model, bwd_model, p0, p1, config, buffer_config)
        self.config.backward_first = False 


    def train_forward_step(self, sb_iter, run):
        self.fwd_model.train()

        dt = self.config.dt
        t_max = self.config.t_max
        alpha = self.config.alpha
        n_steps = self.config.n_steps

        device = self.config.device
        batch_size = self.config.batch_size
        n_trajectories = self.config.n_trajectories

        for step_iter in trange(self.config.num_fwd_steps, 
                                leave=False, desc=f'It {sb_iter} | Forward'):
            self.fwd_optim.zero_grad(set_to_none=True)
            
            _batch_size = batch_size // n_trajectories \
                if n_trajectories > 0 else batch_size
            
            x0 = self.p0.sample(_batch_size).to(device)
            hdim = x0.size(1)

            # run langevin on p1
            if self.buffer_config.buffer_type == "decoupled_langevin" \
                and self.config.off_policy_fraction > 0.0:
                raise NotImplementedError("Langevin not implemented yet.")
                # self.p1_buffer.run_langevin(hdim)

            # compute beta for forward sampler
            beta = 1.0
            if self.anneal_beta_fn:
                beta = self.anneal_beta_fn(step_iter, self.config.num_fwd_steps)

            # compute off-policy x0
            reuse_trajectory_loss = None
            if self.config.off_policy_fraction > 0.0:
                raise NotImplementedError("Off-policy training not implemented yet.")

            if self.config.off_policy_fraction == 1.0:
                x1 = self.p1_buffer.sample(_batch_size, hdim).to(device)
                x0 = sutils.sample_trajectory(
                    self.bwd_model, x1,"backward", dt, n_steps, t_max, 
                    only_last=True
                )

            elif self.config.off_policy_fraction > 0 and sb_iter > 0:
                num_off_policy_samples = int(
                    self.config.off_policy_fraction * (_batch_size)
                )
                x1 = self.p1_buffer.sample(num_off_policy_samples, hdim).to(device)
                x0_off = sutils.sample_trajectory(
                    self.bwd_model, x1, dt, 
                    t_max, n_steps, self.config.alpha, self.config.var,
                    direction='bwd',
                    only_last=True
                )

                x0_on = x0[:_batch_size - num_off_policy_samples].to(device)
                x0 = torch.cat([x0_off, x0_on], dim=0)   

            # compute forward loss
            density_fn = functools.partial(self.p1.log_density, anneal_beta=beta)
            loss = losses.compute_relative_tb_loss(
                self.fwd_model, density_fn, x0, dt, t_max, n_steps, 
                self.config.alpha, self.config.var
            )

            with torch.no_grad():
                x0 = self.p0.sample(self.config.val_batch_size).to(device)
                
                x1 = sutils.sample_trajectory(
                    self.fwd_model,
                    x0, dt, t_max, n_steps, self.config.alpha, self.config.var,
                    direction="fwd",
                    only_last=True,
                    method=self.config.matching_method,
                )
                mean_log_reward = self.p1.get_mean_log_reward(x1).mean()
            
            if n_trajectories > 0:
                loss.backward()
            
            self.fwd_optim.step()
            self.fwd_model_ema.update()

            run.log({
                "train/forward_loss": loss / n_steps,
                "metrics/mean_log_reward": mean_log_reward,
                "fwd_step": sb_iter * self.config.num_fwd_steps + step_iter
            })

    @torch.no_grad()
    def log_forward_step(self, sb_iter, run):
        dt = self.config.dt
        t_max = self.config.t_max
        n_steps = self.config.n_steps

        device = self.config.device

        # Compute metrics
        x0 = self.p0.sample(self.config.metric_batch_size).to(device)
        path_kl = metrics.compute_path_kl(
            self.fwd_model, 
            x0, dt, t_max, n_steps, self.config.alpha, self.config.var,
            method=self.config.matching_method
        )
        x1_pred = sutils.sample_trajectory(
            self.fwd_model,
            x0, dt, t_max, n_steps, self.config.alpha, self.config.var,
            direction="fwd",
            only_last=True,
        )
        x0_x1_transport_cost = (x1_pred - x0).pow(2).sum(1).mean()
        logging_dict = {
            # "metrics/p1_elbo": elbo,
            "metrics/x0_x1_transport_cost": x0_x1_transport_cost,
            "metrics/Path_KL": path_kl,
            "sb_iter": sb_iter,
        }
        try:
            x1_true = self.p1.sample(self.config.val_batch_size).to(device)
            w2 = metrics.compute_w2_distance(x1_true, x1_pred)
            logging_dict = logging_dict | {
                "metrics/W2": w2
            }
        
        except NotImplementedError:
            pass

        # Log trajectories
        x0 = x0[:self.config.val_batch_size]
        trajectory, timesteps = sutils.sample_trajectory(
            self.fwd_model, 
            x0, dt, t_max, n_steps, self.config.alpha, self.config.var,
            direction="fwd", 
            only_last=False,
            return_timesteps=True, 
            method=self.config.matching_method
        )
        fwd_trajectory_fig = utils.plot_trajectory(
            [tensor.cpu() for tensor in trajectory], timesteps, 
            title=f"Forward Process, step={sb_iter}",
            limits=tuple(self.config.plot_limits)
        )
        logging_dict = logging_dict | {
            "images/forward_trajectory": wandb.Image(fwd_trajectory_fig)
        }
        
        run.log(logging_dict)
        plt.close("all")
        
    def train_backward_step(self, sb_iter, run):
        pass

    @torch.no_grad
    def log_backward_step(self, sb_iter, run):
        pass
