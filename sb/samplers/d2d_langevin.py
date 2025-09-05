import torch
import wandb
from tqdm.auto import trange
import matplotlib.pyplot as plt

import sb.utils as utils
import sb.metrics as metrics
import sb.losses as losses
from sb.buffers import DecoupledLangevinBuffer

from . import d2d
from . import utils as sutils


class D2DSBLangevin(d2d.D2DSB):
    def __init__(self, fwd_model, bwd_model, p0, p1, config, buffer_config):
        super().__init__(fwd_model, bwd_model, p0, p1, config)
        self.buffer_config = buffer_config

        if buffer_config.buffer_type == "decoupled_langevin":    
            self.p1_buffer = DecoupledLangevinBuffer(
                p1=self.p1, 
                device=config.device,
                **buffer_config
            )
        else:
            raise ValueError(
                f"Buffer {buffer_config.buffer_type} is not appropriate ",
                "for D2D Langevin sampler."
            )

    def train_forward_step(self, sb_iter, run):        
        dt = self.config.dt
        t_max = self.config.t_max
        n_steps = self.config.n_steps
        
        self.bwd_model_ema.apply()
        for step_iter in trange(self.config.num_fwd_steps, leave=False, 
                               desc=f"It {sb_iter} | Forward"):
            self.fwd_optim.zero_grad(set_to_none=True)

            x0 = self.p0.sample(self.config.batch_size).to(self.config.device)
            self.p1_buffer.run_langevin(x0.size(1))
            x1 = self.p1_buffer.sample(
                self.config.batch_size
            ).to(self.config.device)
            
            loss = losses.compute_fwd_tlm_loss_v2(
                self.fwd_model, self.bwd_model, 
                x0, x1, dt, t_max, n_steps, self.config.alpha, self.config.var, 
                begin=sb_iter==0 and not self.config.backward_first,
                backward=True, 
                method=self.config.matching_method
            )

            run.log({
                "train/forward_loss": loss / n_steps,
                "fwd_step": sb_iter * self.config.num_fwd_steps + step_iter
            })

            self.fwd_optim.step()
            self.fwd_model_ema.update()

        self.bwd_model_ema.restore()

    @torch.no_grad()
    def log_forward_step(self, sb_iter, run):
        dt = self.config.dt
        t_max = self.config.t_max
        n_steps = self.config.n_steps

        x0 = self.p0.sample(self.config.val_batch_size).to(self.config.device)
        trajectory, timesteps = sutils.sample_trajectory(
            self.fwd_model,
            x0, dt, t_max, n_steps, self.config.alpha, self.config.var, 
            direction="fwd",
            return_timesteps=True, 
            method=self.config.matching_method
        )

        figure = utils.plot_trajectory(
            [tensor.cpu() for tensor in trajectory], timesteps, 
            title=f"Forward Process, step={sb_iter}",
            limits=(-3, 3)
        )
        x1_true = self.p1.sample(self.config.val_batch_size).to(self.config.device)
        x1_pred = trajectory[-1]
        w2 = metrics.compute_w2_distance(x1_true,  x1_pred)
        path_energy = metrics.compute_path_kl(
            self.fwd_model, 
            x0, dt, t_max, n_steps, self.config.alpha, self.config.var, 
            self.config.matching_method
        )
        run.log({
            "metrics/W2": w2,
            "metrics/path_energy": path_energy,
            "images/forward_trajectory": wandb.Image(figure), 
            "sb_iter": sb_iter
        })
        plt.close(figure)

    @torch.no_grad()
    def log_backward_step(self, sb_iter, run):
        dt = self.config.dt
        t_max = self.config.t_max
        n_steps = self.config.n_steps

        x1 = self.p1_buffer.sample(self.config.val_batch_size).to(self.config.device)
        trajectory, timesteps = sutils.sample_trajectory(
            self.fwd_model, 
            x1, dt, t_max, n_steps, self.config.alpha, self.config.var, 
            direction="bwd",
            return_timesteps=True, 
            method=self.config.matching_method
        )
        figure = utils.plot_trajectory(
            [tensor.cpu() for tensor in trajectory][::-1], timesteps[::-1], 
            title=f"Backward Process, step={sb_iter}",
            limits=(-3, 3)
        )
        run.log({
            "images/backward_trajectory": wandb.Image(figure), 
            "sb_iter": sb_iter
        })
        plt.close(figure)

    @torch.no_grad()
    def log_final_metric(self, run):
        dt = self.config.dt
        t_max = self.config.t_max
        n_steps = self.config.n_steps

        x0 = self.p0.sample(self.config.metric_batch_size).to(self.config.device)
        x1_pred = sutils.sample_trajectory(
            self.fwd_model, 
            x0, dt, t_max, n_steps, self.config.alpha, self.config.var, 
            direction="fwd",
            only_last=True,
            return_timesteps=False, 
            method=self.config.matching_method
        )
        x1_true = self.p1_buffer.sample(
            self.config.metric_batch_size
        ).to(self.config.device)
        w2 = metrics.compute_w2_distance(
            x1_true,  x1_pred.to(self.config.device)
        )
        path_kl = metrics.compute_path_kl(
            self.fwd_model, 
            x0, dt, t_max, n_steps, self.config.alpha, self.config.var,
            method=self.config.matching_method
        )
        final_metrics = {
            "p1_W2": w2,
            "Path_KL": path_kl,
        }

        wandb.log(final_metrics)

        for k, v in final_metrics.items():
            wandb.run.summary[k] = v
