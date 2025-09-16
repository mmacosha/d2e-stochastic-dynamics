import torch
import wandb

from matplotlib import pyplot as plt

from sb import utils, metrics

from . import d2e
from . import utils as sutils


class D2ESB_2D(d2e.D2ESB):
    @torch.no_grad()
    def log_forward_step(self, sb_iter, run):
        dt = self.config.dt
        t_max = self.config.t_max
        n_steps = self.config.n_steps

        device = self.config.device

        # Compute metrics
        x0 = self.p0.sample(self.config.metric_batch_size).to(device)
        elbo, iw_1, iw_2 = metrics.compute_elbo(
            self.fwd_model, self.bwd_model, 
            self.p1.log_density, 
            x0, dt, t_max, n_steps, 
            n_traj=4
        )
        eubo = metrics.compute_eubo(
            self.fwd_model, self.bwd_model, 
            self.p1.log_density, 
            x0, dt, t_max, n_steps
        )
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
            "metrics/p1_elbo": elbo,
            "metrics/p1_eubo": eubo,
            "metrics/p1_iw_1": iw_1, 
            "metrics/p1_iw_2": iw_2,
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
        x1_true = self.p1.sample(self.config.val_batch_size).to(self.config.device)
        
        W2 = metrics.compute_w2_distance(
            x1_true,  x1_pred.to(self.config.device)
        )
        path_kl = metrics.compute_path_kl(
            self.fwd_model, 
            x0, dt, t_max, n_steps, self.config.alpha, self.config.var,
            method=self.config.matching_method
        )
        x0_x1_transport_cost = (x1_pred - x0).pow(2).sum(1).mean()
        final_metrics = {
            "p1_W2": W2,
            "Path_KL": path_kl,
            "x0_x1_transport_cost": x0_x1_transport_cost,
        }

        wandb.log(final_metrics)

        for k, v in final_metrics.items():
            wandb.run.summary[k] = v
