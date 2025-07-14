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
        val_batch_size = self.config.val_batch_size
        device = self.config.device

        # sample x0 and compute metrics
        x0 = self.p0.sample(val_batch_size).to(device)
        elbo, iw_1, iw_2 = metrics.compute_elbo(self.fwd_model, self.bwd_model, 
                                                self.p1.log_density, x0, dt, 
                                                t_max, n_steps, n_traj=16)
        logging_dict = {
            "metrics/p1_elbo": elbo, 
            "metrics/p1_iw_1": iw_1, 
            "metrics/p1_iw_2": iw_2,
            "sb_iter": sb_iter,
        }

        trajectory, timesteps = sutils.sample_trajectory(
            self.fwd_model, x0, "forward", dt, n_steps, t_max, 
            return_timesteps=True, matching_method=self.config.matching_method
        )
        trajectory = [tensor.cpu() for tensor in trajectory]
        fig = utils.plot_trajectory(trajectory, timesteps, 
                                    title=f"Forward Process, step={sb_iter}",
                                    limits=(-2, 2))
        logging_dict["images/forward_trajectory"] = wandb.Image(fig)
        
        try:
            x1_true = self.p1.sample(val_batch_size).to(device)
            W2 = metrics.compute_w2_distance(
                x1_true, 
                trajectory[-1].to(device)
            )
            logging_dict["metrics/W2"] = W2
        
        except NotImplementedError:
            pass
        
        run.log(logging_dict)
        plt.close("all")

    @torch.no_grad()
    def log_backward_step(self, sb_iter, run):
        dt = self.config.dt
        t_max = self.config.t_max
        n_steps = self.config.n_steps
        
        if sb_iter == 0:
            x0  = self.p0.sample(512).to(self.config.device)
            x1 = sutils.sample_trajectory(self.reference_process, x0, "forward", 
                                           dt, n_steps, t_max, only_last=True)
        else:
            x1 = self.p1_buffer.sample(512).to(self.config.device)
        
        trajectory, timesteps = sutils.sample_trajectory(self.bwd_model, x1, "backward", 
                                                        dt, n_steps, t_max, 
                                                        return_timesteps=True)
        
        trajectory = [tensor.cpu() for tensor in trajectory]
        figure = utils.plot_trajectory(trajectory[::-1], timesteps[::-1], 
                                       title=f"Backward Process, step={sb_iter}")
        
        run.log({"images/backward_trajectory": wandb.Image(figure), "sb_iter": sb_iter})
        plt.close(figure)