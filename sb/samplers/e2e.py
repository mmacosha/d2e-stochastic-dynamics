from dataclasses import dataclass
import functools

import math
import torch
import wandb

from tqdm.auto import trange
from matplotlib import pyplot as plt

import sb.utils as sb_utils
import sb.metrics as metrics
import sb.losses as losses
import sb.buffers as buffers

from . import utils as sampling_utils
from . import base_class


@dataclass
class E2ESBConfig(base_class.SBConfig):
    logging_data: str = "images"
    drift_reg_coeff: float = 0.0
    n_trajectories: int = 2
    num_img_to_log: int = 36
    off_policy_fraction: float = 0.25
    start_mixed_from: int = 0
    watch_models: bool = False
    anneal_beta_fn: str = None
    reuse_bwd_trajectory: bool = False

    def __post__init__(self):
        super().__post__init__()
        assert self.logging_data in {"images", "2d"}, \
            f"Unknown logging data: {self.logging_data}. " \
            "Available options: images, 2d."
        assert self.matching_method == 'll', \
            "D2E training does not support other matching methods"
        assert self.start_mixed_from >= 0
        if self.n_trajectories == 0:
            assert self.off_policy_fraction == 1.0, "`n_trajectories=0` means the " \
                "loss is calculated using TLM on langevin samples, meaning it is 100% " \
                "off-policy"


class E2ESB(base_class.SB):
    def __init__(self, fwd_model, bwd_model, p0, p1, config, buffer_config):
        super().__init__(fwd_model, bwd_model, p0, p1, config)
        self.buffer_config = buffer_config
        self.anneal_beta_fn = eval(self.config.anneal_beta_fn) \
            if self.config.anneal_beta_fn else None
        
        if buffer_config.buffer_type == 'simple':
            self.p0_buffer = buffers.ReplayBuffer(
                **buffer_config, device=self.config.device
            )
            self.p1_buffer = buffers.ReplayBuffer(
                **buffer_config, device=self.config.device
            )

        elif buffer_config.buffer_type == "langevin":
            self.p0_buffer = buffers.LangevinReplayBuffer(
                p1=self.p0, device=self.config.device, **buffer_config
            )
            self.p1_buffer = buffers.LangevinReplayBuffer(
                p1=self.p1, device=self.config.device, **buffer_config
            )

        elif buffer_config.buffer_type == "decoupled_langevin":    
            self.p0_buffer = buffers.DecoupledLangevinBuffer(
                p1=self.p0, device=config.device, **buffer_config
            )
            self.p1_buffer = buffers.DecoupledLangevinBuffer(
                p1=self.p1, device=config.device, **buffer_config
            )

        else:
            raise ValueError(f"Buffer is unknow: {buffer_config.buffer_type}")

    def train_backward_step(self, sb_iter, run):
        self.bwd_model.train()
        self.fwd_model.eval()

        dt = self.config.dt
        t_max = self.config.t_max
        alpha = self.config.alpha
        n_steps = self.config.n_steps

        device = self.config.device
        batch_size = self.config.batch_size
        n_trajectories = self.config.n_trajectories

        self.fwd_model_ema.apply()
        for step_iter in trange(self.config.num_fwd_steps, 
                                leave=False, desc=f'It {sb_iter} | Forward'):
            self.bwd_optim.zero_grad(set_to_none=True)
            
            _batch_size = batch_size // n_trajectories \
                if n_trajectories > 0 else batch_size
            if sb_iter > 0:
                x1 = self.p1_buffer.sample(_batch_size).to(device)
            else:
                x1 = torch.randn(_batch_size, 2).to(device)

            # run langevin on p0
            if self.buffer_config.buffer_type == "decoupled_langevin" \
                and self.config.off_policy_fraction > 0.0:
                self.p0_buffer.run_langevin(2)

            # compute beta for forward sampler
            beta = 1.0
            if self.anneal_beta_fn:
                beta = self.anneal_beta_fn(step_iter, self.config.num_bwd_steps)

            # compute off-policy x0
            reuse_trajectory_loss = None
            if self.config.off_policy_fraction == 1.0:
                x0 = self.p0_buffer.sample(_batch_size, 2).to(device)
                x1 = sampling_utils.sample_trajectory(
                    self.fwd_model, x0, dt, t_max, n_steps, alpha, var=2.0,
                    direction='fwd',
                    only_last=True,
                    method=self.config.matching_method,
                )

            elif self.config.off_policy_fraction > 0:
                num_off_policy_samples = int(
                    self.config.off_policy_fraction * (_batch_size)
                )
                x0 = self.p0_buffer.sample(num_off_policy_samples, 2).to(device)
                x1_off = sampling_utils.sample_trajectory(
                    self.fwd_model, x0, dt, t_max, n_steps, alpha, var=2.0,
                    direction='fwd',
                    only_last=True,
                    method=self.config.matching_method,
                )

                x1_on = x1[:_batch_size - num_off_policy_samples].to(device)
                x0 = torch.cat([x1_off, x1_on], dim=0)   

            # compute forward loss
            if n_trajectories < 2:
                raise NotImplementedError('Use VarGrad for backward model')
            else:
                density_fn = functools.partial(self.p0.log_density, anneal_beta=beta)
                
                loss = losses.compute_bwd_vargrad_loss(
                    self.fwd_model, self.bwd_model, 
                    density_fn, 
                    x1, dt, t_max, n_steps, 
                    p0_buffer=self.p0_buffer, 
                    n_trajectories=n_trajectories
                )
            
            with torch.no_grad():
                x1 = self.p1.sample(self.config.val_batch_size).to(device)
                
                x0 = sampling_utils.sample_trajectory(
                    self.bwd_model, x1, dt, t_max, n_steps, alpha, var=2.0,
                    direction="bwd",
                    only_last=True,
                    method=self.config.matching_method,
                )
                mean_log_reward = self.p0.get_mean_log_reward(x0).mean()
            
            if n_trajectories > 0:
                loss.backward()
            
            self.bwd_optim.step()
            self.bwd_model_ema.update()

            run.log({
                "train/backward_loss": loss / n_steps,
                "metrics/p0_mean_log_reward": mean_log_reward,
                "bwd_step": sb_iter * self.config.num_bwd_steps + step_iter
            })
        
        self.fwd_model_ema.restore()

    def train_forward_step(self, sb_iter, run):
        self.fwd_model.train()
        self.bwd_model.eval()

        dt = self.config.dt
        t_max = self.config.t_max
        alpha = self.config.alpha
        n_steps = self.config.n_steps

        device = self.config.device
        batch_size = self.config.batch_size
        n_trajectories = self.config.n_trajectories

        self.bwd_model_ema.apply()
        for step_iter in trange(self.config.num_fwd_steps, 
                                leave=False, desc=f'It {sb_iter} | Forward'):
            self.fwd_optim.zero_grad(set_to_none=True)
            
            _batch_size = batch_size // n_trajectories \
                if n_trajectories > 0 else batch_size
            if sb_iter > 0:
                x0 = self.p0_buffer.sample(_batch_size).to(device)
            else:
                x0 = torch.randn(_batch_size, 2).to(device)

            # run langevin on p1
            if self.buffer_config.buffer_type == "decoupled_langevin" \
                and self.config.off_policy_fraction > 0.0:
                self.p1_buffer.run_langevin(2)

            # compute beta for forward sampler
            beta = 1.0
            if self.anneal_beta_fn:
                beta = self.anneal_beta_fn(step_iter, self.config.num_fwd_steps)

            # compute off-policy x0
            reuse_trajectory_loss = None
            if self.config.off_policy_fraction == 1.0:
                x1 = self.p1_buffer.sample(_batch_size, 2).to(device)
                x0 = sampling_utils.sample_trajectory(
                    self.bwd_model, x1, dt, t_max, n_steps, alpha, var=2.0,
                    direction='bwd',
                    only_last=True,
                    method=self.config.matching_method,
                )

            elif self.config.off_policy_fraction > 0: # and sb_iter > 0:
                num_off_policy_samples = int(
                    self.config.off_policy_fraction * (_batch_size)
                )
                x1 = self.p1_buffer.sample(num_off_policy_samples, 2).to(device)
                x0_off = sampling_utils.sample_trajectory(
                    self.bwd_model, x1, dt, t_max, n_steps, alpha, var=2.0,
                    direction='bwd',
                    only_last=True,
                    method=self.config.matching_method,
                )

                x0_on = x0[:_batch_size - num_off_policy_samples].to(device)
                x0 = torch.cat([x0_off, x0_on], dim=0)

            # compute forward loss
            if n_trajectories < 2:
                raise NotImplementedError('Use VarGrad for forward model')
            
            else:
                density_fn = functools.partial(self.p1.log_density, anneal_beta=beta)
                loss = losses.compute_fwd_vargrad_loss(
                    self.fwd_model, self.bwd_model, 
                    density_fn, 
                    x0, dt, t_max, n_steps, 
                    p1_buffer=self.p1_buffer, 
                    n_trajectories=n_trajectories
                )
            
            with torch.no_grad():
                x0 = self.p0.sample(self.config.val_batch_size).to(device)
                
                x1 = sampling_utils.sample_trajectory(
                    self.fwd_model,
                    x0, dt, t_max, n_steps, alpha, var=2.0,
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
                "metrics/p1_mean_log_reward": mean_log_reward,
                "fwd_step": sb_iter * self.config.num_fwd_steps + step_iter
            })
        
        self.bwd_model_ema.restore()

    @torch.no_grad()
    def log_forward_step(self, sb_iter, run):
        dt = self.config.dt
        t_max = self.config.t_max
        n_steps = self.config.n_steps

        device = self.config.device

        # Compute metrics
        if sb_iter > 0:
            x0 = self.p0_buffer.sample(self.config.metric_batch_size).to(device)
        else:
            x0 = torch.randn(self.config.metric_batch_size, 2).to(device)
        
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
        x1_pred = sampling_utils.sample_trajectory(
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
                "metrics/p1_W2": w2
            }
        
        except NotImplementedError:
            pass

        # Log trajectories
        x0 = x0[:self.config.val_batch_size]
        trajectory, timesteps = sampling_utils.sample_trajectory(
            self.fwd_model, 
            x0, dt, t_max, n_steps, self.config.alpha, self.config.var,
            direction="fwd", 
            only_last=False,
            return_timesteps=True, 
            method=self.config.matching_method
        )
        trajectory_fig = sb_utils.plot_trajectory(
            [tensor.cpu() for tensor in trajectory], timesteps, 
            title=f"Forward Process, step={sb_iter}",
            limits=tuple(self.config.plot_limits)
        )
        logging_dict = logging_dict | {
            "images/forward_trajectory": wandb.Image(trajectory_fig)
        }
        
        run.log(logging_dict)
        plt.close("all")

    @torch.no_grad()
    def log_backward_step(self, sb_iter, run):
        dt = self.config.dt
        t_max = self.config.t_max
        n_steps = self.config.n_steps

        device = self.config.device

        # Compute metrics
        if sb_iter > 0:
            x1 = self.p1_buffer.sample(self.config.metric_batch_size).to(device)
        else:
            x1 = torch.randn(self.config.metric_batch_size, 2).to(device)

        x0_pred = sampling_utils.sample_trajectory(
            self.bwd_model, x1, dt, t_max, n_steps, self.config.alpha, self.config.var,
            direction="bwd",
            only_last=True,
            method=self.config.matching_method,
        )
        x0_x1_transport_cost = (x0_pred - x1).pow(2).sum(1).mean()
        logging_dict = {
            "metrics/x0_x1_transport_cost": x0_x1_transport_cost,
            "sb_iter": sb_iter,
        }
        try:
            x0_true = self.p0.sample(self.config.val_batch_size).to(device)
            w2 = metrics.compute_w2_distance(x0_true, x0_pred)
            logging_dict = logging_dict | {
                "metrics/p0_W2": w2
            }
        
        except NotImplementedError:
            pass

        # Log trajectories
        x1 = x1[:self.config.val_batch_size]
        trajectory, timesteps = sampling_utils.sample_trajectory(
            self.bwd_model, x1, dt, t_max, n_steps, self.config.alpha, self.config.var,
            direction="bwd", 
            only_last=False,
            return_timesteps=True, 
            method=self.config.matching_method
        )
        trajectory_fig = sb_utils.plot_trajectory(
            [tensor.cpu() for tensor in trajectory], timesteps, 
            title=f"Backward Process, step={sb_iter}",
            limits=tuple(self.config.plot_limits)
        )
        logging_dict = logging_dict | {
            "images/backward_trajectory": wandb.Image(trajectory_fig)
        }
        
        run.log(logging_dict)
        plt.close("all")

    @torch.no_grad()
    def log_final_metric(self, run):
        dt = self.config.dt
        t_max = self.config.t_max
        n_steps = self.config.n_steps

        x0 = self.p0_buffer.sample(
            self.config.metric_batch_size
        ).float().to(self.config.device)
        x1 = self.p1_buffer.sample(
            self.config.metric_batch_size
        ).float().to(self.config.device)
        
        x1_pred = sampling_utils.sample_trajectory(
            self.fwd_model, x0, dt, t_max, n_steps, self.config.alpha, self.config.var, 
            direction="fwd",
            only_last=True,
            return_timesteps=False, 
            method=self.config.matching_method
        )
        x0_pred = sampling_utils.sample_trajectory(
            self.bwd_model, x1, dt, t_max, n_steps, self.config.alpha, self.config.var, 
            direction="bwd",
            only_last=True,
            return_timesteps=False, 
            method=self.config.matching_method
        )

        x0_true = self.p0.sample(
            self.config.val_batch_size
        ).float().to(self.config.device)
        x1_true = self.p1.sample(
            self.config.val_batch_size
        ).float().to(self.config.device)

        p0_w2 = metrics.compute_w2_distance(
            x0_true,  x0_pred.to(self.config.device)
        )
        p1_w2 = metrics.compute_w2_distance(
            x1_true,  x1_pred.to(self.config.device)
        )
        final_metrics = {
            "p0_W2": p0_w2,
            "p1_W2": p1_w2,
        }
        wandb.log(final_metrics)

        for k, v in final_metrics.items():
            wandb.run.summary[k] = v
