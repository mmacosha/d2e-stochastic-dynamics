from dataclasses import dataclass

import math
import torch
import wandb
from tqdm.auto import trange
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

import sb.utils as utils
import sb.metrics as metrics
from sb.buffers import ReplayBuffer, LangevinReplayBuffer, DecoupledLangevinBuffer

from . import losses
from . import utils as sutils
from . import base_class


def _infer_shape(dim):
    if int(math.sqrt(dim))**2 == dim:
        sqrt_dim = int(math.sqrt(dim))
        return (1, sqrt_dim, sqrt_dim)
    
    if int(math.sqrt(dim // 3))**2 == dim // 3:
        sqrt_dim = int(math.sqrt(dim // 3))
        return (3, sqrt_dim, sqrt_dim)
    
    raise ValueError(f"Cannot infer shape from dim={dim}.")


@dataclass
class D2ESBConfig(base_class.SBConfig):
    logging_data: str = "images"
    drift_reg_coeff: float = 0.0
    reuse_backward_trajectory: bool = True
    n_trajectories: int = 2
    off_policy_fraction: float = 0.25
    start_mixed_from: int = 0
    val_batch_size: int = 64
    watch_models: bool = False

    # Buffer parameters
    buffer_type: str = 'simple'
    langevin_freq: int = 250
    ema_lambda: float = 0.0
    buffer_size: int = 512
    init_step_size: float = 0.1
    num_langevin_steps: int = 20
    buffer_sampler: str = 'legacy'
    noise_start_ratio: float = 0.5
    reward_proportional_sample: bool = False
    anneal_value: float = 0.1
    hmc_freq: int = 4

    def __post__init__(self):
        super().__post__init__()
        assert self.logging_data in {"images", "2d"}, \
            f"Unknown logging data: {self.logging_data}. " \
            "Available options: images, 2d."
        assert self.matching_method == 'll', \
            "D2E training does not support other matching methods"
        assert self.start_mixed_from >= 0


class D2ESB(base_class.SB):
    def __init__(self, fwd_model, bwd_model, p0, p1, config):
        super().__init__(fwd_model, bwd_model, p0, p1, config)
        
        if config.buffer_type == 'simple':
            self.p1_buffer = ReplayBuffer(config.buffer_size, config.device)
        elif config.buffer_type == "langevin":
            self.p1_buffer = LangevinReplayBuffer(
                size=config.buffer_size, 
                p1=self.p1, 
                init_step_size=config.init_step_size, 
                num_steps=config.num_langevin_steps,
                sampler=config.buffer_sampler,
                noise_start_ration=config.noise_start_ratio,
                anneal_value=config.anneal_value,
                hmc_freq=config.hmc_freq,
                device=config.device,
            )
        elif config.buffer_type == "decoupled_langevin":
            self.p1_buffer = DecoupledLangevinBuffer(
                langevin_freq=config.langevin_freq,
                size=config.buffer_size, 
                p1=self.p1, 
                init_step_size=config.init_step_size, 
                num_steps=config.num_langevin_steps,
                sampler=config.buffer_sampler,
                noise_start_ration=config.noise_start_ratio,
                anneal_value=config.anneal_value,
                hmc_freq=config.hmc_freq,
                device=config.device,
                reward_proportional_sample=config.reward_proportional_sample
            )
        else:
            raise ValueError(f"Buffer is unknow: {config.buffer_type}")

    def train_backward_step(self, sb_iter, run):
        dt = self.config.dt
        t_max = self.config.t_max
        n_steps = self.config.n_steps

        self.fwd_model_ema.apply()
        for step_iter in trange(self.config.num_bwd_steps, leave=False, 
                                desc=f'It {sb_iter} | Backward'):
            self.bwd_optim.zero_grad(set_to_none=True)

            x0 = self.p0.sample(self.config.batch_size).to(self.config.device)
            fwd_model = self.reference_process if sb_iter == 0 else self.fwd_model
            loss = losses.compute_bwd_tlm_loss(
                fwd_model, self.bwd_model, x0, dt, t_max, n_steps
            )
                
            self.bwd_optim.step()
            self.bwd_model_ema.update()
            run.log({
                "train/backward_loss": loss / n_steps,
                "bwd_step": sb_iter * self.config.num_bwd_steps + step_iter
            })
        
        self.fwd_model_ema.restore()

    def train_forward_step(self, sb_iter, run):
        dt = self.config.dt
        t_max = self.config.t_max
        n_steps = self.config.n_steps

        device = self.config.device
        batch_size = self.config.batch_size
        n_trajectories = self.config.n_trajectories

        self.bwd_model_ema.apply()
        for step_iter in trange(self.config.num_fwd_steps, 
                                leave=False, desc=f'It {sb_iter} | Forward'):
            self.fwd_optim.zero_grad(set_to_none=True)
            
            _batch_size = batch_size // n_trajectories
            x0 = self.p0.sample(_batch_size).to(device)
            hdim = x0.size(1)

            if sb_iter > 30:
                self.config.off_policy_fraction = 0.0

            if self.config.buffer_type == "decoupled_langevin" \
                and self.config.off_policy_fraction > 0.0:
                self.p1_buffer.run_langevin(hdim)

            if self.config.off_policy_fraction == 1.0:
                x1 = self.p1_buffer.sample(_batch_size, hdim).to(device)
                x0 = sutils.sample_trajectory(
                    self.bwd_model, x1,"backward", dt, n_steps, t_max, only_last=True
                )

            elif self.config.off_policy_fraction > 0:
                num_off_policy_samples = int(
                    self.config.off_policy_fraction * (_batch_size)
                )
                x1 = self.p1_buffer.sample(num_off_policy_samples, hdim).to(device)
                x0_off = sutils.sample_trajectory(
                    self.bwd_model, x1,"backward", dt, n_steps, t_max, only_last=True
                )
                x0_on = x0[:_batch_size - num_off_policy_samples].to(device)
                x0 = torch.cat([x0_off, x0_on], dim=0)

            if n_trajectories == 1:
                loss = losses.compute_fwd_tb_loss(
                    self.fwd_model, self.bwd_model, self.p1.log_density, 
                    self.p0.log_density, x0, dt, t_max, n_steps, 
                    p1_buffer=self.p1_buffer
                )
            else:
                x0 = x0.repeat(n_trajectories, 1)
                loss = losses.compute_fwd_vargrad_loss(self.fwd_model, self.bwd_model, 
                                                    self.p1.log_density, x0, dt, t_max, 
                                                    n_steps, p1_buffer=self.p1_buffer,
                                                    n_trajectories=n_trajectories)
            with torch.no_grad():
                x0 = self.p0.sample(self.config.val_batch_size).to(device)
                x1 = sutils.sample_trajectory(
                    self.fwd_model, x0, "forward", dt, n_steps, t_max, only_last=True
                )
                log_reward = self.p1.reward.log_reward(x1).mean()

            loss.backward()
            self.fwd_optim.step()
            self.fwd_model_ema.update()

            run.log({
                "train/forward_loss": loss / n_steps,
                "metrics/mean_log_reward": log_reward,
                "fwd_step": sb_iter * self.config.num_fwd_steps + step_iter
            })
        
        torch.cuda.empty_cache()
        self.bwd_model_ema.restore()

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
        if self.config.logging_data == "images":
            # log image from real latents
            real_img = self.p1.reward.generator(x0[:36]).cpu()
            img_dim = real_img.view(36, -1).shape[1]
            image_shape = (36, ) + _infer_shape(img_dim)
            real_img_grid = make_grid(
                real_img.view(*image_shape), nrow=6, normalize=True
            )
            
            # log image from generated latents
            x1_pred = sutils.sample_trajectory(self.fwd_model, x0, 'forward', 
                                               dt, n_steps, t_max, only_last=True)
            
            output = self.p1.reward(x1_pred)
            pred_img_grid = make_grid(
                output["images"][:36].clip(0, 1).cpu().view(image_shape), 
                nrow=6, normalize=True
            )
            fig1 = utils.plot_annotated_images(
                output["images"][:36].clip(0, 1).cpu().view(image_shape), 
                probas_classes=(output["probas"][:36], output["classes"][:36]), 
                n_col=6, figsize=(18, 18)
            )
            precision = self.p1.reward.get_precision(output["classes"])

            logging_dict = logging_dict | {
                "images/x1_sample_annotated": wandb.Image(fig1),
                "images/x0_sample": wandb.Image(real_img_grid),
                "images/x1-sample": wandb.Image(pred_img_grid), 
                "metrics/precision": precision,
            }

            with torch.enable_grad():
                x_buffer = self.p1_buffer.sample(self.config.buffer_size)
                x_buffer = x_buffer.to(self.config.device)
            
            buffer_output = self.p1.reward(x_buffer)
            fig2 = utils.plot_annotated_images(
                buffer_output["images"][:36].clip(0, 1).cpu().view(image_shape),
                (buffer_output["probas"][:36], buffer_output["classes"][:36]), 
                n_col=6, figsize=(18, 18)
            )
            
            buffer_rwd, buffer_prc = self.p1.reward.get_reward_and_precision(
                outputs=buffer_output
            )

            logging_dict = logging_dict | {
                "images/buffer_samples": wandb.Image(fig2),
                "metrics/buffer_precision": buffer_prc,
                "metrics/buffer_reward": buffer_rwd,
            }
        
        elif self.config.logging_data == "2d":
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
        return None
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
