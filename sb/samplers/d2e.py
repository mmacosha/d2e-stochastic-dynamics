from dataclasses import dataclass

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
from sb.buffers import ReplayBuffer, LangevinReplayBuffer, DecoupledLangevinBuffer

from . import utils as sutils
from . import base_class


def infer_shape(dim):
    if int(math.sqrt(dim))**2 == dim:
        sqrt_dim = int(math.sqrt(dim))
        return (1, sqrt_dim, sqrt_dim)
    
    if int(math.sqrt(dim // 3))**2 == dim // 3:
        sqrt_dim = int(math.sqrt(dim // 3))
        return (3, sqrt_dim, sqrt_dim)
    
    raise ValueError(f"Cannot infer shape from dim={dim}.")

def complete_tensor(x, size):
    if x.size(0) >= size:
        return x

    shape = (size - x.size(0), *x.shape[1:])
    complement = torch.zeros(*shape, dtype=x.dtype, device=x.device)
    return torch.cat([x, complement], dim=0)


@dataclass
class D2ESBConfig(base_class.SBConfig):
    logging_data: str = "images"
    drift_reg_coeff: float = 0.0
    reuse_backward_trajectory: bool = True
    n_trajectories: int = 2
    num_img_to_log: int = 36
    off_policy_fraction: float = 0.25
    start_mixed_from: int = 0
    val_batch_size: int = 64
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


class D2ESB(base_class.SB):
    def __init__(self, fwd_model, bwd_model, p0, p1, config, buffer_config):
        super().__init__(fwd_model, bwd_model, p0, p1, config)
        self.buffer_config = buffer_config
        self.anneal_beta_fn = eval(self.config.anneal_beta_fn) \
            if self.config.anneal_beta_fn else None
        
        if buffer_config.buffer_type == 'simple':
            self.p1_buffer = ReplayBuffer(**config.buffer, device=self.config.device)
        elif buffer_config.buffer_type == "langevin":
            self.p1_buffer = LangevinReplayBuffer(
                p1=self.p1, 
                device=self.config.device,
                **buffer_config,
            )
        elif buffer_config.buffer_type == "decoupled_langevin":    
            self.p1_buffer = DecoupledLangevinBuffer(
                p1=self.p1, 
                device=config.device,
                **buffer_config
            )
        else:
            raise ValueError(f"Buffer is unknow: {buffer_config.buffer_type}")

    def train_backward_step(self, sb_iter, run):
        self.bwd_model.train()
        self.fwd_model.eval()

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
        self.fwd_model.train()
        self.bwd_model.eval()

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
            
            _batch_size = batch_size // n_trajectories \
                if n_trajectories > 0 else batch_size
            x0 = self.p0.sample(_batch_size).to(device)
            hdim = x0.size(1)

            # run langevin on p1
            if self.buffer_config.buffer_type == "decoupled_langevin" \
                and self.config.off_policy_fraction > 0.0:
                self.p1_buffer.run_langevin(hdim)

            # compute beta for forward sampler
            beta = 1.0
            if self.anneal_beta_fn:
                beta = self.anneal_beta_fn(step_iter, self.config.num_fwd_steps)

            # compute off-policy x0
            reuse_trajectory_loss = None
            if self.config.off_policy_fraction == 1.0:
                x1 = self.p1_buffer.sample(_batch_size, hdim).to(device)
                if self.config.reuse_bwd_trajectory:
                    density_fn = functools.partial(
                        self.p1.log_density, anneal_beta=beta
                    )
                    reuse_trajectory_loss, x0 = \
                        losses.compute_fwd_tb_log_difference_reuse_traj(
                            self.fwd_model, self.bwd_model, density_fn, 
                            x1, dt, t_max, n_steps,
                        )
                else:
                    x0 = sutils.sample_trajectory(
                        self.bwd_model, x1,"backward", dt, n_steps, t_max, 
                        only_last=True
                    )

            elif self.config.off_policy_fraction > 0:
                num_off_policy_samples = int(
                    self.config.off_policy_fraction * (_batch_size)
                )
                x1 = self.p1_buffer.sample(num_off_policy_samples, hdim).to(device)
                
                if self.config.reuse_bwd_trajectory:
                    density_fn = functools.partial(
                        self.p1.log_density, anneal_beta=beta
                    )
                    reuse_trajectory_loss, x0_off = \
                        losses.compute_fwd_tb_log_difference_reuse_traj(
                            self.fwd_model, self.bwd_model, density_fn, 
                            x1, dt, t_max, n_steps,
                        )
                else:
                    x0_off = sutils.sample_trajectory(
                        self.bwd_model, x1,"backward", dt, n_steps, t_max, 
                        only_last=True
                    )

                x0_on = x0[:_batch_size - num_off_policy_samples].to(device)
                x0 = (x0_off, x0_on)

            # compute forward loss
            if n_trajectories == 1:
                assert not isinstance(x0, tuple), "if n_trajectories < 2, " \
                                                  "bwd_trajectory should not be reused"
                density_fn = functools.partial(self.p1.log_density, anneal_beta=beta)
                loss = losses.compute_fwd_tb_loss(
                    self.fwd_model, self.bwd_model, density_fn, 
                    self.p0.log_density, x0, dt, t_max, n_steps, 
                    p1_buffer=self.p1_buffer)
            elif n_trajectories == 0:
                assert not isinstance(x0, tuple), "if n_trajectories < 2, " \
                                                  "bwd_trajectory should not be reused"
                loss = losses.compute_fwd_tlm_loss(
                    self.fwd_model, self.bwd_model, x1, dt, t_max, n_steps,
                    backward=True, matching_method="ll"
                )
            else:
                density_fn = functools.partial(self.p1.log_density, anneal_beta=beta)
                if self.config.reuse_bwd_trajectory:
                    x0_off, x0_on = x0
                    x0_off = x0_off.repeat(n_trajectories - 1, 1) 
                    x0_on = x0_on.repeat(n_trajectories, 1) 
                    x0 = torch.cat([x0_on, x0_off], dim=0)
                    
                    log = losses.compute_fwd_vargrad_loss(
                        self.fwd_model, self.bwd_model, density_fn, 
                        x0, dt, t_max, n_steps, 
                        p1_buffer=self.p1_buffer, 
                        n_trajectories=n_trajectories,
                        compute_var=False,
                    )
                    log_on  = log[ :x0_on.size(0)]
                    log_off = log[x0_on.size(0): ]

                    log_off = torch.cat([
                        log_off.reshape(n_trajectories - 1, -1), 
                        reuse_trajectory_loss.unsqueeze(0)
                    ], dim=0)
                    
                    log = torch.cat(
                        [log_on.reshape(n_trajectories, -1),  log_off], 
                        dim=1
                    )
                    
                    loss = (log  - log.mean(0, keepdim=True).detach()).pow(2).mean()
                
                else:
                    x0 = torch.cat(x0, dim=0).repeat(n_trajectories, 1) 
                    loss = losses.compute_fwd_vargrad_loss(
                        self.fwd_model, self.bwd_model, density_fn, 
                        x0, dt, t_max, n_steps, 
                        p1_buffer=self.p1_buffer, n_trajectories=n_trajectories
                    )
            
            with torch.no_grad():
                x0 = self.p0.sample(self.config.val_batch_size).to(device)
                x1 = sutils.sample_trajectory(
                    self.fwd_model, x0, "forward", dt, n_steps, t_max, only_last=True
                )
                log_reward = self.p1.reward.log_reward(x1).mean()
            
            if n_trajectories > 0:
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
        self.fwd_model.eval()
        dt = self.config.dt
        t_max = self.config.t_max
        n_steps = self.config.n_steps
        val_batch_size = self.config.val_batch_size
        device = self.config.device
        num_img_to_log = self.config.num_img_to_log

        # sample x0 and compute metrics
        x0 = self.p0.sample(val_batch_size).to(device)
        elbo, iw_1, iw_2 = metrics.compute_elbo(
            self.fwd_model, self.bwd_model, self.p1.log_density, 
            x0, dt, t_max, n_steps, n_traj=16
        )
        eubo = metrics.compute_eubo(
            self.fwd_model, self.bwd_model, self.p1.log_density, 
            x0, dt, t_max, n_steps
        )

        logging_dict = {
            "metrics/p1_elbo": elbo,
            "metrics/p1_eubo": eubo,
            "metrics/p1_iw_1": iw_1, 
            "metrics/p1_iw_2": iw_2,
            "sb_iter": sb_iter,
        }
        # log image from real latents
        real_img = self.p1.reward.generator(x0[:num_img_to_log]).cpu()
        img_dim = real_img.view(num_img_to_log, -1).shape[1]
        image_shape = (num_img_to_log, ) + infer_shape(img_dim)
        real_img_grid = make_grid(
            real_img.view(*image_shape), nrow=6, normalize=True
        )

        # log image from generated latents
        x1_pred = sutils.sample_trajectory(
            self.fwd_model, x0, 'forward', dt, n_steps, t_max, only_last=True
        )
        
        L2_between_x0_x1 = (x1_pred - x0).pow(2).sum(1).mean()
        path_kl = metrics.compute_path_kl(self.fwd_model, x0, dt, t_max, n_steps)
        
        logging_dict = logging_dict | {
            "metrics/L2^2(x0, x1)": L2_between_x0_x1,
            "metrics/Path_KL": path_kl
        }

        output = self.p1.reward(x1_pred)
        pred_img_grid = make_grid(
            output["images"][:num_img_to_log].clip(0, 1).cpu().view(image_shape), 
            nrow=6, normalize=True
        )

        random_annotated_img = utils.plot_annotated_images(
            output["images"][:num_img_to_log].clip(0, 1).cpu().view(image_shape), 
            probas_classes=(
                output["probas"][:num_img_to_log], 
                output["classes"][:num_img_to_log]
            ), 
            n_col=6, 
            figsize=(18, 18)
        )

        images, probas, classes = self.p1.reward.get_target_class_images(
            output, num_img_to_log
        )
        images = complete_tensor(images, num_img_to_log)
        probas = complete_tensor(probas, num_img_to_log)
        classes = complete_tensor(classes, num_img_to_log)
        
        target_class_fig = utils.plot_annotated_images(
            images.clip(0, 1).cpu().view(image_shape), (probas, classes),
            n_col=6, figsize=(18, 18)
        )

        reward, precision = self.p1.reward.get_reward_and_precision(outputs=output)

        logging_dict = logging_dict | {
            "metrics/precision": precision,
            "metrics/val_mean_log_reward": reward.log().mean(),
        }

        with torch.enable_grad():
            x_buffer = self.p1_buffer.sample(self.p1_buffer.size)
            x_buffer = x_buffer.to(self.config.device)
        
        buffer_output = self.p1.reward(x_buffer)
        buffer_samples_fig = utils.plot_annotated_images(
            buffer_output["images"][:num_img_to_log].clip(0, 1).cpu().view(image_shape),
            probas_classes=(
                buffer_output["probas"][:num_img_to_log], 
                buffer_output["classes"][:num_img_to_log]
            ), 
            n_col=6, figsize=(18, 18)
        )
        
        buffer_rwd, buffer_prc = self.p1.reward.get_reward_and_precision(
            outputs=buffer_output
        )

        logging_dict = logging_dict | {
            "metrics/buffer_precision": buffer_prc,
            "metrics/buffer_reward": buffer_rwd,
        }

        if sb_iter % self.config.log_img_freq == 0:
            logging_dict = logging_dict | {
                "images/buffer_samples": wandb.Image(buffer_samples_fig),
                "images/x1_sample_annotated": wandb.Image(random_annotated_img),
                "images/x1_target_class": wandb.Image(target_class_fig),
                "images/x0_sample": wandb.Image(real_img_grid),
                "images/x1-sample": wandb.Image(pred_img_grid), 
            }
    
        run.log(logging_dict)
        plt.close("all")

    @torch.no_grad()
    def log_backward_step(self, sb_iter, run):
        return None
