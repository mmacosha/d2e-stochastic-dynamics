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
from sb.buffers import (
    ReplayBuffer, 
    LangevinReplayBuffer,
    DecoupledLangevinBuffer
)

from . import utils as sutils
from . import d2e

import seaborn as sns


class D2ESB_IMG(d2e.D2ESB):
    @torch.no_grad()
    def log_forward_step(self, sb_iter, run):
        self.fwd_model.eval()
        
        dt = self.config.dt
        var = self.config.var
        t_max = self.config.t_max
        alpha = self.config.alpha
        
        n_steps = self.config.n_steps
        device = self.config.device

        # Compute metrics
        x0 = self.p0.sample(self.config.metric_batch_size).to(device)
        elbo, iw_1, iw_2 = metrics.compute_elbo(
            self.fwd_model, self.bwd_model, 
            self.p1.log_density, 
            x0, dt, t_max, n_steps, 
            n_traj=16
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

        # Log images
        num_img_to_log = self.config.num_img_to_log
        real_img = self.p1.reward.generator(x0[:num_img_to_log]).cpu()
        img_dim = real_img.view(num_img_to_log, -1).shape[1]
        image_shape = (num_img_to_log, ) + infer_shape(img_dim)
        real_img_grid = make_grid(
            real_img.view(*image_shape), nrow=6, normalize=True
        )
        
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
        x1_true = self.p1.sample(self.config.metric_batch_size).to(self.config.device)
        
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
