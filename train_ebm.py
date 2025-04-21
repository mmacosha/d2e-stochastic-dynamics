"""
    This file is almost exact copy of emb.ipynb, intended for running 
    the training via cli. It is supposed to be used for large experiments.
"""
import math
import hydra
import wandb
import torch
import random
import numpy as np
from pathlib import Path

from tqdm.auto import trange
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from samplers.utils import sample_trajectory
from utils import plot_trajectory

from models.model import (
    SimpleNet, Energy, 
    ReferenceProcess2, 
)

from samplers import losses
from buffers import ReplayBuffer
from data.datasets_legacy import GaussMix
from ema import EMA


def set_seed(seed, device):
    """Fixes random seed for reproducibility."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def langevin_dynamics(energy, init_step_size=0.0001, n_steps=3001, 
                      num_samples=512, log_interval=500, device="cpu"):
    x = torch.randn(num_samples, 2).to(device)  # Ensure tensor is on the correct device
    trajectory = [x]
    timesteps = [0]
    for i in range(1, n_steps + 1):
        step_size = init_step_size * (0.01 ** (i / n_steps)) 
        
        # compute gradient
        _x = x.clone().detach().requires_grad_()
        grad = torch.autograd.grad(energy(_x).sum(), _x)[0]
        
        # make a langevin step
        x = x - 0.5 * step_size * grad 
        x = x + torch.randn_like(x) * math.sqrt(step_size)

        if i % log_interval == 0:
            trajectory.append(x)
            timesteps.append(i)

    return trajectory, timesteps


@torch.no_grad()
def log_trajectory(model, x, direction, config, it, limits=(-5, 5)):
    x = x.to(config.device)  # Ensure input tensor is on the correct device
    dt = config.dt
    t_max = config.t_max
    n_steps = config.n_steps
    trajectory, timesteps = sample_trajectory(model, x, direction, dt, n_steps, 
                                              t_max, return_timesteps=True)
    figure = plot_trajectory(trajectory, timesteps, 
                             title=direction, limits=limits)
    wandb.log({f"{direction}_trajectory": wandb.Image(figure)}, step=it)
    plt.close(figure)


@torch.no_grad()
def log_ebm(energy, fwd_model, dataset_sampler, config, it, 
            limits=(-5, 5), n_points=200):
    dt = config.dt
    t_max = config.t_max
    n_steps = config.n_steps
    noise_std = config.noise_std
    
    figure, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Scatter plot of samples from x_1
    with torch.no_grad():
        x_0_log = dataset_sampler.sample(config.batch_size)
        x_0_log_noisy = x_0_log + torch.randn_like(x_0_log) * noise_std

        x_1_log = sample_trajectory(fwd_model, x_0_log_noisy, "forward", dt, n_steps, 
                                    t_max, only_last=True)
        
    axes[0].scatter(x_0_log[:, 0], x_0_log[:, 1], c='b', alpha=0.5, label='x_0')
    axes[0].scatter(x_1_log[:, 0], x_1_log[:, 1], c='r', alpha=0.5, label='x_1')
    axes[0].legend()
    axes[0].set_title('Samples from x_1')
    axes[0].set_xlim(*limits)
    axes[0].set_ylim(*limits)

    # Energy function plot
    x = torch.linspace(*limits, n_points).to(config.device)
    y = torch.linspace(*limits, n_points).to(config.device)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    grid_points = torch.stack([X, Y], axis=-1).reshape(-1, 2).float().to(config.device)

    log_density = - energy(grid_points)
    log_density = log_density - log_density.max() 

    Z = log_density.cpu().numpy().reshape(200, 200)  # Move to CPU for plotting

    contour = axes[1].contour(X.cpu(), Y.cpu(), Z, levels=7, colors='k')
    axes[1].clabel(contour, inline=True, fontsize=6)
    axes[1].contourf(X.cpu(), Y.cpu(), Z, levels=10, cmap='viridis', alpha=0.5)
    axes[1].set_title('Energy Function')
    axes[1].set_xlim(*limits)
    axes[1].set_ylim(*limits)
    figure.colorbar(axes[1].contourf(X.cpu(), Y.cpu(), Z, levels=10, cmap='viridis', 
                    alpha=0.5), ax=axes[1], label='Energy Value')

    wandb.log({f"- energy": wandb.Image(figure)}, step=it)
    plt.close(figure)


class FixedSizeDataset:
    def __init__(self, dataset):
        self.dataset = dataset
        self.size = dataset.size(0)

    def sample(self, batch_size):
        idxs = torch.randint(0, self.size, (batch_size,))
        return self.dataset[idxs]


@torch.no_grad()
def save_checkpoint(fwd_model, bwd_model, energy, it):
    save_checkpoint = {
        "fwd_model": fwd_model.state_dict(),
        "bwd_model": bwd_model.state_dict(),
        "energy": energy.state_dict(),
    }
    checkpoint_dir = Path(wandb.run.dir) / "files" / "checkpoints"
    checkpoint_name = f"checkpoint_{it}.pt"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(save_checkpoint, checkpoint_dir / checkpoint_name)


def train_sb_ebm(fwd_model, bwd_model, energy, energy_ema, ref_process,
                 fwd_optim, bwd_optim, energy_optim, p1_buffer,
                 fwd_scheduler, bwd_scheduler, energy_scheduler,
                 dataset_sampler, train_config):
    dt = train_config.dt
    t_max = train_config.t_max
    n_steps = train_config.n_steps
    n_trajectories = train_config.n_trajectories
    device = train_config.device
    noise_std = train_config.noise_std
    batch_size = train_config.batch_size

    x_train = dataset_sampler.sample(batch_size)

    for it in trange(train_config.num_iters, desc="EBM training"):        
        # TRAIN BACKWARD PROCESS
        for _ in range(train_config.num_bwd_iters):
            bwd_optim.zero_grad(set_to_none=True)

            if train_config.resample_x_0:
                x_train = dataset_sampler.sample(batch_size)
            x_0 = x_train + torch.randn_like(x_train) * noise_std

            _fwd_model = ref_process if it == 0 else fwd_model
            loss = losses.compute_bwd_tlm_loss(
                _fwd_model, bwd_model, x_0, dt, t_max, n_steps
            )

            assert not torch.isnan(loss).any(), "backward loss is NaN"
            bwd_optim.step()
            bwd_scheduler.step()

        # if we start using bigger num_bwd_iters, we should 
        # conside changing the way we log the backward loss.
        wandb.log({"backward_loss": loss.item()}, step=it)
        wandb.log({"backward_lr": bwd_optim.param_groups[0]["lr"]}, step=it)

        # LOG BACKWARD TRAJECTORY
        if it % train_config.log_interval == 0:
            if p1_buffer.is_empty():
                x_0_log = dataset_sampler.sample(train_config.batch_size).to(device)
                x_0_log = x_0_log + torch.randn_like(x_0_log) * train_config.noise_std
                x_1 = sample_trajectory(fwd_model, x_0_log, "forward", dt, 
                                        n_steps, t_max, only_last=True)
            else:
                x_1 = p1_buffer.sample(train_config.batch_size).to(device)

            log_trajectory(bwd_model, x_1, "backward", train_config, it)

        # TRAIN FORWARD PROCESS
        for _ in range(train_config.num_fwd_iters):
            fwd_optim.zero_grad(set_to_none=True)

            if train_config.resample_x_0:
                x_train = dataset_sampler.sample(batch_size)
            x_0 = x_train[:batch_size // n_trajectories].repeat(n_trajectories, 1)
            x_0 = x_0 + torch.randn_like(x_0) * noise_std
            
            loss = losses.compute_fwd_vargrad_loss(
                fwd_model, bwd_model, lambda x: - energy(x),
                x_0, dt, t_max, n_steps, 
                p1_buffer=p1_buffer,
                n_trajectories=n_trajectories,
                clip_range=(-10000, 10000)
            )
            assert not torch.isnan(loss).any(), "forward loss is NaN"
            loss.backward()
            
            fwd_optim.step()
            fwd_scheduler.step()

        # if we start using bigger num_fwd_iters, we should conside
        # changing the way we log the forward loss
        wandb.log({"forward_loss": loss.item()}, step=it)
        wandb.log({"forward_lr": fwd_optim.param_groups[0]["lr"]}, step=it)

        # LOG FORWARD TRAJECTORY
        if it % train_config.log_interval == 0:
            x_0_log = dataset_sampler.sample(train_config.batch_size).to(device)
            x_0_log = x_0_log + torch.randn_like(x_0_log) * train_config.noise_std
            log_trajectory(fwd_model, x_0_log, "forward", train_config, it)

        # TRAIN ENERGY FUNCTION
        for _ in range(train_config.num_energy_iters):
            if train_config.resample_x_0:
                x_train = dataset_sampler.sample(batch_size)
            
            if train_config.use_samples_from_buffer:
                x_1 = p1_buffer.sample(batch_size)
            else:
                x_train_noisy = x_train + torch.randn_like(x_train) * noise_std
                x_1 = sample_trajectory(fwd_model, x_train_noisy, "forward", dt, 
                                        n_steps, t_max, only_last=True)
            
            energy_optim.zero_grad(set_to_none=True)
            loss = losses.ebm_loss(energy, x_train, x_1, 
                                   alpha=train_config.ebm_loss_alpha,
                                   reg_type=train_config.ebm_reg_type)
            
            assert not torch.isnan(loss).any(), "energy loss is NaN"
            loss.backward()

            energy_optim.step()
            energy_scheduler.step()
            energy_ema.update()
        
        # if we start using bigger num_energy_iters, we should conside
        # changing the way we log the energy loss
        wandb.log({"energy_loss": loss.item()}, step=it)
        wandb.log({"energy_lr": energy_optim.param_groups[0]["lr"]}, step=it)

        # LOG ENERGY FUNCTION
        if it % train_config.log_interval == 0:
            energy_ema.apply_shadow()
            log_ebm(energy, fwd_model, dataset_sampler, train_config, it)
            energy_ema.restore()

        # LOG LANDEVIN TRAJECTORY
        if it % train_config.langevin_log_interval == 0 and it > 0:
            trajectory, timesteps = langevin_dynamics(
                energy, ld_step_size=0.0001, n_steps=3001, 
                log_interval=500, device=device
            )
            
            trajectory.append(
                dataset_sampler.sample(train_config.batch_size).to(device)
            )
            timesteps.append("real samples")

            figure = plot_trajectory(trajectory, timesteps, 
                                     title="langevin", limits=(-5, 5))
            wandb.log({"langevin_trajectory": wandb.Image(figure)}, step=it)
            plt.close(figure)

        if (it > 0 and it % train_config.save_interval == 0) \
            or it + 1 == train_config.num_iters:
            save_checkpoint(fwd_model, bwd_model, energy, it)


@hydra.main(version_base=None, config_path="configs", config_name="ebm")
def main(config):
    # Set device
    device = torch.device(config.train.device if torch.cuda.is_available() else "cpu")

    # Fix seed
    set_seed(config.seed, device)

    # setup models
    ref_process = ReferenceProcess2(alpha=config.train.reference_alpha)
    
    energy = Energy(**config.energy_model_params).to(device)
    fwd_model = SimpleNet(**config.fwd_model_params).to(device)
    bwd_model = SimpleNet(**config.bwd_model_params).to(device)
    
    energy_ema = EMA(energy, decay=config.train.enegy_ema_decay)

    energy_optim = torch.optim.Adam(energy.parameters(), lr=config.train.energy_lr)
    fwd_optim = torch.optim.Adam(fwd_model.parameters(), lr=config.train.fwd_lr)
    bwd_optim = torch.optim.Adam(bwd_model.parameters(), lr=config.train.bwd_lr)

    milestones = config.train.lr_schedule_milestones
    lr_schedule_gamma = config.train.lr_schedule_gamma
    fwd_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        fwd_optim, 
        milestones=[ms * config.train.num_fwd_iters for ms in milestones], 
        gamma=lr_schedule_gamma
    )
    bwd_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        bwd_optim, 
        milestones=[ms * config.train.num_bwd_iters for ms in milestones], 
        gamma=lr_schedule_gamma
    )
    energy_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        energy_optim, 
        milestones=[ms * config.train.num_energy_iters for ms in milestones], 
        gamma=lr_schedule_gamma
    )
     
    p1_buffer = ReplayBuffer(config.train.p1_buffer_size,
                             update_fraction=1.0)
    
    # setup data
    means = torch.tensor([
        [-2, 2], [2, 2], [-3, 0], [3, 0],
        [0, -3], [0, 3], [-2, -2], [2, -2]
    ]).float() * 0.99
    sigmas = torch.ones_like(means) * 0.1
    gm_sampler = GaussMix(means, sigmas)
    dataset_sampler = FixedSizeDataset(gm_sampler.sample(1024))

    wandb.init(project=config.project, name=config.name, mode=config.mode, 
               config=OmegaConf.to_container(config))
    
    train_sb_ebm(fwd_model, bwd_model, energy, energy_ema, ref_process,
                 fwd_optim, bwd_optim, energy_optim, p1_buffer,
                 fwd_scheduler, bwd_scheduler, energy_scheduler,
                 dataset_sampler, config.train)

if __name__ == "__main__":
    main()
