
import wandb

wandb.init(project="ebm-mnist", config={
    "dataset": "MNIST",
    "model": "Energy-Based Model",
    "epochs": 10  # Adjust accordingly
})

import numpy as np

import torch

import torchvision
from torchvision.transforms import v2

import numpy as np
import random

from IPython.display import clear_output
import matplotlib.pyplot as plt


from models.model import MNISTEnergy, MNISTSampler
from samplers import losses
from buffers import ReplayBuffer
from data.infinite_loader import InfiniteDataLoader
from ema import EMA

from samplers.utils import sample_trajectory

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def extract_into_tensor(tensor, shape):
    num_expand_dims = len(shape) - 1
    return tensor.view([-1] + [1 for _ in range(num_expand_dims)])


def check_grad_is_nan(model, name):
    for n, p in model.named_parameters():
        if torch.isnan(p.grad).any():
            print(f"NaN in {n} grad of {name}")
            return True
    return False


@torch.no_grad()
def plot_training_summary(energy, fwd_model, sampler, fwd_losses, 
                          bwd_losses, energy_losses, gamma, 
                          n_steps, t_max, noise_std, n_samples=512):
    fig = plt.figure(figsize=(16, 16))
    
    # Create grid spec for custom layout
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 0.7, 0.7])
    
    # Plot 1: Samples x_0 and x_1
    ax1 = fig.add_subplot(gs[0, 0])
    with torch.no_grad():
        x_0, _ = sampler.sample(n_samples)
        x_0n = x_0 + torch.randn_like(x_0) * noise_std
        x_1 = sample_trajectory(fwd_model, x_0n, "forward", gamma, n_steps, t_max, only_last=True)
    
    ax1.scatter(x_0[:, 0], x_0[:, 1], alpha=0.5, label='x_0', color='blue')
    ax1.scatter(x_1[:, 0], x_1[:, 1], alpha=0.5, label='x_1', color='red')
    ax1.legend()
    ax1.set_title('Samples x_0 and x_1')
    ax1.set_xlim(-5, 5)
    ax1.set_ylim(-5, 5)
    
    # Plot 2: Energy function
    ax2 = fig.add_subplot(gs[0, 1])
    limits = (-5, 5)
    x = np.linspace(*limits, 200)
    y = np.linspace(*limits, 200)
    X, Y = np.meshgrid(x, y)
    grid_points = np.stack([X, Y], axis=-1)
    grid_points = torch.from_numpy(grid_points).reshape(-1, 2).float()
    
    with torch.no_grad():
        log_density = -energy(grid_points).detach()
        log_density = log_density - log_density.max()
    Z = log_density.numpy().reshape(200, 200)
    
    contour = ax2.contour(X, Y, Z, levels=7, colors='k')
    ax2.clabel(contour, inline=True, fontsize=6)
    im = ax2.contourf(X, Y, Z, levels=10, cmap='viridis', alpha=0.5)
    ax2.set_title('- E(x)')
    fig.colorbar(im, ax=ax2, label='Energy Value')
    ax2.set_xlim(*limits)
    ax2.set_ylim(*limits)
    
    # Plot 3: Forward losses
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(fwd_losses)
    ax3.set_title('Forward Losses')
    ax3.set_xlabel('Iterations')
    ax3.set_ylabel('Loss')
    ax3.grid(True)
    if min(fwd_losses) < -10 or max(fwd_losses) > 100:
        ax3.set_yscale('symlog')
        ax3.set_ylabel('Loss (symlog scale)')
    
    # Plot 4: Backward losses
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(bwd_losses)
    ax4.set_title('Backward Losses')
    ax4.set_xlabel('Iterations')
    ax4.set_ylabel('Loss')
    ax4.grid(True)
    if min(bwd_losses) < -10:
        ax4.set_yscale('symlog')
        ax4.set_ylabel('Loss (symlog scale)')
    
    # Plot 5: Energy losses
    ax5 = fig.add_subplot(gs[2, :])
    ax5.plot(energy_losses)
    ax5.set_title('Energy Losses')
    ax5.set_xlabel('Iterations')
    ax5.set_ylabel('Loss')
    ax5.grid(True)
    if min(energy_losses) < -10:
        ax5.set_yscale('symlog')
        ax5.set_ylabel('Loss (symlog scale)')
    
    plt.tight_layout()
    plt.show()


@torch.no_grad()
def plot_losses(fwd_losses, bwd_losses, energy_losses):
    """
    Plot the three loss curves in a single row.
    
    Args:
        fwd_losses: List of forward process losses
        bwd_losses: List of backward process losses
        energy_losses: List of energy function losses
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot forward losses
    axes[0].plot(fwd_losses)
    axes[0].set_title('Forward Losses')
    axes[0].set_xlabel('Iterations')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True)
    if min(fwd_losses) < -10 or max(fwd_losses) > 100:
        axes[0].set_yscale('symlog')
        axes[0].set_ylabel('Loss (symlog scale)')
    
    # Plot backward losses
    axes[1].plot(bwd_losses)
    axes[1].set_title('Backward Losses')
    axes[1].set_xlabel('Iterations')
    axes[1].set_ylabel('Loss')
    axes[1].grid(True)
    if min(bwd_losses) < -10:
        axes[1].set_yscale('symlog')
        axes[1].set_ylabel('Loss (symlog scale)')
    
    # Plot energy losses
    axes[2].plot(energy_losses)
    axes[2].set_title('Energy Losses')
    axes[2].set_xlabel('Iterations')
    axes[2].set_ylabel('Loss')
    axes[2].grid(True)
    if min(energy_losses) < -10:
        axes[2].set_yscale('symlog')
        axes[2].set_ylabel('Loss (symlog scale)')
    
    plt.tight_layout()
    plt.show()


def plot_image_grajectory(images: list[torch.Tensor], timesteps: list[str], title, apply_rescale=True):
    n_images = len(images)
    h = images[0].size(2)

    images = torch.stack(images, dim=0).permute(1, 0, 2, 3, 4)
    images = images.reshape(-1, 1, h, h)
    
    if apply_rescale:
        images = (images + 1) / 2
    images = torch.clip(images, 0.0, 1.0)

    image_grid = torchvision.utils.make_grid(images, nrow=n_images, 
                                             padding=1, pad_value=1)
    tick_positions = np.arange(n_images) * (h + 1) + 1 + h // 2
    
    plt.figure(figsize=(12, 6))
    plt.title(title)
    plt.imshow(image_grid.permute(1, 2, 0))
    plt.xticks(tick_positions, timesteps, rotation=0)
    plt.xlabel("Image Index")
    plt.yticks([])
    plt.show()


def train_mnist_ebm():
    mnist_transform = v2.Compose([
        v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
        v2.Resize((8, 8)),
        v2.Lambda(lambda x: 2*x - 1)
    ])
    mnist_dataset = torchvision.datasets.MNIST('./mnist/data', train=True, download=True, 
                                            transform=mnist_transform)

    sampler = InfiniteDataLoader(mnist_dataset, batch_size=16, shuffle=True)

    alpha = 0.24
    dt = 0.005
    t_max = 0.1
    n_steps = 20

    assert dt * n_steps == t_max

    x_0 = sampler.sample(1)
    trajectory, timesteps = sample_trajectory(ref_process, x_0, 'backward', dt, 
                                            n_steps, t_max, return_timesteps=True)
    plot_image_grajectory(trajectory, timesteps, "Trajectory of the reference process")


    energy = MNISTEnergy()
    fwd_model = MNISTSampler(1, 1, 16, 16)
    bwd_model = MNISTSampler(1, 1, 16, 16)

    energy_ema = EMA(energy, decay=0.999)
    energy_optim = torch.optim.Adam(energy.parameters(), lr=2e-4)
    fwd_optim = torch.optim.Adam(fwd_model.parameters(), lr=2e-4)
    bwd_optim = torch.optim.Adam(bwd_model.parameters(), lr=1e-3)

    num_bwd_iters = 1
    num_fwd_iters = 5
    num_energy_iters = 1
    milestones = [3000, 8000]
    fwd_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        fwd_optim, 
        milestones=[ms * num_fwd_iters for ms in milestones], 
        gamma=0.4
    )
    bwd_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        bwd_optim, 
        milestones=[ms * num_bwd_iters for ms in milestones], 
        gamma=0.4
    )
    energy_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        energy_optim, 
        milestones=[ms * num_energy_iters for ms in milestones], 
        gamma=0.4
    )

    p1_buffer = ReplayBuffer(2048, update_fraction=1.0)
    fwd_losses, bwd_losses, energy_losses = [], [], []
    current_iter = 0

    seed_everything(3704)
    N_TRAJECTORIES = 2
    N_SB_STEPS = 10_000
    BATCH_SIZE = 16

    logging_frequency = 20
    noise_std = 0.0
    use_buffer = False
    resample_x_0 = True


    for it in range(current_iter, N_SB_STEPS):
        current_iter = it

        if it >= 5500:
            energy_optim.param_groups[0]['lr'] = 4e-5
            fwd_optim.param_groups[0]['lr'] = 4e-5
            bwd_optim.param_groups[0]['lr'] = 1e-4
        
        x_train = sampler.sample(BATCH_SIZE)

        # TRAIN BACKWARD PROCESS
        for _ in range(num_bwd_iters):
            bwd_optim.zero_grad(set_to_none=True)

            if resample_x_0:
                x_train = sampler.sample(BATCH_SIZE)
            x_0 = x_train + torch.randn_like(x_train) * noise_std

            _fwd_model = ref_process if it == 0 else fwd_model
            loss = losses.compute_bwd_tlm_loss(
                _fwd_model, bwd_model, x_0, dt, t_max, n_steps
            )
        
            assert not torch.isnan(loss).any(), "backward loss is NaN"
            check_grad_is_nan(bwd_model, "bwd_model")
            
            bwd_optim.step()
            bwd_scheduler.step()
            
        bwd_losses.append(loss.item())

        # LOG BACKWARD TRAJECTORY
        if it % logging_frequency == 0:
            clear_output(wait=True)
            print(
                f'Iter={it}, backward Loss: {loss.item()},', 
                f" backward lr {bwd_optim.param_groups[0]['lr']}"
            )
            with torch.no_grad():
                x_0 = sampler.sample(4)
                # x_0 = x_0 + torch.randn_like(x_0) * noise_std
                
                _fwd_model = ref_process if it == 0 else fwd_model
                fwd_trajectory, timesteps = sample_trajectory(
                    _fwd_model, x_0, "forward", gamma, 
                    n_steps, t_max, return_timesteps=True
                )
                bwd_trajectory, timesteps = sample_trajectory(
                    bwd_model, fwd_trajectory[-1], "backward", 
                    gamma, n_steps, t_max, return_timesteps=True
                )
            
            plot_image_grajectory(bwd_trajectory, timesteps, "Backward trajectory")

        # TRAIN FORWARD PROCESS
        for _ in range(num_fwd_iters):
            fwd_optim.zero_grad(set_to_none=True)

            if resample_x_0:
                x_train = sampler.sample(BATCH_SIZE)

            x_0 = x_train[:BATCH_SIZE // N_TRAJECTORIES].repeat(N_TRAJECTORIES, 1, 1, 1)
            x_0 = x_0 + torch.randn_like(x_0) * noise_std
            
            loss = losses.compute_fwd_vargrad_loss(
                fwd_model, bwd_model, lambda x: - energy(x),
                x_0, dt, t_max, n_steps, 
                p1_buffer=p1_buffer,
                n_trajectories=N_TRAJECTORIES,
                clip_range=(-10000, 10000)
            )
            
            assert not torch.isnan(loss).any(), "forward loss is NaN"
            loss.backward()
            check_grad_is_nan(fwd_model, "fwd_model")
            
            fwd_optim.step()
            fwd_scheduler.step()

        fwd_losses.append(loss.item())

        # LOG FORWARD TRAJECTORY
        if it % logging_frequency == 0:
            print(
                f'Iter={it}, Forward Loss: {loss.item()},', 
                f" forward lr {fwd_optim.param_groups[0]['lr']}"
            )
            with torch.no_grad():
                x_0 = sampler.sample(4)
                # x_0 = x_0 + torch.randn_like(x_0) * noise_std
                fwd_trajectory, timesteps = sample_trajectory(
                    fwd_model, x_0, "forward", gamma, 
                    n_steps, t_max, return_timesteps=True
                )
            
            plot_image_grajectory(fwd_trajectory, timesteps, "Forward trajectory")

        # TRAIN ENERGY FUNCTION
        for _ in range(num_energy_iters):
            if resample_x_0:
                x_train = sampler.sample(BATCH_SIZE)

            if use_buffer:
                x_1 = p1_buffer.sample(BATCH_SIZE)
            else:
                x_train_noisy = x_train + torch.randn_like(x_train) * noise_std
                x_1 = sample_trajectory(fwd_model, x_train_noisy, "forward", dt, 
                                        n_steps, t_max, only_last=True)

            energy_optim.zero_grad(set_to_none=True)
            loss = losses.ebm_loss(energy, x_train, x_1, alpha=0.01, reg_type='l2')

            assert not torch.isnan(loss).any(), "energy loss is NaN"
            loss.backward()
            check_grad_is_nan(energy, "energy")

            energy_optim.step()
            # energy_ema.update()
            energy_scheduler.step()
        
        energy_losses.append(loss.item())

        # LOG ENERGY FUNCTION
        limits = (-5, 5)
        if it % logging_frequency == 0:
            # energy_ema.apply_shadow()
            print(
                f'Iter={it}, Energy Loss: {loss.item()},', 
                f" energy lr {energy_optim.param_groups[0]['lr']}"
            )
            plot_losses(fwd_losses, bwd_losses, energy_losses)
            # energy_ema.restore()

        