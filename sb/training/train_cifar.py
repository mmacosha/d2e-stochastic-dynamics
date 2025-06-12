from tqdm import trange
from pathlib import Path

import torch
from torch import nn
from torch import optim

from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import make_grid

import wandb
import numpy as np
from omegaconf import OmegaConf
from tqdm import trange, tqdm

from sb.nn.cifar import (
    CifarDisc, 
    CifarGen,
    CifarVAE,
    CifarCls    
    )

from .utils import get_dataloader


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def generate_and_log_images(generator, step, fixed_noise):
    with torch.no_grad():
        fake_images = generator(fixed_noise)
        fake_images = fake_images.detach().cpu() * 0.5 + 0.5
        grid = make_grid(fake_images, nrow=8, padding=2, normalize=False)
        wandb.log({"generated_images": [wandb.Image(grid)]}, step=step)


def cycle(iterable):
    while True:
        for i in iterable:
            yield i


def vae_loss_function(recon_x, x, z, mu, logvar, beta=1.0):
    x = x.view(-1, 3 * 32 * 32)
    recon_x = recon_x.view(-1, 3 * 32 * 32)
    
    rec_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kl = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1)
    
    return rec_loss + beta * kl, rec_loss.mean(), kl.mean()


def train_vae_for_epoch(model, config, train_loader, optimizer, scheduler, epoch):
    model.train()
    train_loss = 0
    recon_loss = 0
    kl_loss = 0
    
    beta = max(3.0, config.beta + epoch * 0.025)
    for _, (data, _) in enumerate(train_loader):
        data = data.to(config.device)
        optimizer.zero_grad()
        
        recon_batch, z, mu, logvar = model(data)
        loss, bce, kld = vae_loss_function(
            recon_batch, data, z, mu, logvar, beta
        )
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        recon_loss += bce.item()
        kl_loss += kld.item()
        
    avg_loss = train_loss / len(train_loader.dataset)
    avg_recon = recon_loss / len(train_loader.dataset)
    avg_kl = kl_loss / len(train_loader.dataset)
    
    wandb.log({
        'epoch': epoch + 1,
        'train_loss': avg_loss,
        'reconstruction_loss': avg_recon,
        'kl_divergence': avg_kl,
        'lr': scheduler.get_last_lr()[0]
    }, step=epoch)
    scheduler.step()
    
    return avg_loss


def test_vae_for_epoch(model, config, test_loader, epoch):
    model.eval()
    test_loss = 0
    
    beta = max(3.0, config.beta + epoch * 0.025)
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(config.device)
            recon, z, mu, logvar = model(data)
            
            loss, _, _ = vae_loss_function(recon, data, z, mu, logvar, beta)
            test_loss += loss.item()
    
    test_loss /= len(test_loader.dataset)
    
    original_image = wandb.Image(
        make_grid(data[:64].view(64, 3, 32, 32), nrow=8, normalize=True),
        caption=f"Epoch {epoch+1}: Top: Original, Bottom: Reconstructed"
    )
    reconstructed_images = wandb.Image(
        make_grid(recon[:64].view(64, 3, 32, 32), nrow=8, normalize=True),
        caption=f"Epoch {epoch+1}: Top: Original, Bottom: Reconstructed"
    )
    
    with torch.no_grad():
        sample = torch.randn(64, config.latent_dim).to(config.device)
        sample = model.decode(sample).cpu()
        
        generated_images = wandb.Image(
            make_grid(sample.view(64, 3, 32, 32), nrow=8, normalize=True),
            caption=f"Epoch {epoch+1}: Generated samples"
        )

    wandb.log({
        "test_loss": test_loss,
        "original images": original_image,
        "reconstructions": reconstructed_images,
        "generated_samples": generated_images,
    }, step=epoch)
    
    return test_loss


def run_gan_training(cfg):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x - 0.5) * 2), 
    ])
    train_loader, _ = get_dataloader(CIFAR10, cfg, transform)
    train_loader = cycle(train_loader)

    generator = CifarGen(latent_dim=cfg.latent_dim).to(cfg.device)
    discriminator = CifarDisc().to(cfg.device)

    generator.apply(weights_init)
    discriminator.apply(weights_init)

    optimizer_g = optim.Adam(generator.parameters(), lr=cfg.glr, 
                             betas=cfg.gbetas, weight_decay=cfg.gdecay)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=cfg.dlr, 
                             betas=cfg.dbetas, weight_decay=cfg.ddecay)

    run_dir = Path(wandb.run.dir)

    wandb.watch(generator)
    wandb.watch(discriminator)

    fixed_noise = torch.randn(64, cfg.latent_dim, 1, 1, device=cfg.device)

    for step in trange(cfg.n_steps):
        for _ in range(20):
            images, _ = next(train_loader)
            batch_size = images.size(0)
            real_images = images.to(cfg.device)
            noise = torch.randn(batch_size, cfg.latent_dim, 1, 1, device=cfg.device)
            fake_images = generator(noise)

            real_output = discriminator(real_images)
            fake_output = discriminator(fake_images.detach())

            d_loss_real = nn.functional.softplus(-real_output).mean()
            d_loss_fake = nn.functional.softplus(fake_output).mean()

            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward()
            optimizer_d.step()

            D_x = nn.functional.sigmoid(real_output).mean().item()
            D_G_z1 = nn.functional.sigmoid(fake_output).mean().item()
        
        optimizer_g.zero_grad()
        fake_images = generator(noise)
        output = discriminator(fake_images)
        g_loss = nn.functional.softplus(-output).mean()
        g_loss.backward()
        optimizer_g.step()
        
        D_G_z2 = nn.functional.sigmoid(output).mean().item()
        
        if step % cfg.log_freq == 0:
            wandb.log({
                "step": step,
                "d_loss": d_loss.item(),
                "g_loss": g_loss.item(),
                "D_x": D_x,
                "D_G_z1": D_G_z1,
                "D_G_z2": D_G_z2
            }, step=step)
        
        if step % cfg.image_freq == 0:
            generate_and_log_images(generator, step, fixed_noise)
        
        if step % cfg.save_freq == 0:
            torch.save(
                generator.state_dict(), run_dir / f'generator_epoch_{step}.pth'
            )
            torch.save(
                discriminator.state_dict(), run_dir / f'discriminator_epoch_{step}.pth'
            )
            wandb.save(f'generator_epoch_{step}.pth')
            wandb.save(f'discriminator_epoch_{step}.pth')

    wandb.finish()


def run_vae_training(cfg):
    model = CifarVAE(latent_dim=cfg.latent_dim).to(cfg.device)
    
    optimizer = optim.Adam(model.parameters(), 
                           lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.lr_decay)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_loader, test_loader = get_dataloader(CIFAR10, cfg, transform)

    run_dir = Path(wandb.run.dir)
    wandb.watch(model, log="all", log_freq=100)

    best_loss = float('inf')
    for epoch in (pbar := trange(cfg.n_epochs, desc='Training', 
                        leave=False, unit='epoch')):
        train_loss = train_vae_for_epoch(model, cfg, train_loader, optimizer, scheduler, epoch)
        pbar.set_postfix({'loss': train_loss}) 
        
        with torch.no_grad():
             
            test_loss = test_vae_for_epoch(model, cfg, test_loader, epoch)
        
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), run_dir / 'best_cifar_vae-t.pt')
            wandb.save(run_dir / 'best_cifar_vae.pt')

    wandb.run.summary['num_model_params'] = sum(p.numel() for p in model.parameters())
    wandb.finish()


def run_cls_training(cfg):
    model = CifarCls().to(cfg.device)
    optimizer = optim.Adam(model.parameters(), 
                           lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_loader, test_loader = get_dataloader(CIFAR10, cfg, transform)

    run_dir = Path(wandb.run.dir)
    wandb.watch(model, log="all", log_freq=100)

    for epoch in range(1, cfg.n_epochs + 1):
        train_acc, test_acc = 0, 0
        train_loss, test_loss = 0, 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.n_epochs}",
                          leave=False, unit="batch"):
            data, target = (t.to(cfg.device) for t in batch)

            optimizer.zero_grad()
            output = model(data)
            loss = nn.functional.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += (output.argmax(dim=1) == target).sum().item()

        with torch.no_grad():
            for test_batch in test_loader:
                data, target = (t.to(cfg.device) for t in test_batch)

                output = model(data)
                loss = nn.functional.cross_entropy(output, target)
                test_loss += loss.item()
                test_acc += (output.argmax(dim=1) == target).sum().item()
        
        train_loss /= len(train_loader)
        test_loss /= len(test_loader)
        test_acc /= len(test_loader.dataset)
        train_acc /= len(train_loader.dataset)
        
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'test_loss': test_loss,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
        }, step=epoch)
    
    wandb.run.summary['train_accuracy'] = train_acc
    wandb.run.summary['test_accuracy'] = test_acc
    wandb.run.summary['num_model_params'] = sum(p.numel() for p in model.parameters())

    torch.save(model.state_dict(), run_dir / 'cifar_classifier.pt')
    wandb.save('cifar_classifier.pt')
    wandb.finish()


def train_cifar_model(cfg):
    if cfg.model_type == 'gan':
        wandb.init(project=cfg.project, name=cfg.name, mode=cfg.mode,
                   config=OmegaConf.to_container(cfg.gan, resolve=True))
        run_gan_training(cfg.gan)
    elif cfg.model_type == 'vae':
        wandb.init(project=cfg.project, name=cfg.name, mode=cfg.mode,
                   config=OmegaConf.to_container(cfg.vae, resolve=True))
        run_vae_training(cfg.vae)
    elif cfg.model_type == 'classifier':
        wandb.init(project=cfg.project, name=cfg.name, mode=cfg.mode,
                   config=OmegaConf.to_container(cfg.cls, resolve=True))
        run_cls_training(cfg.cls)
    else:
        raise ValueError(f"Unknown model type: {cfg.model_type}")
