import torch
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid

import wandb

from sb.nn.mnist import MnistCLS, MnistVAE

from .utils import get_dataloader


def train_mnist_vae(cfg):
    vae = MnistVAE()
    optimizer = torch.optim.AdamW(vae.parameters(), lr=1e-3)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))
    ])

    train_loder, val_loader = get_dataloader(MNIST, cfg, transform)
    n_epochs = 10
    vae.train()

    fixed_val_batch, *_ = next(iter(val_loader))
    run_dir = wandb.run.dir
    
    for epoch in range(n_epochs):
        for i, (x, _) in enumerate(train_loder):
            optimizer.zero_grad()
            loss, l2, kl = vae.compute_loss(x, kl_weight=1.0)
            loss.backward()
            optimizer.step()

        wandb.log({
            'epoch': epoch,
            'loss': loss.item(),
            'l2_loss': l2.item(),
            'kl_loss': kl.item()
        }, step=epoch)

        with torch.no_grad():
            img_rec, *_ = vae(fixed_val_batch.view(-1, 784))
            z = torch.randn(64, 10, device=img_rec.device)
            img_gen = vae.decoder(z).view(-1, 1, 28, 28)
        



def train_mnist_cls(cfg):
    for epoch in range(n_epochs):
        for i, (x, y) in enumerate(dataloader):
            cls_opt.zero_grad(set_to_none=True)
            
            if torch.rand(1) < 0.5:
                x, *_ = vae(x)

            logits = cls_model(x)

            loss = nn.CrossEntropyLoss()(logits, y)
            loss.backward()
            cls_opt.step()

        print(f'Epoch {epoch+1}/{n_epochs}, Step {i}, Loss: {loss.item():.4f}')