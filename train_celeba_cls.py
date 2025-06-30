from typing import Tuple
import os
import click
from tqdm.auto import tqdm, trange

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from torchvision import io
from torchvision.transforms import v2 as transforms

import pandas as pd

import wandb

import sys
sys.path.append('./external/convnext')
from models.convnext import ConvNeXt


class CelebACustomCSV(Dataset):
    def __init__(self, image_dir, attr_csv, split_csv=None, split="train", transform=None):
        self.image_dir = image_dir
        self.attr_df = pd.read_csv(attr_csv)
        
        if split_csv is not None:
            split_df = pd.read_csv(split_csv)
            split_df = split_df[split_df['split'] == split]
            self.attr_df = self.attr_df[self.attr_df['image_id'].isin(split_df['image_id'])]

        self.transform = transform
        self.image_ids = self.attr_df['image_id'].values
        self.labels = self.attr_df.drop(columns=['image_id']).values.astype(int)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path = os.path.join(self.image_dir, image_id)
        image = io.read_image(img_path)

        image = image / 255

        if self.transform:
            image = self.transform(image)

        label = (torch.tensor(self.labels[idx]) + 1) // 2
        return image, label


class MultclassLoss(nn.Module):
    def forward(self, logits, ys):
        loss = ys * nn.functional.softplus(-logits) + \
               (1 - ys) * nn.functional.softplus(logits)
        return loss.sum(1).mean(0)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        out = nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = nn.functional.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, num_classes=40, resolution=64):
        super().__init__()
        self.resolution = resolution
        
        # Initial layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Final classifier
        if resolution == 32:
            self.avg_pool = nn.AvgPool2d(4)
        else:  # 64
            self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResBlock(in_channels, out_channels, stride))
            in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


@click.command()
@click.option('--num-epochs',       "num_epochs",       default=30,             type=click.INT, required=True, help='Number of epochs for training.')
@click.option('--device',           "device",           default=0,              type=click.INT, required=True, help='Device to use for training (e.g., "cuda" or "cpu").')
@click.option('--lr',               "lr",               default=5e-5,           type=click.FLOAT, required=True, help='Learning rate.')
@click.option('--batch-size',       "batch_size",       default=512,            type=click.INT, required=True, help='Batch size for training.')
@click.option('--val-batch-size',   "val_batch_size",   default=1024,           type=click.INT, required=True, help='Batch size for validation.')
@click.option('--num-classes',      "num_classes",      default=40,             type=click.INT, required=True, help='Number of classes.')
@click.option('--num-workers',      "num_workers",      default=32,             type=click.INT, required=True, help='Number of workers for data loading.')
@click.option('--weight-decay',     "weight_decay",     default=1e-8,           type=click.FLOAT, required=True, help='Weight decay for optimizer.')
@click.option('--betas',            "betas",            default=(0.9, 0.999),   type=(click.FLOAT, click.FLOAT), required=True, help='Betas for Adam optimizer.')
@click.option('--t-max',            "T_max",            default=150,            type=click.INT, required=True, help='Maximum number of iterations for the scheduler.')
@click.option('--ckpt-path',        "ckpt_path",        default=None,           type=click.Path(exists=True), help='Path to the checkpoint file to resume training from.')
@click.option('--resolution',        "resolution",      default=256,            type=click.INT, help='Input Image resolution.')
def train(
    num_epochs: int,
    device: int,
    lr: float,
    batch_size: int,
    val_batch_size: int,
    num_classes: int,
    num_workers: int,
    weight_decay: float,
    betas: Tuple[int, int],
    T_max: int,
    ckpt_path: str = None,
    resolution: int = 256
):  
    assert resolution in {32, 64, 256}, "Resolution must be one of [32, 64, 256]"
    
    torch.manual_seed(42)
    device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    celeba_transform = transforms.Compose([
        transforms.Resize([resolution, resolution])
    ])
    train_data = CelebACustomCSV(
        image_dir='./data/celebA/img_align_celeba/img_align_celeba', 
        attr_csv='./data/celebA/list_attr_celeba.csv',
        split="train",
        transform=celeba_transform,
    )
    test_data = CelebACustomCSV(
        image_dir='./data/celebA/img_align_celeba/img_align_celeba', 
        attr_csv='./data/celebA/list_attr_celeba.csv',
        split="test",
        transform=celeba_transform,
    )
    train_loader = DataLoader(train_data, batch_size=batch_size, 
                              shuffle=True, num_workers=num_workers,
                              drop_last=True, 
                              prefetch_factor=2)
    test_loader = DataLoader(test_data, batch_size=val_batch_size, 
                             shuffle=False, num_workers=num_workers)
    loss_fn = MultclassLoss()

    if resolution == 256:
        print("Building ConvNeXt model...")
        model = ConvNeXt(num_classes=num_classes)
        model.to(device)
        print("ConvNeXt model is build")
    else:
        print("Building ResNet model...")
        model = ResNet(num_classes=num_classes, resolution=resolution)
        model.to(device)
        print("ResNet model is build")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max)

    val_images, val_ys = (t.to(device) for t in next(iter(test_loader)))

    config = {
        "num_epochs": num_epochs,
        "device": device,
        "lr": lr,
        "batch_size": batch_size,
        "val_batch_size": val_batch_size,
        "num_classes": num_classes,
        "num_workers": num_workers,
        "betas": betas,  
    }
    if ckpt_path is not None: 
        print(f"Loading checkpoint from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        config.update(checkpoint['config'])

    wandb.init(project='celeba-cls', name=f'celeba-cls-{resolution=}', config=config)

    step = 0
    for epoch in trange(num_epochs, desc="Epochs"):
        model.train()
        for (images, ys) in tqdm(
                train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False
            ):
            optimizer.zero_grad(set_to_none=True)
            images = images.to(device)
            ys = ys.to(device)

            logits = model(images)
            loss = loss_fn(logits, ys)

            loss.backward()
            optimizer.step()
            scheduler.step()
            
            wandb.log({
                "train/loss": loss,
                "step": step
            }, step=step)
            
            step += 1

        with torch.no_grad():
            model.eval()
            val_logits = model(val_images)
            val_probas = nn.functional.sigmoid(val_logits)
            pred_categories = (val_probas > 0.5).float()

            per_class_acc = (pred_categories == val_ys).float().mean(dim=0)
            log_dict = {}

            for class_idx, acc in enumerate(per_class_acc):
                log_dict[f'class_{class_idx}'] = acc

            wandb.log(log_dict, step=step)


        checkpoint = {
            'config': config,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch,
        }
        torch.save(checkpoint, f'{wandb.run.dir}/celeba_convnext_epoch_{epoch}.pth')


if __name__ == "__main__":
    train()





