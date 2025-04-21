import math
import collections

import torch 
from torch import nn


class ReferenceProcess2:
    def __init__(self, alpha: float):
        self.alpha = alpha
    
    def __call__(self, x, t):
        return ModelOutput(drift= - self.alpha * x)


def fourier_proj(time, embed_dim, max_dim=1e4):
    max_log_dim = math.log(max_dim) / (embed_dim // 2 - 1)
    embeddings = torch.arange(embed_dim // 2, device=time.device) * (- max_log_dim)
    embeddings = time[:, None] * torch.exp(embeddings)[None, :]
    return torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)


def random_fourier_proj(time, embed_dim):
    W = torch.randn(1, embed_dim // 2) * 2 * torch.pi
    embeddings = time[:, None] * W
    return torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)


class ModelOutput(collections.UserDict):
    def __getattr__(self, name):
        return self.data[name]
    
    def contains(self, name) -> bool:
        return name in self.data
    
    def detach(self):
        for name, value in self.data.items():
            self.data[name] = value.detach()
    
    def cpu(self):
        for name, value in self.data.items():
            self.data[name] = value.cpu()
    
    def cuda(self, device=None):
        for name, value in self.data.items():
            self.data[name] = value.cuda(device)


class SimpleNetBlock(nn.Module):     
    def __init__(self, in_dim, out_dim, skip_connection=False):
        super().__init__()
        self.skip_connection = skip_connection
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ELU()
        )

    def forward(self, x):
        if self.skip_connection:
            return x + self.block(x)
        return self.block(x)


class SimpleNet(nn.Module):
    def __init__(
            self, 
            x_emb_size: int, 
            in_dim: int = 2,
            t_emb_size: int | None = None ,
            n_main_body_layers: int = 2,
            predict_log_var: bool = False,
        ):
        super().__init__()
        self.x_emb_size = x_emb_size
        self.t_emb_size = t_emb_size
        self.predict_log_var = predict_log_var

        self.x_embed = nn.Sequential(
            nn.Linear(in_dim, x_emb_size),
            nn.LayerNorm(x_emb_size),
            nn.ELU(),
        )
        
        combined_hidden_size = x_emb_size
        
        if self.t_emb_size is not None:
            self.t_embed = nn.Sequential(
                nn.Linear(t_emb_size, x_emb_size),
                nn.LayerNorm(x_emb_size),
                nn.ELU(),
            )
            combined_hidden_size += x_emb_size

        layers = []
        for i in range(n_main_body_layers):
            layers.append(
                nn.Linear(
                    combined_hidden_size if i == 0 else x_emb_size,
                    x_emb_size)
            )
            layers.append(nn.LayerNorm(x_emb_size))
            layers.append(nn.ELU())
        self.main_body = nn.Sequential(*layers)
        
        self.drift_head = nn.Linear(x_emb_size, 2)

        if self.predict_log_var:
            self.log_var_head = nn.Linear(x_emb_size, 2)

    def forward(self, x, t):
        if self.t_emb_size is None:
            x = torch.cat([x, t.view(-1, 1)], dim=1)

        embeddings = self.x_embed(x)

        if self.t_emb_size is not None:
            t_embed = fourier_proj(t, self.t_emb_size)
            t_embed = self.t_embed(t_embed)
            embeddings = torch.cat([embeddings, t_embed], dim=-1)
        
        
        embeddings = self.main_body(embeddings)
        drift = self.drift_head(embeddings)

        if self.predict_log_var:
            log_var = self.log_var_head(embeddings)
            return ModelOutput(drift=drift, log_var=log_var)

        return ModelOutput(drift=drift)


class Block(nn.Module):
    def __init__(self, dim, use_ln: bool = False, 
                 skip_connection: bool = False):
        super().__init__()
        self.skip_connection = skip_connection
        self.block = nn.Sequential(
            nn.Linear(dim, dim), 
            nn.LayerNorm(dim) if use_ln else nn.Identity(), 
            nn.ELU()
        )
    
    def forward(self, x):
        if self.skip_connection:
            return x + self.block(x)
        return self.block(x)


class Energy(nn.Module):
    def __init__(self, in_dim=2, out_dim=1, 
                 hidden_dim=64, n_blocks=3, 
                 use_ln: bool = False, block_type='simple'):
        super().__init__()

        self.proj_in = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ELU()
        )
        self.proj_out = nn.Linear(hidden_dim, out_dim)
        
        self.body = nn.Sequential(
            *[Block(hidden_dim, use_ln, block_type=="res") 
              for _ in range(n_blocks)]
        )

    def forward(self, x):
        x = self.proj_in(x)
        x = self.body(x)
        out = self.proj_out(x)
        return out.squeeze(1)
    
    def save(self, path: str):
        torch.save({
            'config': {
                'in_dim': self.in_dim,
                'out_dim': self.out_dim,
                'hidden_dim': self.hidden_dim,
                'n_blocks': self.n_blocks,
                'use_ln': self.use_ln,
                'block_type': self.block_type
            },
            'state_dict': self.state_dict()
        }, path)

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path)
        model = cls(**checkpoint['config'])
        model.load_state_dict(checkpoint['state_dict'])
        return model


########################################################################
#                    MODELS FOR MNIST EXPERIMENTS                      #
########################################################################

class MNISTEnergy(nn.Module):
    def __init__(self, as_cls: bool = False):
        super().__init__()
        self.as_cls = as_cls
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.fc1 = nn.LazyLinear(128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
    
        if self.as_cls:
            return x
        return x.mean(-1)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)


class MNISTSampler(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, 
                 base_channels=32, t_emb_size=32,
                 train_var: bool = False):
        super().__init__()
        self.train_var = train_var
        self.t_emb_size = t_emb_size
        self.t_embed = nn.Linear(t_emb_size, base_channels)
        
        self.encoder1 = ConvBlock(t_emb_size, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = ConvBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(base_channels * 2, base_channels * 4)

        self.upconv2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.decoder2 = ConvBlock(base_channels * 4, base_channels * 2)
        self.upconv1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.decoder1 = ConvBlock(base_channels * 2, base_channels)

        self.drift_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        if train_var:
            self.log_var_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)


    def forward(self, x, t):
        if self.t_emb_size > 0:
            t_embed = fourier_proj(t, self.t_emb_size)
            t_embed = self.t_embed(t_embed)
            x = x + t_embed.view(-1, self.t_emb_size, 1, 1)
        
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        bottleneck = self.bottleneck(self.pool2(enc2))
        dec2 = self.decoder2(torch.cat([self.upconv2(bottleneck), enc2], dim=1))
        dec1 = self.decoder1(torch.cat([self.upconv1(dec2), enc1], dim=1))
        
        drift = self.drift_conv(dec1)
        if self.train_var:
            log_var = self.log_var_conv(dec1)
            return ModelOutput(drift=drift, log_var=log_var)
        
        return ModelOutput(drift=drift)

