
import torch
from torch import nn

from einops.layers.torch import Rearrange


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x - torch.mean(x, dim=1, keepdim=True)
        out = out / torch.sqrt(torch.square(out).mean(dim=1, keepdim=True) + self.eps)
        if self.elementwise_affine:
            out = out * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return out


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()
        self.down_conv = nn.Conv2d(in_channels, out_channels, 
                                   kernel_size=3, stride=2, padding=1)
        self.res_block = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 
                      kernel_size=3, stride=1, padding=1),
            LayerNorm2d(out_channels),
            nn.SiLU(),
        )

    def forward(self, x):
        x = self.down_conv(x)
        return x + self.res_block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()
        self.down_conv = nn.ConvTranspose2d(in_channels, out_channels, 
                                            kernel_size=4, stride=2, padding=1)
        self.res_block = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 
                      kernel_size=3, stride=1, padding=1),
            LayerNorm2d(out_channels),
            nn.SiLU(),
        )

    def forward(self, x):
        x = self.down_conv(x)
        return x + self.res_block(x)


# Define the VAE model
class CifarVAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        
        self.encoder = nn.Sequential(
            # nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            # nn.SiLU(),
            # nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            # nn.SiLU(),
            # nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            # nn.SiLU(),
            # nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            # nn.SiLU(),
            nn.Conv2d(3, 32, kernel_size=1),
            DownBlock(32, 64),
            DownBlock(64, 128),
            DownBlock(128, 256),
            # DownBlock(128, 256),
            Rearrange('b c h w -> b (c h w)'),
            nn.Linear(256 * 4 * 4, latent_dim * 2),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256 * 4 * 4),
            Rearrange('b (c h w) -> b c h w', c=256, h=4, w=4),
            # nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            # nn.SiLU(),
            # nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            # nn.SiLU(),
            # nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            # nn.SiLU(),
            # nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            UpBlock(256, 128),
            UpBlock(128, 64),
            UpBlock(64, 32),
            # UpBlock(32, 32),
            nn.Conv2d(32, 3, kernel_size=1),
            nn.Sigmoid(),
        )

    def state_dict(self):
        return {'encoder': self.encoder.state_dict(),
                'decoder': self.decoder.state_dict()}
    
    def load_state_dict(self, state_dict):
        self.encoder.load_state_dict(state_dict['encoder'])
        self.decoder.load_state_dict(state_dict['decoder'])

    def encode(self, x):
        mu, logvar = self.encoder(x).chunk(2, dim=1)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, z, mu, logvar
