import torch.nn as nn


class CifarGen(nn.Module):
    def __init__(self, latent_dim: int = 100, channels: int = 3):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 32, 4, 1, 0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(32, 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(16, 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(8, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)


class CifarDisc(nn.Module):
    def __init__(self, channels: int = 3, sigmoid: bool = False):
        super().__init__()
        ch = 32
        self.main = nn.Sequential(
            nn.Conv2d(channels, ch, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ch, ch * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ch * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ch * 2, ch * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ch * 4),
            nn.LeakyReLU(0.2, inplace=True),
    
            nn.Conv2d(ch * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid() if sigmoid else nn.Identity()
        )

    def forward(self, x):
        return self.main(x).view(-1, 1).squeeze(1)
