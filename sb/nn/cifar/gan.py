import torch
import torch.nn as nn


class CifarGen(nn.Module):
    def __init__(self, z_dim, ngf, inference: bool = False):
        super().__init__()  
        self.z_dim = z_dim
        self.ngf = ngf  
        self.inference = inference
        self.main = nn.Sequential(
            nn.ConvTranspose2d(z_dim, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(False),
            
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(False),
            
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(False),
            
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        if z.ndim == 2:
            z = z.reshape(z.size(0), z.size(1), 1, 1)
        
        img = self.main(z)

        return img

    @torch.no_grad()
    def generate(self, z):
        img = self(z)
        return (img / 2 + 0.5).clip(0, 1)
    
    def save_model(self, path):
        ckpt = {
            "config": {"z_dim": self.z_dim, "ngf": self.nfg},
            "state_dict": self.state_dict()
        }
        torch.save(ckpt, path)


class CifarDisc(nn.Module):
    def __init__(self, ndf, sigmoid: bool = False):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=False),
            
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=False),
            
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=False),
            
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv2d(ndf * 8, 1, 2, 1, 0, bias=False),
            nn.Sigmoid() if sigmoid else nn.Identity()
        )

    def forward(self, x):
        return self.main(x).view(-1, 1).squeeze(1)
