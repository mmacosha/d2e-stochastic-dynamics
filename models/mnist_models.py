import torch
import torch.nn as nn
from models.utils import ModelOutput, fourier_proj
from models.module import Module


class MNISTEnergy(Module):
    def __init__(self, as_cls: bool = False):
        super().__init__(as_cls=as_cls)
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


class MNISTSampler(Module):
    def __init__(self, in_channels=1, out_channels=1, 
                 base_channels=32, t_emb_size=32,
                 train_var: bool = False):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            t_emb_size=t_emb_size,
            train_var=train_var,
        )
        self.train_var = train_var
        self.t_emb_size = t_emb_size
        self.t_embed = nn.Linear(t_emb_size, base_channels)
        
        self.encoder1 = ConvBlock(t_emb_size, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = ConvBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(base_channels * 2, base_channels * 4)

        self.upconv2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 
                                          kernel_size=2, stride=2)
        self.decoder2 = ConvBlock(base_channels * 4, base_channels * 2)
        self.upconv1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 
                                          kernel_size=2, stride=2)
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
