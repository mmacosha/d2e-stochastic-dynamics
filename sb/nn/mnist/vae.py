import torch
from torch import nn


def kl_loss(mu, std):
    log_std = torch.log(std)
    return -0.5 * (1.0 + 2 * log_std - mu.pow(2) - std.pow(2)).sum(dim=1)


class MnistVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 400), nn.SiLU(),
            nn.Linear(400, 200), nn.SiLU(),
            nn.Linear(200, 20), nn.SiLU(),
            nn.Linear(20, 20)
        )
        self.decoder = nn.Sequential(
            nn.Linear(10, 200), nn.SiLU(),
            nn.Linear(200, 400), nn.SiLU(),
            nn.Linear(400, 784), nn.Sigmoid()
        )

    def encode(self, x):
        mu, log_var = self.encoder(x)
        return mu, log_var
    
    def state_dict(self):
        return {
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict()
        }
    
    def load_state_dict(self, state_dict):
        self.encoder.load_state_dict(state_dict['encoder'])
        self.decoder.load_state_dict(state_dict['decoder'])
    
    def forward(self, x):
        hidden = self.encoder(x)
        mu, log_std = hidden.chunk(2, dim=1)
        std = nn.functional.softplus(log_std)

        z = torch.randn_like(mu)
        x_pred = self.decoder(mu + z *std)
        return x_pred, mu, std

    def compute_loss(self, x, kl_weight=1.0):
        x_pred, mu, std = self(x)

        rec_loss = (x - x_pred).pow(2).sum(dim=1)  # Reconstruction loss
        kl = kl_loss(mu, std)

        loss = torch.mean(rec_loss + kl_weight * kl)
        return loss, rec_loss.mean(), kl.mean()
