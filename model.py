import math
import collections

import torch 
from torch import nn


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


class SimpleNet(nn.Module):
    def __init__(
            self, 
            t_emb_size: int, 
            x_emb_size: int, 
            n_main_body_layers: int = 3,
            predict_log_var: bool = False
        ):
        super().__init__()
        self.t_emb_size = t_emb_size
        self.x_emb_size = x_emb_size
        self.use_t = t_emb_size > 0
        self.predict_log_var = predict_log_var

        self.x_embed = nn.Sequential(
            nn.Linear(2, x_emb_size),
            nn.ReLU(),
            nn.Linear(x_emb_size, x_emb_size)
        )
        
        combined_hidden_size = x_emb_size
        
        if self.use_t:
            self.t_embed = nn.Sequential(
                nn.Linear(t_emb_size, x_emb_size),
                nn.ReLU(),
                nn.Linear(x_emb_size, x_emb_size)
            )
            combined_hidden_size += x_emb_size

        layers = []
        for i in range(n_main_body_layers):
            layers.append(nn.Linear(combined_hidden_size if i == 0 else x_emb_size, x_emb_size))
            layers.append(nn.ReLU())
        self.main_body = nn.Sequential(*layers)
        
        self.drift_head = nn.Linear(x_emb_size, 2)

        if self.predict_log_var:
            self.log_var_head = nn.Linear(x_emb_size, 2)

    def forward(self, x, t):
        embeddings = self.x_embed(x)

        if self.use_t:
            t_embed = fourier_proj(t, self.t_emb_size)
            t_embed = self.t_embed(t_embed)
            embeddings = torch.cat([embeddings, t_embed], dim=-1)
        
        embeddings = self.main_body(embeddings)
        drift = self.drift_head(embeddings)

        if self.predict_log_var:
            log_var = self.log_var_head(embeddings)
            return ModelOutput(drift=drift, log_var=log_var)

        return ModelOutput(drift=drift)
