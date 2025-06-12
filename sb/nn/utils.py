import collections
import math
import torch


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


class ReferenceProcess2:
    def __init__(self, alpha: float):
        self.alpha = alpha
    
    def __call__(self, x, t):
        return ModelOutput(drift= - self.alpha * x)
