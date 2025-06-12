import torch
import torch.nn as nn


class Module(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.config = kwargs

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path)
        model = cls(**checkpoint['config'])
        model.load_state_dict(checkpoint['state_dict'])
        return model
    
    def save(self, path: str):
        torch.save({
            'config': self.config,
            'state_dict': self.state_dict()
        }, path)


class Block(nn.Module):
    def __init__(self, in_dim, out_dim, use_ln: bool = False, 
                 skip_connection: bool = False):
        super().__init__()
        if in_dim != out_dim:
            assert not skip_connection, "Skip connection requires in_dim == out_dim"
        self.skip_connection = skip_connection
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim), 
            nn.LayerNorm(out_dim) if use_ln else nn.Identity(), 
            nn.ELU()
        )
    
    def forward(self, x):
        if self.skip_connection:
            return x + self.block(x)
        return self.block(x)