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