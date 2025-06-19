from torch import nn


class MnistCLS(nn.Module):
    def __init__(self, input_size=784, num_classes=10, softmax=False):
        super().__init__()
        self.cls = nn.Sequential(
            nn.Linear(input_size, 400), nn.SiLU(),
            nn.Linear(400, 200), nn.SiLU(),
            nn.Linear(200, num_classes),  
            nn.Softmax(dim=1) if softmax else nn.Identity()
        )

    def state_dict(self):
        return {'cls': self.cls.state_dict()}
    
    def load_state_dict(self, state_dict):
        if 'cls'in state_dict:
            self.cls.load_state_dict(state_dict['cls'])
        else:
            self.cls.load_state_dict(state_dict)

    def forward(self, x):
        return self.cls(x.view(-1, 784))
    
    def compute_loss(self, x, y, return_logits: bool = False):
        logits = self(x)
        if return_logits:
            return logits, nn.functional.cross_entropy(logits, y)
        return nn.functional.cross_entropy(logits, y)
    