from torch import nn


class MnistGen(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.z_dim = z_dim

        self.net = nn.Sequential(
            nn.Linear(z_dim, 200), nn.ReLU(),
            nn.Linear(200, 400), nn.ReLU(),
            nn.Linear(400, 784), nn.Tanh(),
        )
    
    def forward(self, z):
        img = self.net(z)
        return img.reshape(-1, 1, 28, 28)


class MnistDisc(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(784, 200), nn.LeakyReLU(),
            nn.Linear(200, 1), 
        )

    def forward(self, x):
        x = x.reshape(-1, 784)
        return self.net(x)
