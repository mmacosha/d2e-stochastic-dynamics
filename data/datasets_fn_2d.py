import torch
from sklearn import datasets


def mix_of_gaussians(batch_size, means=None, sigmas=None):
    if means is None and sigmas is None:
        return torch.randn(batch_size, 2)
    
    if means is None:
        assert batch_size % sigmas.size(0) == 0, 'batch size should be divisible by the number of modes'
        z = torch.randn(batch_size // sigmas.size(0), sigmas.size(0), 2)
        return (z * sigmas).view(-1, 2)

    if sigmas is None:
        assert batch_size % means.size(0) == 0, 'batch size should be divisible by the number of modes'
        z = torch.randn(batch_size // means.size(0), means.size(0), 2)
        return (z + means).view(-1, 2)
    

    assert batch_size % sigmas.size(0) == 0, 'batch size should be divisible by the number of modes'
    z = torch.randn(batch_size // means.size(0), means.size(0), 2)
    return (z * sigmas + means).view(-1, 2)


def two_moons(batch_size, shift=None, noise=None, scale: float = 1.0):
    samples, _ = datasets.make_moons(batch_size, noise=(noise or 0))
    samples[:, 0] = samples[:, 0] * 2 / 3 - 1 / 3
    samples[:, 1] = samples[:, 1] * 4 / 3 - 1 / 3
    samples = torch.from_numpy(samples) + (shift if shift is not None else 0)
    return samples.float()


def two_circles(batch_size, shift=None, noise=None):
    assert noise < 0.04, 'very high noise'
    samples, _ = datasets.make_circles(batch_size, noise=(noise or 0))
    samples = torch.from_numpy(samples) + (shift if shift is not None else 0)
    return samples.float()


def s_curve(batch_size, shift=None, noise=None):
    samples, *_ = datasets.make_s_curve(batch_size, noise=(noise or 0))
    samples[:, 2] /= 2 
    samples = torch.from_numpy(samples[:, [0, 2]]) + (shift if shift is not None else 0)
    return samples.float()


def swiss_roll(batch_size, shift=None, noise=None):
    samples, _ = datasets.make_swiss_roll(batch_size, noise=(noise or 0))
    samples = samples * 8 / 7  - 1 / 7
    samples = torch.from_numpy(samples[:, [0, 2]]) / 15 + (shift if shift is not None else 0)
    return samples.float()


def checkboard(batch_size, shift=None):
    samples = torch.rand(batch_size * 3, 2) * 2 + torch.tensor([[-1, -1]])

    x_mask = (samples[:, 0] > 0.5) + (0 > samples[:, 0]) * (samples[:, 0] > -0.5)
    y_mask = (samples[:, 1] > 0.5) + (0 > samples[:, 1]) * (samples[:, 1] > -0.5)

    mask = x_mask ^ y_mask
    samples = samples[mask][:batch_size] + (shift if shift is not None else 0)
    return samples.float()
