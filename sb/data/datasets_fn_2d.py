import math
import torch
from sklearn import datasets


def two_moons(batch_size, noise=0.01, *args, **kwargs):
    samples, _ = datasets.make_moons(batch_size, noise=(noise or 0))
    return torch.from_numpy(samples)


def two_circles(batch_size, noise=0.01, *args, **kwargs):
    samples, _ = datasets.make_circles(batch_size, noise=(noise or 0))
    return  torch.from_numpy(samples) 


def s_curve(batch_size, noise=0.01,  *args, **kwargs):
    samples, _ = datasets.make_s_curve(batch_size, noise=(noise or 0))
    return torch.from_numpy(samples[:, [0, 2]]).float()


def swiss_roll(batch_size, noise=0.01,  *args, **kwargs):
    samples, _ = datasets.make_swiss_roll(batch_size, noise=(noise or 0))
    return torch.from_numpy(samples[:, [0, 2]]).float()


def checkboard(batch_size, shift=None):
    samples = torch.rand(batch_size * 3, 2) * 2 + torch.tensor([[-1, -1]])

    x_mask = (samples[:, 0] > 0.5) + (0 > samples[:, 0]) * (samples[:, 0] > -0.5)
    y_mask = (samples[:, 1] > 0.5) + (0 > samples[:, 1]) * (samples[:, 1] > -0.5)

    mask = x_mask ^ y_mask
    samples = samples[mask][:batch_size] + (shift if shift is not None else 0)
    return samples.float()


def two_circles_custom(batch_size, r1, r2, noise = 0, scale = 1.0):
    U = torch.randn(batch_size) * 2 * math.pi
    c = torch.stack([torch.cos(U), torch.sin(U)], dim=-1)
    mask = torch.rand_like(U) > 0.5
    c[mask] *= r1
    c[~mask] *= r2
    return (c + torch.randn_like(c) * noise) * scale
