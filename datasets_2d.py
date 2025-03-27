import math
import torch

from torch import distributions
from sklearn import datasets as sk_datasets


class DatasetSampler:
    def __init__(self, p_0: str, p_1: str, p_0_args: list | None = None, p_1_args: list | None = None):
        datasets = set(registry.available_datasets)
        
        assert p_0 in datasets, f'{p_0} is not one of {datasets}'
        assert p_1 in datasets, f'{p_1} is not one of {datasets}'
        
        self.p_0 = p_0
        self.p_1 = p_1

        self.p_0_args = p_0_args
        self.p_1_args = p_1_args
        
    def _iterator(self, batch_size):
        
        x_0 = registry[self.p_0](batch_size, *self.p_0_args)
        x_1 = registry[self.p_1](batch_size, *self.p_1_args)
        yield x_0, x_1

    def sample(self, batch_size):
        return next(self._iterator(batch_size))


class DatasetRegistry:
    def __init__(self):
        self.dataset_generators = {}

    def add(self, func=None, name=None):
        def _decorator(func):
            self.dataset_generators[name or func.__name__] = func
            return func
        
        return _decorator if func is None else _decorator(func) 
    
    def __getitem__(self, name):
        return self.dataset_generators[name]
    
    @property
    def available_datasets(self):
        return list(self.dataset_generators)


registry = DatasetRegistry()


def rotate_mean(mean, angle=torch.pi / 3):
    rotate = torch.as_tensor([
        [math.cos(angle), -math.sin(angle)],
        [math.sin(angle), math.cos(angle)],
    ])
    return mean @ rotate


@registry.add(name='mix_of_gaussians')
def mix_of_gaussians(batch_size, means=None, sigmas=None):
    if means is None and sigmas is None:
        return torch.randn(batch_size, 2)
    
    if means is None:
        assert batch_size % sigmas.size(0) == 0, 'batch size should be divisible by the number of modes'
        # sigmas = torch.tensor(sigmas)
        z = torch.randn(batch_size // sigmas.size(0), sigmas.size(0), 2)
        return (z * sigmas).view(-1, 2)

    if sigmas is None:
        assert batch_size % means.size(0) == 0, 'batch size should be divisible by the number of modes'
        # means = torch.tensor(means)
        z = torch.randn(batch_size // means.size(0), means.size(0), 2)
        return (z + means).view(-1, 2)
    

    assert batch_size % sigmas.size(0) == 0, 'batch size should be divisible by the number of modes'
    z = torch.randn(batch_size // means.size(0), means.size(0), 2)
    return (z * sigmas + means).view(-1, 2)


@registry.add(name='two_moons')
def two_moons(batch_size, shift=None, noise=None):
    samples, _ = sk_datasets.make_moons(batch_size, noise=(noise or 0))
    samples[:, 0] = samples[:, 0] * 2 / 3 - 1 / 3
    samples[:, 1] = samples[:, 1] * 4 / 3 - 1 / 3
    samples = torch.from_numpy(samples) + (shift if shift is not None else 0)
    return samples.float()


@registry.add(name='two_circles')
def two_circles(batch_size, shift=None, noise=None):
    assert noise < 0.04, 'very high noise'
    samples, _ = sk_datasets.make_circles(batch_size, noise=(noise or 0))
    samples = torch.from_numpy(samples) + (shift if shift is not None else 0)
    return samples.float()


@registry.add(name='s_curve')
def s_curve(batch_size, shift=None, noise=None):
    samples, *_ = sk_datasets.make_s_curve(batch_size, noise=(noise or 0))
    samples[:, 2] /= 2 
    samples = torch.from_numpy(samples[:, [0, 2]]) + (shift if shift is not None else 0)
    return samples.float()


@registry.add(name='swiss_roll')
def swiss_roll(batch_size, shift=None, noise=None):
    samples, _ = sk_datasets.make_swiss_roll(batch_size, noise=(noise or 0))
    samples = samples * 8 / 7  - 1 / 7
    samples = torch.from_numpy(samples[:, [0, 2]]) / 15 + (shift if shift is not None else 0)
    return samples.float()


@registry.add(name='checkboard')
def checkboard(batch_size, shift=None):
    samples = torch.rand(batch_size * 3, 2) * 2 + torch.tensor([[-1, -1]])

    x_mask = (samples[:, 0] > 0.5) + (0 > samples[:, 0]) * (samples[:, 0] > -0.5)
    y_mask = (samples[:, 1] > 0.5) + (0 > samples[:, 1]) * (samples[:, 1] > -0.5)

    mask = x_mask ^ y_mask
    samples = samples[mask][:batch_size] + (shift if shift is not None else 0)
    return samples.float()

class GaussMix:
    def __init__(self, means, sigmas):
        mix = distributions.Categorical(torch.ones(means.size(0), device=means.device))
        comp = distributions.Independent(distributions.Normal(means, sigmas), 1)
        self.gmm = distributions.MixtureSameFamily(mix, comp)

        self._grad = torch.func.grad(lambda y: self.log_prob(y).sum())

    def sample(self, n_samples):
        return self.gmm.sample((n_samples,))

    def to(self, device='cpu'):
        self.gmm.to(device)

    def log_prob(self, x):
        return self.gmm.log_prob(x)
    
    def grad(self, x):
        return self._grad(x)
