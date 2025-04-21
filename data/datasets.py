import torch
from torch import distributions 

from . import base
from . import datasets_fn_2d


registry = base.DatasetRegistry()


@registry.add(name="mix_of_gaussians")
class MixOfGaussians(base.Dataset):
    def __init__(self, means, sigmas, device: str = 'cpu'):
        means = torch.as_tensor(means, device=device)
        sigmas = torch.as_tensor(sigmas, device=device)

        mix = distributions.Categorical(torch.ones(means.size(0), device=device))
        comp = distributions.Independent(distributions.Normal(means, sigmas), 1)
        self.gmm = distributions.MixtureSameFamily(mix, comp)

        self.grad_fn = torch.func.grad(lambda y: self.gmm.log_prob(y).sum())

    def sample(self, size):
        return self.gmm.sample((size, ))
    
    def log_density(self, x):
        return self.gmm.log_prob(x)
    
    def grad_log_density(self, x):
        return self.grad_fn(x)


@registry.add(name="two_moons")
class TwoMoons(base.Dataset):
    def __init__(self, shift=None, noise=None):
        self.shift = shift
        self.noise = noise or 0
        self.fn = datasets_fn_2d.two_moons

    def sample(self, size: int):
        def _iterator():
            yield self.fn(size, self.shift, self.noise)
        return next(_iterator())


@registry.add(name="swiss_roll")
class SwissRoll(base.Dataset):
    def __init__(self, shift=None, noise=None):
        self.shift = shift
        self.noise = noise or 0
        self.fn = datasets_fn_2d.swiss_roll

    def sample(self, size: int):
        def _iterator():
            yield self.fn(size, self.shift, self.noise)
        return next(_iterator())


@registry.add(name="s_curve")
class SCurve(base.Dataset):
    def __init__(self, shift=None, noise=None):
        self.shift = shift
        self.noise = noise or 0
        self.fn = datasets_fn_2d.s_curve

    def sample(self, size: int):
        def _iterator():
            yield self.fn(size, self.shift, self.noise)
        return next(_iterator())


@registry.add(name="two_circles")
class TwoCircles(base.Dataset):
    def __init__(self, shift=None, noise=None):
        self.shift = shift
        self.noise = noise or 0
        self.fn = datasets_fn_2d.two_circles

    def sample(self, size: int):
        def _iterator():
            yield self.fn(size, self.shift, self.noise)
        return next(_iterator())


@registry.add(name="two_circles_custom")
class TwoCirclesCustom(base.Dataset):
    def __init__(self, r1, r2, noise=None):
        self.r1 = r1
        self.r2 = r2
        self.noise = noise or 0
        self.fn = datasets_fn_2d.two_circles

    def sample(self, size: int):
        def _iterator():
            yield self.fn(size, self.r1, self.r2, noise=self.noise)
        return next(_iterator())


@registry.add(name="checkboard")
class Checkboard(base.Dataset):
    def __init__(self, shift=None):
        self.shift = shift
        self.fn = datasets_fn_2d.two_circles

    def sample(self, size: int):
        def _iterator():
            yield self.fn(size, self.shift)
        return next(_iterator())
