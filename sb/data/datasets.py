import math

import torch
from torch import distributions 

from sb.nn.reward import ClsReward

from . import base
from . import datasets_fn_2d


registry = base.DatasetRegistry()


@registry.add(name="mix_of_gaussians")
class MixOfGaussians(base.Dataset):
    def __init__(self, means, sigmas, device: str = 'cpu'):
        means = torch.as_tensor(means, device=device).float()
        sigmas = torch.as_tensor(sigmas, device=device).float()

        mix = distributions.Categorical(torch.ones(means.size(0), device=device))
        comp = distributions.Independent(distributions.Normal(means, sigmas), 1)
        self.gmm = distributions.MixtureSameFamily(mix, comp)

        self.grad_fn = torch.func.grad(lambda y: self.gmm.log_prob(y).sum())
        self.reward = lambda x: torch.as_tensor(1.0)

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


@registry.add(name="simple_gaussian")
class SimpleGaussian(base.Dataset):
    def __init__(self, mean=0, std=1, dim=2, device='cpu'):
        self.dim = dim
        self.mean = torch.as_tensor(mean, device=device)
        self.sigma = torch.as_tensor(std, device=device)
        self.dist = distributions.Normal(self.mean, self.sigma)

    def sample(self, size: int):
        return self.dist.sample((size, self.dim))

    def log_density(self, x):
        return self.dist.log_prob(x)


@registry.add(name="cls_reward_dist")
class ClsRewardDist(base.Dataset):
    def __init__(self, generator_type: str, classifier_type: str, prior_dim: int, 
                 target_classes, reward_type: str, device: str):
        self.reward = ClsReward.build_reward(
            generator_type, classifier_type, target_classes, reward_type
        )
        self.reward.to(device).eval()
        self.prior = SimpleGaussian(0, 1, dim=prior_dim, device=device)

    def sample(self, *args):
        raise NotImplementedError

    def log_density(self, x):
        return self.prior.log_density(x).sum(1) + self.reward.log_reward(x)

    def __call__(self, x):
        return self.log_density(x)

    def grad_log_density(self, x):
        x_ = x.clone().detach().requires_grad_(True)

        log_density = self.log_density(x_)
        log_density.sum().backward()

        return x_.grad
