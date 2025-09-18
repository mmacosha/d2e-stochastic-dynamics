import math
import numpy as np

import torch
from torch import distributions 

from sb.nn.reward import ClsReward

from . import base
from . import datasets_fn_2d


registry = base.DatasetRegistry()


@registry.add(name="two_moons")
class TwoMoons(base.Dataset):
    def __init__(self, shift=None, noise=0.01):
        self.shift = shift
        self.noise = noise or 0
        self.fn = datasets_fn_2d.two_moons

    def sample(self, size: int):
        def _iterator():
            yield self.fn(size, self.shift, self.noise)
        return next(_iterator())


@registry.add(name="swiss_roll")
class SwissRoll(base.Dataset):
    def __init__(self, shift=None, noise=0.01):
        self.shift = shift
        self.noise = noise or 0
        self.fn = datasets_fn_2d.swiss_roll

    def sample(self, size: int):
        def _iterator():
            yield self.fn(size, self.shift, self.noise)
        return next(_iterator())


@registry.add(name="s_curve")
class SCurve(base.Dataset):
    def __init__(self, shift=None, noise=0.01):
        self.shift = shift
        self.noise = noise or 0
        self.fn = datasets_fn_2d.s_curve

    def sample(self, size: int):
        def _iterator():
            yield self.fn(size, self.noise, self.shift)
        return next(_iterator())


@registry.add(name="two_circles")
class TwoCircles(base.Dataset):
    def __init__(self, shift=None, noise=0.01):
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


@registry.add(name="mix_of_gaussians")
class GMM(base.Dataset):
    def __init__(self, r: int = 1, n_modes: int = 8, noise=0.01, device: str = 'cpu'):
        angles = torch.linspace(0, 2 * math.pi, n_modes + 1)[:-1]
        cos, sin = torch.cos(angles) * r, torch.sin(angles) * r
        
        means = torch.stack([cos, sin], dim=1).to(device)
        stds = torch.ones_like(means) * noise
        mode_probs = torch.ones(n_modes, device=device) / n_modes
        
        mix = distributions.Categorical(mode_probs)
        comp = distributions.Independent(
            base_distribution = distributions.Normal(means, stds),
            reinterpreted_batch_ndims=1
        )
        self.gmm = distributions.MixtureSameFamily(mix, comp)
        self.reward = None

    def sample(self, size):
        return self.gmm.sample((size, ))
    
    def log_density(self, x, *args, **kwargs):
        return self.gmm.log_prob(x)

    def get_mean_log_reward(self, x):
        return self.log_density(x).mean()


@registry.add(name="cross")
class Cross(base.Dataset):
    def __init__(self, r: int = 1, noise=0.01, device: str = 'cpu'):
        means = torch.tensor([
                [0, 0], [0.6, 0.36], [-0.6, 0.36], [-0.6, -0.36], [0.6, -0.36],
            ], device=device) * r
        stds = torch.ones_like(means) * noise
        mode_probs = torch.ones(means.size(0), device=device) / means.size(0)
        
        mix = distributions.Categorical(mode_probs)
        comp = distributions.Independent(
            base_distribution=distributions.Normal(means, stds),
            reinterpreted_batch_ndims=1
        )
        self.gmm = distributions.MixtureSameFamily(mix, comp)
        self.reward = None

    def sample(self, size):
        return self.gmm.sample((size, ))
    
    def log_density(self, x, *args, **kwargs):
        return self.gmm.log_prob(x)

    def get_mean_log_reward(self, x):
        return self.log_density(x).mean()


@registry.add(name="simple_gaussian")
class SimpleGaussian(base.Dataset):
    def __init__(self, mean=0, std=1, dim=2, device='cpu'):
        self.dim = dim
        self.mean = torch.as_tensor(mean, device=device)
        self.sigma = torch.as_tensor(std, device=device)
        self.dist = distributions.Normal(self.mean, self.sigma)
        self.reward = None

    def sample(self, size: int):
        shape = (size, *self.dim) if isinstance(self.dim, tuple) else (size, self.dim)
        return self.dist.sample(shape)

    def log_density(self, x):
        log_density = self.dist.log_prob(x)
        log_density = log_density.view(log_density.size(0), -1).sum(dim=1)
        return log_density

    def get_mean_log_reward(self, x):
        return self.log_density(x).mean()


def rejection_sampling(
    n_samples: int, 
    proposal: torch.distributions.Distribution, 
    target_log_prob_fn, 
    k: float
) -> torch.Tensor:
    """Rejection sampling. See Pattern Recognition and ML by Bishop Chapter 11.1"""
    z_0 = proposal.sample((n_samples * 10,))
    u_0 = torch.distributions.Uniform(0, k * torch.exp(proposal.log_prob(z_0))).sample().to(z_0)
    accept = torch.exp(target_log_prob_fn(z_0)) > u_0
    samples = z_0[accept]
    if samples.shape[0] >= n_samples:
        return samples[:n_samples]
    else:
        required_samples = n_samples - samples.shape[0]
        new_samples = rejection_sampling(
            required_samples, proposal, target_log_prob_fn, k
        )
        samples = torch.concat([samples, new_samples], dim=0)
        return samples


@registry.add(name="manywell")
class ManyWell(base.Dataset):
    """
    log p(x1, x2) = - x1^4 + 6*x1^2 + 1/2*x1 - 1/2*x2^2 + constant
    """
    def __init__(self, dim=2, device='cpu'):
        super().__init__()
        self.device = device
        self.reward = None

        self.data = torch.ones(dim, dtype=torch.float32).to(self.device)
        self.data_ndim = dim

        assert dim % 2 == 0
        self.n_wells = dim // 2

        # as rejection sampling proposal
        self.component_mix = torch.tensor([0.2, 0.8])
        self.means = torch.tensor([-1.7, 1.7])
        self.scales = torch.tensor([0.5, 0.5])

        self.Z_x1 = 11784.50927
        self.logZ_x2 = 0.5 * np.log(2 * np.pi)
        self.logZ_doublewell = np.log(self.Z_x1) + self.logZ_x2

    @property
    def bounds(self):
        return (-4.0, 4.0)

    @property
    def gt_logz(self):
        return self.n_wells * self.logZ_doublewell

    def energy(self, x):
        return -self.manywell_logprob(x)

    def doublewell_logprob(self, x):
        assert x.shape[1] == 2 and x.ndim == 2
        x1 = x[:, 0]
        x2 = x[:, 1]
        x1_term = 0.5 * x1 + 6 * x1.pow(2) - x1.pow(4)
        x2_term = -0.5 * x2.pow(2)
        return x1_term + x2_term

    def log_density(self, x):
        return torch.stack([
            self.doublewell_logprob(x[:, i * 2 : i * 2 + 2]) 
            for i in range(self.n_wells)], 
            dim=1
        ).sum(dim=1)

    def sample_first_dimension(self, batch_size):
        def target_log_prob(x):
            return -(x**4) + 6 * x**2 + 1 / 2 * x

        # Define proposal
        mix = torch.distributions.Categorical(self.component_mix)
        com = torch.distributions.Normal(self.means, self.scales)
        proposal = torch.distributions.MixtureSameFamily(mixture_distribution=mix, component_distribution=com)

        k = self.Z_x1 * 3
        samples = rejection_sampling(batch_size, proposal, target_log_prob, k)
        return samples

    def sample_doublewell(self, batch_size):
        x1 = self.sample_first_dimension(batch_size)
        x2 = torch.randn_like(x1)
        return torch.stack([x1, x2], dim=1)

    def sample(self, batch_size):
        return torch.cat([self.sample_doublewell(batch_size) for _ in range(self.n_wells)], dim=-1)

    def get_mean_log_reward(self, x):
        return self.log_density(x).mean()


@registry.add(name="funnel")
class Funnel(base.Dataset):
    def __init__(self, dim=1, device='cpu'):
        self.dim = dim
        self.device=device
        self.reward = None

    def sample(self, size: int):
        v = torch.randn(size, 1, device=self.device) * 3
        x = torch.randn(size, self.dim - 1, device=self.device) * torch.exp(v / 2)
        return torch.cat([x, v], dim=1)

    def log_density(self, x, anneal_beta=1.0):
        x, v = x[:, :-1], x[:, -1:]
        print(x.shape, v.shape)
        log_density = - 0.5 * (self.dim * (math.log(2 * math.pi) + v) + \
                               torch.sum(x.pow(2) / v.exp(), 1) + \
                               math.log(2 * math.pi * 3) + \
                               v.pow(2).sum(1) / 9 )
        return log_density

    def get_mean_log_reward(self, x):
        return self.log_density(x).mean()


@registry.add(name="cls_reward_dist")
class ClsRewardDist(base.Dataset):
    def __init__(self, generator_type: str, classifier_type: str, prior_dim, 
                 reward_dir: str, target_classes, reward_type: str, device: str):
        self.reward = ClsReward.build_reward(
            generator_type, classifier_type, target_classes, reward_type, reward_dir
        )
        self.reward.to(device).eval()
        self.prior = SimpleGaussian(0, 1, dim=prior_dim, device=device)

    def sample(self, *args):
        raise NotImplementedError

    def log_density(self, x, anneal_beta=1.0):
        return self.prior.log_density(x) + self.reward.log_reward(x, beta=anneal_beta)

    def get_mean_log_reward(self, x):
        return self.reward.log_reward(x).mean()


@registry.add(name="40gmm")
class FortyGaussianMixture:
    def __init__(self, dim, device):
        loc = (torch.rand((40, dim), device=device) - 0.5) * 2 * 40
        scale = torch.ones_like(loc).to(device)
        
        mixture_weights = torch.ones(loc.shape[0], device=loc.device)
        modes = distributions.Independent(distributions.Normal(loc, scale), 1)
        mix = distributions.Categorical(mixture_weights)
        
        self.nmode = 40
        self.means = loc
        self.covariance_matrices = [torch.diag(scale[i]) for i in range(self.nmode)]
        self.gmm = distributions.MixtureSameFamily(mix, modes)
        self.reward = None

    def log_density(self, x, anneal_beta=1.0):
        return self.gmm.log_prob(x)

    def sample(self, size: int):
        return self.gmm.sample((size,))

    def get_mean_log_reward(self, x):
        return self.log_density(x).mean()
