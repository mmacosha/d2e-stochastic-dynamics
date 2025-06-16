import torch
from torch import nn

from sb.nn.cifar import CifarVAE, CifarGen, CifarCls
from sb.nn.mnist import MnistVAE, MnistCLS

import sys
sys.path.append("./external/stylegan3")

import dnnlib
import legacy


def dirichlet_reward(
        target_values: torch.Tensor, 
        alpha=-0.01, 
        max_reward: float = 5.0
    ):
    if target_values.shape[1] == 1:
        return target_values.squeeze(1)

    x = target_values / target_values.sum(dim=1, keepdim=True)
    alphas = torch.ones_like(x) * alpha
    
    return torch.minimum(x ** alphas, torch.as_tensor(max_reward)).prod(dim=1)


def max_reward(target_values):
    return target_values.max(dim=1).values


def sum_reward(target_values):
    return target_values.sum(dim=1)


REWARD_FUNCTIONS = {
    'dirichlet': dirichlet_reward,
    'max': max_reward,
    'sum': sum_reward
}

class CifarStyleGAN3(nn.Module):
    network_pkl = './reward_ckpt/stylegan2-cifar10-32x32.pkl'
    def __init__(self):
        super().__init__()
        with dnnlib.util.open_url(self.network_pkl) as f:
            self.G = legacy.load_network_pkl(f)['G_ema'] #.to(device)
            self.G.eval()
    
    def forward(self, latents):
        with torch.no_grad():
            c = torch.zeros((latents.shape[0], self.G.c_dim), device=latents.device)
            x = self.G(latents, c, noise_mode='const')
            x = (x + 1) / 2
        return x.clip(0, 1)


class ClsReward(nn.Module): 
    def __init__(self, generator, classifier, target_classes, reward_type='sum'):
        super().__init__()

        if reward_type not in REWARD_FUNCTIONS:
            raise ValueError(
                f"Unknown reward type: {reward_type}. "
                f"Chose from {list(REWARD_FUNCTIONS.keys())}."
            )
        self.reward_fn = REWARD_FUNCTIONS[reward_type]
        self.target_classes = target_classes
        self.generator = generator
        self.classifier = classifier
        self.eval()

    def forward(self, latents):
        with torch.no_grad():
            x_pred = self.generator(latents)
            logits = self.classifier(x_pred)
        
        probas = logits.softmax(dim=1)
        rewards = probas[:, self.target_classes]
        
        reward = self.reward_fn(rewards)
        return reward

    def state_dict(self):
        return {
            'generator': self.generator.state_dict(),
            'classifier': self.classifier.state_dict()
        }
    
    def load_state_dict(self, state_dict):
        self.generator.load_state_dict(state_dict['generator'])
        self.classifier.load_state_dict(state_dict['classifier'])

    @classmethod
    def build_reward(
        cls, generator_type: str, classifier_type: str, 
        target_classes, reward_type='sum'):        
        if generator_type == 'cifar_stylegan':
            generator = CifarStyleGAN3()
        
        elif generator_type == 'mnist_vae':
            generator = MnistVAE().decoder
            ckpt = torch.load('./rewards/mnist/mnist_reward.pth', 
                              map_location='cpu', weights_only=True)
            generator.load_state_dict(ckpt['decoder'])

        elif generator_type in {'cifar-gan-z50', 'cifar-gan-z100', 'cifar-gan-z256'}:
            ckpt_path = f'./rewards/cifar/{generator_type}.pt'
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
            generator = CifarGen(**ckpt['config'], inference=True)
            generator.load_state_dict(ckpt['state_dict'])
            generator.eval()
        
        else:
            raise NotImplemented(f"Unknown generator type: {generator_type}")

        if classifier_type == 'cifar_cls':
            classifier = CifarCls()
            ckpt = torch.load('./rewards/cifar/cifar_cls.pt', 
                              map_location='cpu', weights_only=True)
            classifier.load_state_dict(ckpt)
        
        elif classifier_type == 'mnist_cls':
            classifier = MnistCLS()
            ckpt = torch.load('./rewards/mnist/mnist_reward.pth', 
                              map_location='cpu', weights_only=True)
            generator.load_state_dict(ckpt['cls'])
        
        else:
            raise NotImplemented(f"Unknown classifier type: {classifier_type}")
        
        generator.eval()
        classifier.eval()
        
        return cls(generator, classifier, target_classes, reward_type)
