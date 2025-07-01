import torch
from torch import nn

from sb.nn.cifar import CifarGen, CifarCls
from sb.nn.mnist import MnistGen, MnistCLS

import sys
sys.path.append("./external/sg3")

import dnnlib, legacy


def max_reward(target_values):
    return target_values.max(dim=1).values


def sum_reward(target_values):
    return target_values.sum(dim=1)


REWARD_FUNCTIONS = {
    'max': max_reward,
    'sum': sum_reward
}

class StyleGanWrapper(nn.Module):
    def __init__(self, checkpoint: str):
        super().__init__()
        with dnnlib.util.open_url(checkpoint) as f:
            self.G = legacy.load_network_pkl(f)['G_ema']
    
    def forward(self, latents):
        c = torch.zeros(
            (latents.shape[0], self.G.c_dim), 
            device=latents.device
        )
        x = self.G(latents, c, noise_mode='const')
        return x


def _renormalize(image):
    if image.min() < 0:
        image = (image + 1) / 2
    return image


class ClsReward(nn.Module): 
    def __init__(self, generator, classifier, target_classes, reward_type='sum'):
        super().__init__()

        if reward_type not in REWARD_FUNCTIONS:
            raise ValueError(
                f"Unknown reward type: {reward_type}. "
                f"Chose from {list(REWARD_FUNCTIONS.keys())}."
            )
        self.reward_type = reward_type
        self.reward_fn = REWARD_FUNCTIONS[reward_type]
        self.target_classes = target_classes
        
        self.generator = generator
        self.classifier = classifier
        
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, latents):
        x_pred = self.generator(latents)
        x_pred = _renormalize(x_pred)
        logits = self.classifier(x_pred)
        all_probas = logits.softmax(dim=1)
        probas, classes = all_probas.max(dim=1)
        return {
            'logits': logits,
            'probas': probas,
            'all_probas': all_probas,
            'classes': classes,
            'images': x_pred    
        }
    
    def get_reward(self, probas):
        rewards = probas[:, self.target_classes]
        return self.reward_fn(rewards)

    def get_precision(self, classes):
        target_classes = torch.tensor(self.target_classes).view(-1, 1)
        precision = (classes == target_classes.to(classes.device))
        return precision.float().mean()

    def reward(self, latents):
        probas = self(latents)['all_probas']
        rewards = probas[:, self.target_classes]
        return self.reward_fn(rewards)
    
    def get_reward_and_precision(self, latents=None, outputs=None):
        if latents is None and outputs is None:
            raise ValueError(
                "Either latents or outputs must be provided."
            )
        if latents is not None:
            outputs = self(latents)

        reward = self.get_reward(outputs['all_probas'])
        precision = self.get_precision(outputs['classes'])

        return reward, precision
    
    def log_reward(self, latents):
        logits = self(latents)['logits']
        logits = logits - logits.max(dim=1, keepdim=True).values
        peaked_logits = logits[:, self.target_classes]
        
        if self.reward_type == "sum":
            return torch.logsumexp(peaked_logits, dim=1) \
                   - torch.logsumexp(logits, dim=1)

        elif self.reward_type == "max":
            peaked_logits = peaked_logits.max(dim=1).values
            return peaked_logits - torch.logsumexp(logits, dim=1)
        else:
            raise NotImplementedError(
                "efficient log_reward is not implemented for this reward type."
            )


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
        target_classes, reward_type='sum', reward_dir="./"):        
        if generator_type == 'cifar10-stylegan':
            generator = StyleGanWrapper(
                f'{reward_dir}/rewards/cifar10/stylegan2-cifar10-32x32.pkl'
            )
        
        elif generator_type in {"mnist-gan-z10", "mnist-gan-z50"}:
            ckpt = torch.load(
                f'{reward_dir}/rewards/mnist/{generator_type}.pt', 
                map_location='cpu', weights_only=True
            )
            generator = MnistGen(**ckpt["config"])
            generator.load_state_dict(ckpt["state_dict"])

        elif generator_type in {'cifar10-gan-z50', 'cifar10-gan-z100', 'cifar10-gan-z256'}:
            ckpt = torch.load(
                f'{reward_dir}/rewards/cifar10/{generator_type}.pt',
                map_location='cpu', weights_only=True
            )
            generator = CifarGen(**ckpt['config'], inference=True)
            generator.load_state_dict(ckpt['state_dict'])
        
        else:
            raise NotImplementedError(
                f"Unknown generator type: {generator_type}"
            )

        if classifier_type == 'cifar10-cls':
            classifier = CifarCls()
            ckpt = torch.load(
                f'{reward_dir}/rewards/cifar10/cifar10-cls.pt', 
                map_location='cpu', weights_only=True
            )
            classifier.load_state_dict(ckpt)
        
        elif classifier_type == 'mnist-cls':
            ckpt = torch.load(
                f'{reward_dir}/rewards/mnist/mnist-cls.pth', 
                map_location='cpu', weights_only=True
            )
            classifier = MnistCLS()
            classifier.load_state_dict(ckpt)

        else:
            raise NotImplemented(f"Unknown classifier type: {classifier_type}")
        
        generator.eval()
        classifier.eval()
        
        return cls(generator, classifier, target_classes, reward_type)
