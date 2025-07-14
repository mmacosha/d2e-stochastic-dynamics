import torch
from torch import nn
from torchvision.transforms import v2
from torchvision import models

from sb.nn.cifar  import CifarGen, CifarCls
from sb.nn.mnist  import MnistGen, MnistCLS
from sb.nn.celeba import CelebaCls

from transformers import ViTForImageClassification, ViTImageProcessor
from efficientvit.ae_model_zoo import DCAE_HF

import sys
sys.path.append("./external/sg3")
sys.path.append("./external/cifar10_cls")
sys.path.append("./external/sngan")

import dnnlib, legacy
import cifar10_models.vgg as vgg
import cifar10_models.resnet as resnet
from models.sngan_cifar10 import Generator


def _renormalize(image):
    if image.min() < 0:
        image = (image + 1) / 2
    return image

def max_reward(target_values):
    return target_values.max(dim=1).values


def sum_reward(target_values):
    return target_values.sum(dim=1)


REWARD_FUNCTIONS = {
    'max': max_reward,
    'sum': sum_reward
}


def rgb_to_3ch_grey(image):
    scale = torch.tensor([0.2989, 0.5870, 0.1140], device=image.device)
    grey_image = (image * scale[:, None, None]).sum(dim=1, keepdim=True)
    return grey_image.repeat(1, 3, 1, 1)


class StyleGanWrapper(nn.Module):
    def __init__(self, checkpoint: str):
        super().__init__()
        with dnnlib.util.open_url(checkpoint) as f:
            self.G = legacy.load_network_pkl(f)['G_ema']
    
    @staticmethod
    def get_checkpoint_from_name(name, base_dir):
        name2checkpoint = {
            "sg-celeba-256":        "rewards/sg/stylegan2-celebahq-256x256.pkl",
            "sg-ffhq-1024":         "rewards/sg/stylegan3-r-ffhqu-1024x1024.pkl",
            "sg-metafaces-1024":    "rewards/sg/stylegan3-r-metfaces-1024x1024.pkl",
            "sg-cifar10-32":        "rewards/sg/stylegan2-cifar10-32x32.pkl",
            "sg-ffhq-256":          "rewards/sg/stylegan3-r-ffhqu-256x256.pkl",
        }
        return f"{base_dir}/{name2checkpoint[name]}"

    def forward(self, latents):
        c = torch.zeros(
            (latents.shape[0], self.G.c_dim), 
            device=latents.device
        )
        x = self.G(latents, c, noise_mode='const')
        return x


class TransformersModelWrapper(nn.Module):
    def __init__(self, model, name, shape, to_grey=False):
        super().__init__()
        self.shape = shape
        self.to_grey = to_grey
        self.model = model.from_pretrained(name)
        
        processor = ViTImageProcessor.from_pretrained(name)
        self.normalize = v2.Normalize(processor.image_mean, processor.image_std)

    def forward(self, x):
        if self.to_grey:
            x = rgb_to_3ch_grey(x)
        
        x = nn.functional.interpolate(x, self.shape)
        x = self.normalize(x)
        return self.model(x).logits


class CIFAR10ClsWrapper(nn.Module):
    model_registry = {
        "cifar10-vgg13": vgg.vgg13_bn,
        "cifar10-vgg19": vgg.vgg19_bn,
        "cifar10-resnet18": resnet.resnet18,
        "cifar10-resnet50": resnet.resnet50
    }
    def __init__(self, name):
        super().__init__()

        # self.register_buffer('mean', torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1))
        # self.register_buffer('std', torch.tensor([0.2471, 0.2435, 0.2616]).view(3, 1, 1))
        
        mean, std = [0.4914, 0.4822, 0.4465], [0.2471, 0.2435, 0.2616]
        self.normalize = v2.Normalize(mean, std)
        self.model = self.model_registry[name](pretrained=True)

    def forward(self, x):
        # x = (x - self.mean) / self.std
        x = self.normalize(x)
        return self.model(x)


class DCAEWrapper(nn.Module):
    def __init__(self, model: str):
        super().__init__()
        self.model = DCAE_HF.from_pretrained(model).decode

    def forward(self, x):
        x = self.model(x)
        return x * 0.5 + 0.5


class ClsReward(nn.Module): 
    def __init__(self, generator, classifier, classifier_type, target_classes, reward_type='sum'):
        super().__init__()

        if reward_type not in REWARD_FUNCTIONS:
            raise ValueError(
                f"Unknown reward type: {reward_type}. "
                f"Chose from {list(REWARD_FUNCTIONS.keys())}."
            )
        assert classifier_type != "celeba-cls-256", "This cls is not supported."
        self.classifier_type = classifier_type
        self.reward_type = reward_type
        self.reward_fn = REWARD_FUNCTIONS[reward_type]
        self.target_classes = target_classes
        
        self.generator = generator
        self.classifier = classifier
        
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, latents, beta=1.0):
        x_pred = self.generator(latents)
        x_pred = _renormalize(x_pred)
        logits = self.classifier(x_pred) / beta
        
        if self.classifier_type in {"celeba-cls-64", "celeba-cls-32"}:
            all_probas = nn.functional.sigmoid(logits)
            probas, classes = all_probas.max(dim=1)
        else:
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
        return precision.float().sum(0).mean()
    
    def get_target_class_images(self, output, size):
        classes = output['classes']
        probas = output['probas']
        images = output['images']

        target_classes = torch.tensor(self.target_classes).view(-1, 1)
        target_classes = target_classes.to(classes.device)
        idx = torch.nonzero((classes == target_classes), as_tuple=False)[:, -1]
        return images[idx][:size], probas[idx][:size], classes[idx][:size]

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
    
    def log_reward(self, latents, beta=1.0):
        logits = self(latents, beta=beta)['logits']

        if self.classifier_type in {"celeba-cls-64", "celeba-cls-32"}:
            assert self.reward_type == "sum", "Only sum reward is supported for Celeba."
            log_probas = nn.functional.softplus(-logits, dim=1)
            log_reward = torch.logsumexp(log_probas[:, self.target_classes], dim=1)
            return log_reward
            
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

        if generator_type in {"sg-celeba-256", "sg-ffhq-1024", "sg-metafaces-1024",
                              "sg-cifar10-32", "sg-ffhq-256"}:
            checkpoint = StyleGanWrapper.get_checkpoint_from_name(
                generator_type, reward_dir
            )
            generator = StyleGanWrapper(checkpoint)
        
        elif generator_type in {"mnist-gan-z10", "mnist-gan-z50"}:
            ckpt = torch.load(
                f'{reward_dir}/rewards/mnist/{generator_type}.pt', 
                map_location='cpu', weights_only=True
            )
            generator = MnistGen(**ckpt["config"])
            generator.load_state_dict(ckpt["state_dict"])

        if generator_type == "dc-ae-imagenet":
            generator = DCAEWrapper("mit-han-lab/dc-ae-f32c32-in-1.0")

        elif generator_type == "cifar10-sngan":
            class AttrDict(dict):
                def __getattr__(self, name):
                    return self[name]
            args = AttrDict(bottom_width=4, gf_dim=256, latent_dim=128)
            generator = Generator(args)
            generator.load_state_dict(torch.load('external/sngan/sngan_cifar10.pth'))

        elif generator_type in {'cifar10-gan-z50', 'cifar10-gan-z100', 
                                'cifar10-gan-z256'}:
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

        elif classifier_type in {"cifar10-vgg13", "cifar10-vgg19", 
                                 "cifar10-resnet18", "cifar10-resnet50"}:
            classifier = CIFAR10ClsWrapper(classifier_type)    
        
        elif classifier_type == "celeba-cls-64":
            checkpoint_path = f"{reward_dir}/rewards/celeba/celeba-cls-64x64.pth"
            ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            classifier = CelebaCls(**ckpt['config'])
            classifier.load_state_dict(ckpt['state_dict'])

        elif classifier_type == "imagenet":
            classifier = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        elif classifier_type == "faces-gender":
            classifier = TransformersModelWrapper(
                ViTForImageClassification,
                "rizvandwiki/gender-classification",
                shape=224,
            )
        elif classifier_type == "faces-emotions":
            classifier = TransformersModelWrapper(
                ViTForImageClassification,
                "trpakov/vit-face-expression",
                shape=224,
                to_grey=True
            )
        else:
            raise NotImplementedError(f"Unknown classifier type: {classifier_type}")
        
        generator.eval()
        classifier.eval()
        
        return cls(generator, classifier, classifier_type, target_classes, reward_type)
