import torch
import torchvision 
from torch import nn
from torchvision.transforms import v2

from transformers import ViTForImageClassification, ViTImageProcessor
from efficientvit.ae_model_zoo import DCAE_HF

import sys, os
sys.path.append(os.path.abspath('external/sg3'))
sys.path.append(os.path.abspath('external/sngan'))
sys.path.append(os.path.abspath('external/cifar10_cls'))

import dnnlib, legacy
import cifar10_models.vgg as vgg
import cifar10_models.resnet as resnet
import models as sngan_models

from .utils import rgb_to_3ch_grey, AttrDict


# -------------------------------- Generator Wrappers -------------------------------- #


class CIFAR10SNGANWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        config = AttrDict(bottom_width=4, gf_dim=256, latent_dim=128)
        self.model = sngan_models.Generator(config)
        self.model.load_state_dict(torch.load('external/sngan/sngan_cifar10.pth'))

    def forward(self, x):
        return self.model(x)


class DCAEWrapper(nn.Module):
    def __init__(self, model: str):
        super().__init__()
        self.model = DCAE_HF.from_pretrained(model)

    def forward(self, x):
        x = self.model.decode(x)
        return x


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


# -------------------------------- Classifier Wrappers ------------------------------- #


class ViTClsWrapper(nn.Module):
    def __init__(self, name, to_grey=False):
        super().__init__()
        self.to_grey = to_grey

        processor = ViTImageProcessor.from_pretrained(name)
        self.transform = v2.Compose([
            v2.Lambda(lambda x: rgb_to_3ch_grey(x) if to_grey else x),
            v2.Resize(
                tuple(processor.size.values()), 
                interpolation=v2.InterpolationMode.BICUBIC
            ),
            v2.Lambda(lambda x: x * 0.5 + 0.5),  # Rescale to [0, 1]
            v2.Normalize(processor.image_mean, processor.image_std),
        ])
        self.model = ViTForImageClassification.from_pretrained(name)

    def forward(self, x):
        x = self.transform(x)
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
        self.transform = v2.Compose([
            v2.Lambda(lambda x: x * 0.5 + 0.5),  # Rescale to [0, 1]
            v2.Normalize([0.4914, 0.4822, 0.4465], [0.2471, 0.2435, 0.2616])
        ])
        self.model = self.model_registry[name](pretrained=True)

    def forward(self, x):
        x = self.transform(x)
        return self.model(x)


class ImageNetClsWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.transform = v2.Compose([
            v2.Resize(256, interpolation=v2.InterpolationMode.BICUBIC),
            v2.CenterCrop(224),
            v2.Lambda(lambda x: x * 0.5 + 0.5),
            v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.model = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1
        )

    def forward(self, x):
        x = self.transform(x)
        return self.model(x)
