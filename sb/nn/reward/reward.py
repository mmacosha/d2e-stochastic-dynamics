import torch
from torch import nn

from sb.nn.cifar  import CifarGen, CifarCls
from sb.nn.mnist  import MnistGen, MnistCLS
from sb.nn.celeba import CelebaCls

from .utils import REWARD_FUNCTIONS
from .reward_wrappers import (
    StyleGanWrapper, DCAEWrapper, 
    CIFAR10ClsWrapper, ViTClsWrapper,
    ImageNetClsWrapper, CIFAR10SNGANWrapper,
    CelebAClsWrapper
)


class ClsReward(nn.Module): 
    def __init__(self, generator, classifier, 
                 classifier_type, target_classes, 
                 reward_type='sum'):
        super().__init__()

        if reward_type not in REWARD_FUNCTIONS:
            raise ValueError(
                f"Unknown reward type: {reward_type}. "
                f"Chose from {list(REWARD_FUNCTIONS.keys())}."
            )
        # assert classifier_type != "celeba-cls-256", "This cls is not supported."
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
        logits = self.classifier(x_pred) / beta
        
        if self.classifier_type.startswith("celeba"):
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
            'images': torch.clamp(x_pred * 0.5 + 0.5, 0, 1),
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

        elif generator_type == "dc-ae-imagenet":
            generator = DCAEWrapper("mit-han-lab/dc-ae-f32c32-in-1.0")

        elif generator_type == "cifar10-sngan":
            generator = CIFAR10SNGANWrapper()

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
            assert 0, "Normalisation is not handled in this classifier."
            classifier = CifarCls()
            ckpt = torch.load(
                f'{reward_dir}/rewards/cifar10/cifar10-cls.pt', 
                map_location='cpu', weights_only=True
            )
            classifier.load_state_dict(ckpt)
        
        elif classifier_type == 'mnist-cls':
            assert 0, "Normalisation is not handled in this classifier."
            ckpt = torch.load(
                f'{reward_dir}/rewards/mnist/mnist-cls.pth', 
                map_location='cpu', weights_only=True
            )
            classifier = MnistCLS()
            classifier.load_state_dict(ckpt)

        elif classifier_type in {"cifar10-vgg13", "cifar10-vgg19", 
                                 "cifar10-resnet18", "cifar10-resnet50"}:
            classifier = CIFAR10ClsWrapper(classifier_type)    

        elif classifier_type == "celeba-cls-256":
            classifier = CelebAClsWrapper()

        elif classifier_type == "celeba-cls-64":
            checkpoint_path = f"{reward_dir}/rewards/celeba/celeba-cls-64x64.pth"
            ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            classifier = CelebaCls(**ckpt['config'])
            classifier.load_state_dict(ckpt['state_dict'])

        elif classifier_type == "imagenet":
            classifier = ImageNetClsWrapper()

        elif classifier_type == "faces-gender":
            classifier = ViTClsWrapper(
                "rizvandwiki/gender-classification",
            )
        elif classifier_type == "faces-emotions":
            classifier = ViTClsWrapper(
                "trpakov/vit-face-expression",
                to_grey=True
            )
        else:
            raise NotImplementedError(f"Unknown classifier type: {classifier_type}")
        
        generator.eval()
        classifier.eval()
        
        return cls(generator, classifier, classifier_type, target_classes, reward_type)
