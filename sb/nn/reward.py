import torch
from torch import nn

from sb.nn.cifar import CifarVAE, CifarGen, CifarCls
from sb.nn.mnist import MnistVAE, MnistCLS


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

    def forward(self, latents):
        with torch.no_grad():
            x_pred = self.generator(latents)
            logits = self.classifier(x_pred)
        
        probas = logits.softmax(dim=1)[:, self.target_classes]
        return self.reward_fn(probas)

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
        target_classes, reward_type='sum', checkpoint_path=None):
        if generator_type == 'cifar_vae':
            generator = CifarVAE().decoder
        elif generator_type == 'cifar_gen':
            generator = CifarGen()
        elif generator_type == 'mnist_vae':
            generator = MnistVAE().decoder
        else:
            raise NotImplemented(f"Unknown generator type: {generator_type}")

        if classifier_type == 'cifar_cls':
            classifier = CifarCls()
        elif classifier_type == 'mnist_cls':
            classifier = MnistCLS()
        else:
            raise NotImplemented(f"Unknown classifier type: {classifier_type}")
        
        reward = cls(generator, classifier, target_classes, reward_type)
        if checkpoint_path is not None:
            ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            reward.load_state_dict(ckpt)

        return reward
