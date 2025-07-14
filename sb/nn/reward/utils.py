import torch


def max_reward(target_values):
    return target_values.max(dim=1).values


def sum_reward(target_values):
    return target_values.sum(dim=1)


REWARD_FUNCTIONS = {
    'max': max_reward,
    'sum': sum_reward
}

def renormalize(image):
    if image.min() < 0:
        image = (image + 1) / 2
    return image


def rgb_to_3ch_grey(image):
    scale = torch.tensor([0.2989, 0.5870, 0.1140], device=image.device)
    grey_image = (image * scale[:, None, None]).sum(dim=1, keepdim=True)
    return grey_image.repeat(1, 3, 1, 1)


class AttrDict(dict):
    def __getattr__(self, name):
        return self[name]
