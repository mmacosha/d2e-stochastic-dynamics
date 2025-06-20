import inspect
import functools

import numpy as np
import matplotlib.pyplot as plt


class EMA:
    def __init__(self, model, decay, start_update=0):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self._register()
        
        self.curr_update = 0
        self.start_update = start_update

    def _register(self):
        if self.decay == 0.0:
            return

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def clear(self):
        if self.decay == 0.0:
            return
        self.shadow = {}
        self.backup = {}
        self.curr_update = 0
        self._register()

    def update(self):
        if self.decay == 0.0:
            return
        self.curr_update += 1
        if self.curr_update < self.start_update:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    assert name in self.shadow
                    self.shadow[name] = param.data.clone()
            return

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data \
                              + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply(self):
        if self.decay == 0.0:
            return
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        if self.decay == 0.0:
            return

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def plot_annotated_images(batch, classes_probas, n_col=8, figsize=(12, 6)):
    probas, classes = classes_probas
    f, ax = plt.subplots(batch.size(0) // n_col, n_col, figsize=figsize)
    for i in range(batch.size(0)):
        row, col = divmod(i, n_col)
        ax[row, col].imshow(batch[i].permute(1, 2, 0).cpu().numpy())
        ax[row, col].axis('off')
        ax[row, col].set_title(
            f"Class `{classes[i]}`; proba={probas[i]:.2f}"
        )
    plt.tight_layout()
    return f


def plot_trajectory(trajectory, timesteps, indices: list = None, 
                    title: str = None, limits=(-5, 5)):
    if indices is not None:
        trajectory = [trajectory[i] for i in indices]
        timesteps = [timesteps[i] for i in indices]

    figure, axes  = plt.subplots(1, len(trajectory), figsize=(4 * len(trajectory), 4))
    if title is not None:
        figure.suptitle(title)
    
    for i, sample in enumerate(trajectory):
        if type(timesteps[i]) == str:
            title = f'{timesteps[i]}' 
        else:
            title = f'timestep: {round(timesteps[i], 4)}'
        
        axes[i].set_title(title)
        axes[i].scatter(sample[:, 0], sample[:, 1], c='b', s=0.5)
        axes[i].set_xlim(*limits)
        axes[i].set_ylim(*limits)
    
    return figure


def plot_graph(graph):
    figure = plt.figure(figsize=(5, 3))
    plt.plot(graph)
    return figure


def build_config(init_func):
    @functools.wraps(init_func)
    def wrapper(self, *args, **kwargs):
        sig = inspect.signature(init_func)
        param_pairs = list(sig.parameters.items())[1:]
        
        config = {
            name: None if param.default is inspect.Parameter.empty else param.default
            for name, param in param_pairs
        }

        config.update(dict(zip(config.keys(), args)))
        config.update(kwargs)
        self.config = config

        init_func(self, *args, **kwargs)

    return wrapper


class Registry:
    def __init__(self):
        self._regirstry = {}

    def add(self, cls=None, name=None):
        def _decorator(func):
            self._regirstry[cls or cls.__name__] = cls
            return func
        
        return _decorator if cls is None else _decorator(cls) 
    
    def __getitem__(self, name):
        return self._regirstry[name]
    
    @property
    def available(self):
        return list(self._regirstry)