import inspect
import functools

import numpy as np
import matplotlib.pyplot as plt


class EMALoss:
    def __init__(self, alpha=0.1):
        self.ema = []
        self.loss = []
        self.alpha = alpha

    def update(self, loss_value):
        self.loss.append(loss_value)
        if not self.ema:
            self.ema.append(loss_value)
        else:
            self.ema.append(loss_value * self.alpha + self.ema[-1] * (1 - self.alpha))


class VarCriterion:
    def __init__(self, loss, threshold=0.001, measure_size: int = 10, max_iter=1000):
        self.loss = loss
        self.threshold = threshold
        self.measure_size = measure_size
        
        self.max_iter = max_iter
        self.curr_iter = 0

    def check(self):
        self.curr_iter += 1
        
        if self.curr_iter > self.max_iter:
            return False
        
        if len(self.loss) < self.measure_size:
            return True
        
        return np.var(self.loss[-self.measure_size:]) > self.threshold
    

def plot_trajectory(trajectory, timesteps, indices: None | list = None, 
                    title: str | None = None, limits=(-5, 5)):
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
        axes[i].scatter(sample[:, 0], sample[:, 1], c='b')
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