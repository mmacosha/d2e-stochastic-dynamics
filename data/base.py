from abc import ABC

class Dataset(ABC):
    def sample(self, x):
        raise NotImplementedError(
            "Sampling is impossible for that dataset"
        )

    def log_density(self, x):
        raise NotImplementedError(
            "Computing log density is impossible for that dataset"
        )

    def grad_log_density(self, x):
        raise NotImplementedError(
            "Computing grad of log density is impossible for that dataset"
        )


class DatasetRegistry:
    def __init__(self):
        self.dataset_generators = {}

    def add(self, func=None, name=None):
        def _decorator(func):
            self.dataset_generators[name or func.__name__] = func
            return func
        
        return _decorator if func is None else _decorator(func) 
    
    def __getitem__(self, name):
        return self.dataset_generators[name]
    
    @property
    def available_datasets(self):
        return list(self.dataset_generators)
