import torch
from torch.utils.data import DataLoader, Dataset
from itertools import cycle


class InfiniteDataLoader:
    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool = True, num_workers: int = 0):
        """
        Wraps a PyTorch DataLoader to provide infinite data sampling.
        
        Args:
            dataset (Dataset): The dataset to load data from.
            batch_size (int): The number of samples per batch.
            shuffle (bool): Whether to shuffle the dataset after each epoch.
            num_workers (int): Number of worker processes for data loading.
        """
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        self.iterator = cycle(self.dataloader)
    
    def sample(self, batch_size: int):
        """Returns a batch of data infinitely."""
        batch = next(self.iterator)
        return batch[0][:batch_size]

    def __iter__(self):
        """Allows using the instance as an iterable."""
        return self
    
    def __next__(self):
        return self.sample()
    
    def __len__(self):
        """Returns the number of batches per epoch (not infinite)."""
        return len(self.dataloader)
