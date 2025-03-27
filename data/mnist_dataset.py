import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np
import random

class ClassFilteredMNIST(Dataset):
    """MNIST dataset that can sample from specific classes only."""
    
    def __init__(self, root='./data/datasets', train=True, transform=None, 
                 target_transform=None, download=True, classes=None):
        """
        Args:
            root (string): Directory to store the dataset.
            train (bool, optional): If True, use the training set, else the test set.
            transform (callable, optional): Transform to apply to the images.
            target_transform (callable, optional): Transform to apply to the targets.
            download (bool, optional): If True, downloads the dataset from the internet.
            classes (list, optional): List of class indices to include (0-9).
                                      If None, all classes are included.
        """
        self.mnist = datasets.MNIST(root=root, train=train, transform=transform,
                                   target_transform=target_transform, download=download)
        
        # Filter by classes if specified
        if classes is not None:
            self.classes = set(classes)
            self.indices = [i for i, (_, label) in enumerate(self.mnist) 
                            if label in self.classes]
        else:
            self.classes = set(range(10))  # All MNIST classes (0-9)
            self.indices = list(range(len(self.mnist)))
        
        # Create a dictionary mapping class to its indices for efficient sampling
        self.class_indices = {}
        for idx in self.indices:
            _, label = self.mnist[idx]
            if label not in self.class_indices:
                self.class_indices[label] = []
            self.class_indices[label].append(idx)
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        return self.mnist[real_idx]
    
    def sample(self, n_samples, classes=None):
        """
        Sample n_samples randomly from specified classes.
        
        Args:
            n_samples (int): Number of samples to return
            classes (list, optional): Specific classes to sample from.
                                     If None, uses the classes specified in constructor.
        
        Returns:
            tuple: (images, labels)
        """
        # Determine which classes to sample from
        sample_classes = self.classes
        if classes is not None:
            sample_classes = set(classes).intersection(self.classes)
            if not sample_classes:
                raise ValueError("No valid classes to sample from")
        
        # Get available indices for these classes
        available_indices = []
        for cls in sample_classes:
            available_indices.extend(self.class_indices.get(cls, []))
        
        if not available_indices:
            raise ValueError("No samples found for the specified classes")
        
        # Sample randomly
        sampled_indices = random.choices(available_indices, k=n_samples)
        
        # Get the images and labels
        images = []
        labels = []
        for idx in sampled_indices:
            image, label = self.mnist[idx]
            images.append(image)
            labels.append(label)
        
        # Convert to tensors
        if isinstance(images[0], torch.Tensor):
            images = torch.stack(images)
            labels = torch.tensor(labels)
        
        return images, labels
    
    def sample_balanced(self, n_samples_per_class, classes=None):
        """
        Sample an equal number of examples from each specified class.
        
        Args:
            n_samples_per_class (int): Number of samples per class
            classes (list, optional): Specific classes to sample from.
                                     If None, uses the classes specified in constructor.
        
        Returns:
            tuple: (images, labels)
        """
        # Determine which classes to sample from
        sample_classes = list(self.classes)
        if classes is not None:
            sample_classes = [cls for cls in classes if cls in self.classes]
            if not sample_classes:
                raise ValueError("No valid classes to sample from")
        
        images = []
        labels = []
        
        # Sample from each class
        for cls in sample_classes:
            cls_indices = self.class_indices.get(cls, [])
            if not cls_indices:
                continue
                
            # Sample with replacement if needed
            sampled_indices = random.choices(cls_indices, k=n_samples_per_class)
            
            for idx in sampled_indices:
                image, label = self.mnist[idx]
                images.append(image)
                labels.append(label)
        
        # Convert to tensors
        if isinstance(images[0], torch.Tensor):
            images = torch.stack(images)
            labels = torch.tensor(labels)
            
        return images, labels
