"""
Dataset loading and forget/retain set management.
"""

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from typing import Tuple, List, Optional, Dict
import numpy as np
from pathlib import Path

from ..config import DataConfig


class ForgetRetainDataset:
    """
    Wrapper for datasets with forget/retain set splitting.
    
    Manages the division of a training dataset into:
    - forget_set: samples to be unlearned
    - retain_set: samples to keep
    - test_set: held-out evaluation set
    """
    
    def __init__(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        forget_indices: List[int],
        train_targets: Optional[List[int]] = None,
    ):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.forget_indices = set(forget_indices)
        
        # Get all indices
        all_indices = set(range(len(train_dataset)))
        self.retain_indices = list(all_indices - self.forget_indices)
        self.forget_indices = list(self.forget_indices)
        
        # Store targets for evaluation
        if train_targets is not None:
            self.train_targets = train_targets
        elif hasattr(train_dataset, 'targets'):
            self.train_targets = train_dataset.targets
        else:
            self.train_targets = [train_dataset[i][1] for i in range(len(train_dataset))]
        
        # Create subsets
        self.forget_set = Subset(train_dataset, self.forget_indices)
        self.retain_set = Subset(train_dataset, self.retain_indices)
        
        # Store forget labels for evaluation
        self.forget_labels = [self.train_targets[i] for i in self.forget_indices]
    
    def get_forget_loader(self, batch_size: int, shuffle: bool = True, **kwargs) -> DataLoader:
        """Get DataLoader for forget set."""
        return DataLoader(self.forget_set, batch_size=batch_size, shuffle=shuffle, **kwargs)
    
    def get_retain_loader(self, batch_size: int, shuffle: bool = True, **kwargs) -> DataLoader:
        """Get DataLoader for retain set."""
        return DataLoader(self.retain_set, batch_size=batch_size, shuffle=shuffle, **kwargs)
    
    def get_test_loader(self, batch_size: int, shuffle: bool = False, **kwargs) -> DataLoader:
        """Get DataLoader for test set."""
        return DataLoader(self.test_dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
    
    def get_full_train_loader(self, batch_size: int, shuffle: bool = True, **kwargs) -> DataLoader:
        """Get DataLoader for full training set (for initial training)."""
        return DataLoader(self.train_dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
    
    def get_stats(self) -> Dict:
        """Get dataset statistics."""
        return {
            'total_train': len(self.train_dataset),
            'forget_size': len(self.forget_indices),
            'retain_size': len(self.retain_indices),
            'test_size': len(self.test_dataset),
            'forget_ratio': len(self.forget_indices) / len(self.train_dataset),
        }


def get_transforms(dataset_name: str, train: bool = True, use_augmentation: bool = True) -> transforms.Compose:
    """Get data transforms for a dataset."""
    
    if dataset_name in ['cifar10', 'cifar100']:
        normalize = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )
        if train and use_augmentation:
            return transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
    
    elif dataset_name == 'svhn':
        normalize = transforms.Normalize(
            mean=[0.4377, 0.4438, 0.4728],
            std=[0.1980, 0.2010, 0.1970]
        )
        if train and use_augmentation:
            return transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_forget_indices_by_class(
    targets: List[int],
    forget_classes: List[int]
) -> List[int]:
    """Get indices of samples belonging to specific classes."""
    targets_array = np.array(targets)
    forget_mask = np.isin(targets_array, forget_classes)
    return np.where(forget_mask)[0].tolist()


def get_forget_indices_random(
    num_samples: int,
    forget_ratio: float,
    seed: int = 42
) -> List[int]:
    """Get random indices for forget set."""
    rng = np.random.RandomState(seed)
    num_forget = int(num_samples * forget_ratio)
    return rng.choice(num_samples, num_forget, replace=False).tolist()


def load_dataset(config: DataConfig, seed: int = 42) -> ForgetRetainDataset:
    """
    Load dataset and create forget/retain splits.
    
    Args:
        config: Data configuration
        seed: Random seed for reproducibility
        
    Returns:
        ForgetRetainDataset with forget/retain/test splits
    """
    data_dir = Path(config.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Get transforms
    train_transform = get_transforms(config.name, train=True, use_augmentation=config.use_augmentation)
    test_transform = get_transforms(config.name, train=False)
    
    # Load base datasets
    if config.name == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10(
            root=str(data_dir), train=True, download=True, transform=train_transform
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=str(data_dir), train=False, download=True, transform=test_transform
        )
        targets = train_dataset.targets
        
    elif config.name == 'cifar100':
        train_dataset = torchvision.datasets.CIFAR100(
            root=str(data_dir), train=True, download=True, transform=train_transform
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root=str(data_dir), train=False, download=True, transform=test_transform
        )
        targets = train_dataset.targets
        
    elif config.name == 'svhn':
        train_dataset = torchvision.datasets.SVHN(
            root=str(data_dir), split='train', download=True, transform=train_transform
        )
        test_dataset = torchvision.datasets.SVHN(
            root=str(data_dir), split='test', download=True, transform=test_transform
        )
        targets = train_dataset.labels.tolist()
        
    else:
        raise ValueError(f"Unknown dataset: {config.name}")
    
    # Determine forget indices based on forget type
    if config.forget_type == 'class':
        forget_indices = get_forget_indices_by_class(targets, config.forget_classes)
    elif config.forget_type == 'random':
        forget_indices = get_forget_indices_random(len(train_dataset), config.forget_ratio, seed)
    elif config.forget_type == 'user' and config.forget_indices is not None:
        forget_indices = config.forget_indices
    else:
        raise ValueError(f"Invalid forget configuration: type={config.forget_type}")
    
    return ForgetRetainDataset(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        forget_indices=forget_indices,
        train_targets=targets,
    )


class IndexedDataset(Dataset):
    """Dataset wrapper that returns sample indices along with data."""
    
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
    
    def __getitem__(self, index: int):
        data, target = self.dataset[index]
        return data, target, index
    
    def __len__(self):
        return len(self.dataset)
