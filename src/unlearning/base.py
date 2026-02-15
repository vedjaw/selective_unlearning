"""
Base class for unlearning methods.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import copy
from tqdm import tqdm


class BaseUnlearner(ABC):
    """
    Abstract base class for machine unlearning methods.
    
    All unlearning methods should inherit from this class and implement
    the `unlearn` method.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        verbose: bool = True,
    ):
        """
        Initialize unlearner.
        
        Args:
            model: The trained model to unlearn from
            device: Device to run computations on
            verbose: Whether to print progress
        """
        self.original_model = copy.deepcopy(model)
        self.model = model
        self.device = device
        self.verbose = verbose
        
    @abstractmethod
    def unlearn(
        self,
        forget_loader: DataLoader,
        retain_loader: DataLoader,
        **kwargs
    ) -> nn.Module:
        """
        Perform unlearning.
        
        Args:
            forget_loader: DataLoader for samples to forget
            retain_loader: DataLoader for samples to retain
            **kwargs: Additional method-specific arguments
            
        Returns:
            The unlearned model
        """
        pass
    
    def get_model(self) -> nn.Module:
        """Get the current model state."""
        return self.model
    
    def reset_model(self):
        """Reset model to original state."""
        self.model.load_state_dict(self.original_model.state_dict())
    
    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on a dataset.
        
        Args:
            loader: DataLoader to evaluate on
            
        Returns:
            Dictionary with accuracy and loss
        """
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()
        
        for inputs, targets in loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        return {
            'accuracy': 100. * correct / total,
            'loss': total_loss / total,
        }
    
    def _print(self, message: str):
        """Print message if verbose mode is on."""
        if self.verbose:
            print(message)


class TrainingMixin:
    """Mixin class providing common training utilities."""
    
    def _train_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: str,
        maximize: bool = False,
    ) -> float:
        """
        Train for one epoch.
        
        Args:
            model: Model to train
            loader: Training data loader
            optimizer: Optimizer
            criterion: Loss function
            device: Device
            maximize: If True, maximize loss (for unlearning)
            
        Returns:
            Average loss for the epoch
        """
        model.train()
        total_loss = 0.0
        
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            if maximize:
                loss = -loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * inputs.size(0)
        
        return total_loss / len(loader.dataset)
