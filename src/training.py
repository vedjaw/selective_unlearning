"""
Training utilities for models before unlearning.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple, Callable
from pathlib import Path
import time
from tqdm import tqdm


class Trainer:
    """Standard training loop for neural network classifiers."""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        verbose: bool = True,
    ):
        self.model = model
        self.device = device
        self.verbose = verbose
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        lr: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        lr_scheduler: str = 'step',
        lr_decay_epochs: Tuple[int, ...] = (50, 75),
        lr_decay_factor: float = 0.1,
        checkpoint_dir: Optional[str] = None,
        checkpoint_frequency: int = 10,
    ) -> nn.Module:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            epochs: Number of training epochs
            lr: Initial learning rate
            momentum: SGD momentum
            weight_decay: L2 regularization weight
            lr_scheduler: Type of LR schedule ('step', 'cosine', 'none')
            lr_decay_epochs: Epochs at which to decay LR (for step scheduler)
            lr_decay_factor: Factor to decay LR by
            checkpoint_dir: Directory to save checkpoints
            checkpoint_frequency: Save checkpoint every N epochs
            
        Returns:
            Trained model
        """
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
        
        # Setup learning rate scheduler
        if lr_scheduler == 'step':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=list(lr_decay_epochs), gamma=lr_decay_factor
            )
        elif lr_scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs
            )
        else:
            scheduler = None
        
        criterion = nn.CrossEntropyLoss()
        
        if checkpoint_dir:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            # Training phase
            train_loss, train_acc = self._train_epoch(
                train_loader, optimizer, criterion
            )
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            
            # Validation phase
            if val_loader is not None:
                val_loss, val_acc = self._evaluate(val_loader, criterion)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    if checkpoint_dir:
                        self._save_checkpoint(
                            Path(checkpoint_dir) / 'best_model.pt',
                            epoch, optimizer, val_acc
                        )
            else:
                val_loss, val_acc = 0.0, 0.0
            
            # Update learning rate
            if scheduler is not None:
                scheduler.step()
            
            # Logging
            if self.verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1}/{epochs}: "
                      f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
                      f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%, "
                      f"LR={current_lr:.6f}")
            
            # Periodic checkpoint
            if checkpoint_dir and (epoch + 1) % checkpoint_frequency == 0:
                self._save_checkpoint(
                    Path(checkpoint_dir) / f'checkpoint_epoch_{epoch+1}.pt',
                    epoch, optimizer, val_acc
                )
        
        if self.verbose:
            print(f"Training complete. Best validation accuracy: {best_val_acc:.2f}%")
        
        return self.model
    
    def _train_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        return total_loss / total, 100. * correct / total
    
    @torch.no_grad()
    def _evaluate(
        self,
        loader: DataLoader,
        criterion: nn.Module,
    ) -> Tuple[float, float]:
        """Evaluate model on a dataset."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        return total_loss / total, 100. * correct / total
    
    def _save_checkpoint(
        self,
        path: Path,
        epoch: int,
        optimizer: torch.optim.Optimizer,
        accuracy: float,
    ):
        """Save training checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'accuracy': accuracy,
            'history': self.history,
        }, path)
    
    def load_checkpoint(self, path: str, load_optimizer: bool = False):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'history' in checkpoint:
            self.history = checkpoint['history']
        return checkpoint
