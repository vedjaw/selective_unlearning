"""
Retrain-from-scratch baseline.

The gold standard for machine unlearning: train a new model on
retain-only data. This is the theoretical optimum that all 
unlearning methods should try to approximate.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any
from tqdm import tqdm

from .base import BaseUnlearner, TrainingMixin


class RetrainUnlearner(BaseUnlearner, TrainingMixin):
    """
    Retrain from scratch on retain data only.
    
    This is the gold standard baseline — the model never sees 
    the forget data at all. All unlearning methods should try
    to approximate this result.
    
    Downside: Extremely expensive (full retraining).
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        verbose: bool = True,
        epochs: int = 100,
        lr: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
    ):
        super().__init__(model, device, verbose)
        self.epochs = epochs
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
    
    def unlearn(
        self,
        forget_loader: DataLoader,
        retain_loader: DataLoader,
        **kwargs
    ) -> nn.Module:
        """
        'Unlearn' by retraining from scratch on retain data only.
        
        Ignores forget_loader entirely — trains fresh model on retain data.
        """
        self._print(f"Retrain from scratch for {self.epochs} epochs...")
        self._print(f"  (This is the gold standard — ignoring forget data entirely)")
        
        # Re-initialize model weights (train from scratch)
        def reset_weights(m):
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        
        self.model.apply(reset_weights)
        
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )
        
        # Learning rate scheduler (step decay at 50, 75)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[50, 75], gamma=0.1
        )
        
        criterion = nn.CrossEntropyLoss()
        
        best_acc = 0.0
        best_state = None
        
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, targets in retain_loader:
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
            
            scheduler.step()
            
            train_acc = 100.0 * correct / total
            
            # Track best model
            if train_acc > best_acc:
                best_acc = train_acc
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            
            if self.verbose and (epoch + 1) % max(1, self.epochs // 5) == 0:
                retain_metrics = self.evaluate(retain_loader)
                self._print(f"  Epoch {epoch+1}/{self.epochs}: "
                           f"Train Acc={train_acc:.2f}%, "
                           f"Retain Acc={retain_metrics['accuracy']:.2f}%, "
                           f"LR={optimizer.param_groups[0]['lr']:.6f}")
        
        # Load best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
        
        final_retain = self.evaluate(retain_loader)['accuracy']
        self._print(f"Retrain complete. Final retain acc: {final_retain:.2f}%")
        
        return self.model
