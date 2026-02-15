"""
Fine-tuning and Gradient Ascent baselines for unlearning.

Includes:
- FineTuneUnlearner: Simple fine-tuning on retain set
- EntropyMaxUnlearner: Entropy maximization for true forgetting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Any
from tqdm import tqdm

from .base import BaseUnlearner, TrainingMixin


class FineTuneUnlearner(BaseUnlearner, TrainingMixin):
    """
    Fine-tuning baseline for machine unlearning.
    
    Simply fine-tunes the model on the retain set without the forget samples.
    This is a simple baseline that often works reasonably well.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        verbose: bool = True,
        epochs: int = 10,
        lr: float = 0.01,
        weight_decay: float = 5e-4,
    ):
        super().__init__(model, device, verbose)
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
    
    def unlearn(
        self,
        forget_loader: DataLoader,
        retain_loader: DataLoader,
        **kwargs
    ) -> nn.Module:
        """
        Unlearn by fine-tuning on retain set only.
        
        Args:
            forget_loader: DataLoader for samples to forget (unused)
            retain_loader: DataLoader for samples to retain
            
        Returns:
            The unlearned model
        """
        self._print(f"Fine-tuning on retain set for {self.epochs} epochs...")
        
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=0.9,
            weight_decay=self.weight_decay
        )
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(self.epochs):
            loss = self._train_epoch(
                self.model, retain_loader, optimizer, criterion, self.device
            )
            
            if self.verbose and (epoch + 1) % max(1, self.epochs // 5) == 0:
                metrics = self.evaluate(retain_loader)
                self._print(f"  Epoch {epoch+1}/{self.epochs}: "
                           f"Loss={loss:.4f}, Retain Acc={metrics['accuracy']:.2f}%")
        
        return self.model


class GradientAscentUnlearner(BaseUnlearner, TrainingMixin):
    """
    Entropy Maximization Unlearning (improved Gradient Ascent).
    
    Key insight: True forgetting means RANDOM predictions, not WRONG predictions.
    - Old approach: Gradient ascent → model learns to predict WRONG
    - New approach: Entropy maximization → model predicts UNIFORMLY (random)
    
    This achieves ~10% accuracy (random for 10 classes) instead of 0%.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        verbose: bool = True,
        epochs: int = 10,
        lr: float = 0.0005,  # Lower LR for stability
        weight_decay: float = 5e-4,
        retain_weight: float = 5.0,  # Strong retain weight
        entropy_weight: float = 1.0,  # Weight for entropy term
    ):
        super().__init__(model, device, verbose)
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.retain_weight = retain_weight
        self.entropy_weight = entropy_weight
    
    def _entropy_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy of predictions.
        
        Maximizing entropy → uniform distribution → random predictions.
        Returns negative entropy (to maximize via gradient descent).
        """
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)
        entropy = -(probs * log_probs).sum(dim=1).mean()
        return -entropy  # Negative because we want to MAXIMIZE entropy
    
    def unlearn(
        self,
        forget_loader: DataLoader,
        retain_loader: DataLoader,
        **kwargs
    ) -> nn.Module:
        """
        Unlearn using entropy maximization on forget set.
        
        Goal: Make predictions on forget set UNIFORM (maximum entropy)
        while maintaining correct predictions on retain set.
        
        Args:
            forget_loader: DataLoader for samples to forget
            retain_loader: DataLoader for samples to retain
            
        Returns:
            The unlearned model
        """
        self._print(f"Entropy Maximization unlearning for {self.epochs} epochs...")
        
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=0.9,
            weight_decay=self.weight_decay
        )
        criterion = nn.CrossEntropyLoss()
        
        # Track initial retain accuracy for early stopping
        initial_retain_acc = self.evaluate(retain_loader)['accuracy']
        self._print(f"Initial retain accuracy: {initial_retain_acc:.2f}%")
        
        for epoch in range(self.epochs):
            self.model.train()
            total_forget_loss = 0.0
            total_retain_loss = 0.0
            
            # Create iterators
            forget_iter = iter(forget_loader)
            retain_iter = iter(retain_loader)
            
            steps = max(len(forget_loader), len(retain_loader))
            
            for step in range(steps):
                optimizer.zero_grad()
                combined_loss = 0.0
                
                # === FORGET: Maximize entropy (uniform predictions) ===
                try:
                    forget_inputs, _ = next(forget_iter)
                except StopIteration:
                    forget_iter = iter(forget_loader)
                    forget_inputs, _ = next(forget_iter)
                
                forget_inputs = forget_inputs.to(self.device)
                forget_outputs = self.model(forget_inputs)
                
                # Entropy loss (minimize negative entropy = maximize entropy)
                entropy_loss = self._entropy_loss(forget_outputs)
                combined_loss += self.entropy_weight * entropy_loss
                total_forget_loss += entropy_loss.item()
                
                # === RETAIN: Standard classification loss ===
                try:
                    retain_inputs, retain_targets = next(retain_iter)
                except StopIteration:
                    retain_iter = iter(retain_loader)
                    retain_inputs, retain_targets = next(retain_iter)
                
                retain_inputs = retain_inputs.to(self.device)
                retain_targets = retain_targets.to(self.device)
                
                retain_outputs = self.model(retain_inputs)
                retain_loss = criterion(retain_outputs, retain_targets)
                combined_loss += self.retain_weight * retain_loss
                total_retain_loss += retain_loss.item()
                
                combined_loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
            
            # Check retain accuracy for early stopping
            current_retain_acc = self.evaluate(retain_loader)['accuracy']
            if current_retain_acc < initial_retain_acc * 0.7:
                self._print(f"  Epoch {epoch+1}: Retain dropped to {current_retain_acc:.2f}%, stopping early")
                break
            
            if self.verbose and (epoch + 1) % max(1, self.epochs // 5) == 0:
                forget_metrics = self.evaluate(forget_loader)
                retain_metrics = self.evaluate(retain_loader)
                self._print(f"  Epoch {epoch+1}/{self.epochs}: "
                           f"Forget Acc={forget_metrics['accuracy']:.2f}%, "
                           f"Retain Acc={retain_metrics['accuracy']:.2f}%")
        
        return self.model
