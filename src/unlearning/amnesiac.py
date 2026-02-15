"""
Amnesiac Unlearning method.

Based on: "Eternal Sunshine of the Spotless Net: Selective Forgetting 
in Deep Networks" (Golatkar et al.)

Simple approach: fine-tune on retain data with label noise injected 
on forget samples to 'corrupt' the learned associations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Any
from tqdm import tqdm

from .base import BaseUnlearner, TrainingMixin


class AmnesiacUnlearner(BaseUnlearner, TrainingMixin):
    """
    Amnesiac unlearning via noisy fine-tuning.
    
    Key idea: Fine-tune on the combined dataset but with 
    randomly shuffled labels for forget samples. This corrupts
    the model's memory of the forget data while maintaining
    performance on retain data.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        verbose: bool = True,
        epochs: int = 10,
        lr: float = 0.001,
        noise_strength: float = 1.0,  # 1.0 = fully random labels
    ):
        """
        Args:
            model: Model to unlearn from
            device: Computation device
            verbose: Print progress
            epochs: Number of unlearning epochs
            lr: Learning rate
            noise_strength: Probability of replacing forget labels with random ones
        """
        super().__init__(model, device, verbose)
        self.epochs = epochs
        self.lr = lr
        self.noise_strength = noise_strength
        self.num_classes = None
    
    def _detect_num_classes(self, loader: DataLoader) -> int:
        """Detect number of classes from data loader."""
        max_label = 0
        for _, targets in loader:
            max_label = max(max_label, targets.max().item())
            if max_label >= 9:
                break
        return max_label + 1
    
    def unlearn(
        self,
        forget_loader: DataLoader,
        retain_loader: DataLoader,
        **kwargs
    ) -> nn.Module:
        """
        Unlearn using Amnesiac method (noisy label fine-tuning).
        
        Fine-tunes on retain data normally, and on forget data with
        random labels to corrupt the learned associations.
        """
        self.num_classes = self._detect_num_classes(retain_loader)
        self._print(f"Amnesiac unlearning for {self.epochs} epochs...")
        self._print(f"  Noise strength: {self.noise_strength}")
        
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=0.9,
            weight_decay=5e-4
        )
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0
            
            forget_iter = iter(forget_loader)
            retain_iter = iter(retain_loader)
            steps = max(len(forget_loader), len(retain_loader))
            
            for _ in range(steps):
                optimizer.zero_grad()
                loss = 0.0
                
                # === Retain: normal training ===
                try:
                    r_inputs, r_targets = next(retain_iter)
                except StopIteration:
                    retain_iter = iter(retain_loader)
                    r_inputs, r_targets = next(retain_iter)
                
                r_inputs = r_inputs.to(self.device)
                r_targets = r_targets.to(self.device)
                
                r_outputs = self.model(r_inputs)
                retain_loss = criterion(r_outputs, r_targets)
                loss += retain_loss
                
                # === Forget: training with RANDOM labels ===
                try:
                    f_inputs, f_targets = next(forget_iter)
                except StopIteration:
                    forget_iter = iter(forget_loader)
                    f_inputs, f_targets = next(forget_iter)
                
                f_inputs = f_inputs.to(self.device)
                f_targets = f_targets.to(self.device)
                
                # Replace labels with random ones
                if self.noise_strength >= 1.0:
                    noisy_targets = torch.randint(
                        0, self.num_classes, f_targets.shape, device=self.device
                    )
                else:
                    mask = torch.rand(f_targets.shape, device=self.device) < self.noise_strength
                    noisy_targets = f_targets.clone()
                    random_labels = torch.randint(
                        0, self.num_classes, f_targets.shape, device=self.device
                    )
                    noisy_targets[mask] = random_labels[mask]
                
                f_outputs = self.model(f_inputs)
                forget_loss = criterion(f_outputs, noisy_targets)
                loss += forget_loss
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if self.verbose and (epoch + 1) % max(1, self.epochs // 5) == 0:
                forget_metrics = self.evaluate(forget_loader)
                retain_metrics = self.evaluate(retain_loader)
                self._print(f"  Epoch {epoch+1}/{self.epochs}: "
                           f"Forget={forget_metrics['accuracy']:.2f}%, "
                           f"Retain={retain_metrics['accuracy']:.2f}%")
        
        return self.model
