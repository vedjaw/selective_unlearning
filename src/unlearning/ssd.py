"""
Selective Synaptic Dampening (SSD) unlearning method - Fixed Version.

Based on: "Selective Synaptic Dampening: A Gradient-Based Approach 
to Selective Unlearning of Neural Networks"

Improvements:
- Adaptive dampening strength based on Fisher ratio
- Less aggressive dampening (scaled dampening_lambda)
- Validation checks during fine-tuning
- Early stopping if retain drops too much
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Any, List
from tqdm import tqdm
import copy

from .base import BaseUnlearner, TrainingMixin


class SSDUnlearner(BaseUnlearner, TrainingMixin):
    """
    Selective Synaptic Dampening (SSD) unlearning - Fixed Version.
    
    Uses Fisher Information to identify and dampen parameters important
    for the forget set while preserving parameters important for retention.
    
    Key improvements:
    - Adaptive dampening strength (not binary)
    - Less aggressive overall dampening
    - Early stopping when retain drops
    - Uses soft dampening instead of hard zeroing
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        verbose: bool = True,
        dampening_lambda: float = 0.5,  # Reduced from 1.0
        selection_alpha: float = 0.05,  # Reduced from 0.1 (less params dampened)
        epochs: int = 10,
        lr: float = 0.001,
        min_retain_ratio: float = 0.8,  # Early stopping threshold
    ):
        """
        Initialize SSD unlearner.
        
        Args:
            model: Model to unlearn from
            device: Computation device
            verbose: Print progress
            dampening_lambda: Strength of parameter dampening (0-1)
            selection_alpha: Fraction of parameters to dampen (0-1)
            epochs: Number of fine-tuning epochs
            lr: Learning rate for fine-tuning
            min_retain_ratio: Stop if retain acc drops below this ratio of initial
        """
        super().__init__(model, device, verbose)
        self.dampening_lambda = dampening_lambda
        self.selection_alpha = selection_alpha
        self.epochs = epochs
        self.lr = lr
        self.min_retain_ratio = min_retain_ratio
    
    def _compute_fisher_information(
        self,
        loader: DataLoader,
        num_samples: int = 1000
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Fisher Information for each parameter.
        
        The Fisher Information measures how much each parameter
        contributes to the model's predictions on the given data.
        
        Args:
            loader: Data loader
            num_samples: Maximum samples to use
            
        Returns:
            Dictionary mapping parameter names to Fisher values
        """
        fisher = {}
        for name, param in self.model.named_parameters():
            fisher[name] = torch.zeros_like(param)
        
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        samples_processed = 0
        
        for inputs, targets in loader:
            if samples_processed >= num_samples:
                break
                
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            batch_size = inputs.size(0)
            
            self.model.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fisher[name] += (param.grad.data ** 2) * batch_size
            
            samples_processed += batch_size
        
        # Normalize by number of samples
        for name in fisher:
            fisher[name] /= samples_processed
        
        return fisher
    
    def _get_dampening_mask_adaptive(
        self,
        forget_fisher: Dict[str, torch.Tensor],
        retain_fisher: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute adaptive dampening mask based on Fisher ratios.
        
        Instead of binary mask, uses continuous dampening strength:
        - Higher ratio (more important for forget) → more dampening
        - But capped to prevent over-dampening
        
        Args:
            forget_fisher: Fisher information for forget set
            retain_fisher: Fisher information for retain set
            
        Returns:
            Continuous mask (0-1) indicating dampening strength
        """
        mask = {}
        all_ratios = []
        
        # Compute importance ratios
        for name in forget_fisher:
            # Ratio of forget importance to retain importance
            ratio = forget_fisher[name] / (retain_fisher[name] + 1e-10)
            all_ratios.append(ratio.flatten())
        
        # Find threshold for top alpha fraction
        all_ratios_cat = torch.cat(all_ratios)
        threshold = torch.quantile(all_ratios_cat, 1 - self.selection_alpha)
        
        # Normalize ratios for adaptive dampening
        max_ratio = all_ratios_cat.max()
        
        # Create adaptive mask
        for name in forget_fisher:
            ratio = forget_fisher[name] / (retain_fisher[name] + 1e-10)
            
            # Binary selection: only dampen top alpha parameters
            selected = (ratio >= threshold).float()
            
            # Adaptive strength: normalize ratio to [0, 1]
            normalized_ratio = (ratio / (max_ratio + 1e-10)).clamp(0, 1)
            
            # Combine: selection mask * strength * lambda
            mask[name] = selected * normalized_ratio * self.dampening_lambda
        
        return mask
    
    def unlearn(
        self,
        forget_loader: DataLoader,
        retain_loader: DataLoader,
        **kwargs
    ) -> nn.Module:
        """
        Unlearn using Selective Synaptic Dampening with adaptive dampening.
        
        Args:
            forget_loader: DataLoader for samples to forget
            retain_loader: DataLoader for samples to retain
            
        Returns:
            The unlearned model
        """
        # Get initial accuracy
        initial_retain_acc = self.evaluate(retain_loader)['accuracy']
        self._print(f"Initial retain accuracy: {initial_retain_acc:.2f}%")
        
        self._print("Computing Fisher Information for forget set...")
        forget_fisher = self._compute_fisher_information(forget_loader)
        
        self._print("Computing Fisher Information for retain set...")
        retain_fisher = self._compute_fisher_information(retain_loader)
        
        self._print("Computing adaptive dampening mask...")
        mask = self._get_dampening_mask_adaptive(forget_fisher, retain_fisher)
        
        # Count dampened parameters
        total_params = sum(m.numel() for m in mask.values())
        dampened_params = sum((m > 0).sum().item() for m in mask.values())
        avg_dampening = sum(m.sum().item() for m in mask.values()) / total_params
        self._print(f"Dampening {dampened_params:.0f}/{total_params} parameters "
                   f"({100*dampened_params/total_params:.2f}%), "
                   f"avg strength: {avg_dampening:.4f}")
        
        # Apply adaptive dampening to selected parameters
        self._print("Applying adaptive dampening...")
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in mask:
                    # Soft dampening: multiply by (1 - mask)
                    dampening_factor = (1 - mask[name]).clamp(0, 1)
                    param.data *= dampening_factor
        
        # Check post-dampening accuracy
        post_dampen_acc = self.evaluate(retain_loader)['accuracy']
        self._print(f"Post-dampening retain accuracy: {post_dampen_acc:.2f}%")
        
        # Fine-tune on retain set
        self._print(f"Fine-tuning on retain set for {self.epochs} epochs...")
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=0.9,
            weight_decay=5e-4
        )
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(self.epochs):
            self._train_epoch(
                self.model, retain_loader, optimizer, criterion, self.device
            )
            
            # Check retain accuracy
            current_retain_acc = self.evaluate(retain_loader)['accuracy']
            
            # Early stopping
            if current_retain_acc < initial_retain_acc * self.min_retain_ratio:
                self._print(f"  Epoch {epoch+1}: Retain at {current_retain_acc:.2f}% "
                           f"(below {self.min_retain_ratio*100:.0f}% threshold), stopping")
                break
            
            if self.verbose and (epoch + 1) % max(1, self.epochs // 3) == 0:
                forget_metrics = self.evaluate(forget_loader)
                retain_metrics = self.evaluate(retain_loader)
                self._print(f"  Epoch {epoch+1}/{self.epochs}: "
                           f"Forget Acc={forget_metrics['accuracy']:.2f}%, "
                           f"Retain Acc={retain_metrics['accuracy']:.2f}%")
        
        return self.model
