"""
SalUn: Saliency-based Unlearning (NeurIPS 2023).

Reference: Fan et al., "SalUn: Empowering Machine Unlearning via 
Gradient-based Weight Saliency in Both Image Classification and Generation"

Key idea: Use gradient saliency to identify which weights are most
important for the forget set, then selectively perturb ONLY those weights
using random labeling. This preserves utility on retain data while
effectively erasing forget-set memorization.

Algorithm:
1. Compute gradient saliency mask (top-k% of weights by gradient magnitude)
2. Assign random labels to forget samples
3. Train with random labels but ONLY update salient weights (mask others)
4. Simultaneously train on retain set with standard CE (no mask)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
import copy
import numpy as np

from .base import BaseUnlearner, TrainingMixin


class SalUnUnlearner(BaseUnlearner, TrainingMixin):
    """
    SalUn: Saliency-based Unlearning.
    
    Uses gradient-based weight saliency to identify and selectively update
    the most influential weights for the forget data, while preserving
    the rest of the model.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        verbose: bool = True,
        epochs: int = 10,
        lr: float = 0.01,
        weight_decay: float = 5e-4,
        threshold: float = 0.5,       # Top 50% of weights by saliency
        alpha: float = 1.0,           # Forget loss weight
        beta: float = 1.0,            # Retain loss weight
        num_classes: int = 10,
    ):
        super().__init__(model, device, verbose)
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.threshold = threshold
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        
    def _compute_saliency_mask(
        self, 
        forget_loader: DataLoader,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute gradient saliency mask for all model parameters.
        
        For each parameter, we accumulate the absolute gradient magnitude
        over the forget set, then threshold to keep only the top-k%.
        
        Returns:
            Dictionary mapping parameter name -> binary mask tensor
        """
        self._print("Computing weight saliency mask...")
        
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        
        # Accumulate gradient magnitudes
        grad_magnitudes = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                grad_magnitudes[name] = torch.zeros_like(param.data)
        
        # Forward + backward on forget set to get gradients
        num_batches = 0
        for inputs, targets in forget_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.model.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    grad_magnitudes[name] += param.grad.data.abs()
            
            num_batches += 1
        
        # Average gradients
        for name in grad_magnitudes:
            grad_magnitudes[name] /= max(num_batches, 1)
        
        # Create binary mask: keep top threshold% of weights by saliency
        masks = {}
        total_params = 0
        salient_params = 0
        
        for name, grad_mag in grad_magnitudes.items():
            flat = grad_mag.flatten()
            k = max(1, int(len(flat) * self.threshold))
            
            # Get threshold value (k-th largest)
            if k < len(flat):
                threshold_val = torch.kthvalue(flat, len(flat) - k + 1).values
            else:
                threshold_val = flat.min()
            
            mask = (grad_mag >= threshold_val).float()
            masks[name] = mask
            
            total_params += mask.numel()
            salient_params += mask.sum().item()
        
        self._print(f"  Saliency mask: {salient_params/total_params*100:.1f}% "
                     f"of weights marked as salient "
                     f"({int(salient_params)}/{total_params})")
        
        return masks
    
    def _generate_random_labels(
        self, 
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate random (incorrect) labels for forget samples.
        Each target is replaced with a uniformly random different class.
        """
        batch_size = targets.size(0)
        random_labels = torch.randint(0, self.num_classes, (batch_size,), 
                                       device=targets.device)
        # Ensure labels are different from true labels
        same_mask = (random_labels == targets)
        while same_mask.any():
            random_labels[same_mask] = torch.randint(
                0, self.num_classes, (same_mask.sum(),), 
                device=targets.device
            )
            same_mask = (random_labels == targets)
        
        return random_labels
    
    def unlearn(
        self,
        forget_loader: DataLoader,
        retain_loader: DataLoader,
        **kwargs
    ) -> nn.Module:
        """
        Unlearn using SalUn: saliency-guided random labeling.
        
        Steps:
        1. Compute saliency mask from forget set gradients
        2. For each epoch:
           a. Train on forget set with random labels (masked gradients)
           b. Train on retain set with true labels (unmasked gradients)
        """
        # Detect number of classes
        all_labels = set()
        for _, targets in forget_loader:
            all_labels.update(targets.numpy().tolist() 
                            if hasattr(targets, 'numpy') else targets.tolist())
        for _, targets in retain_loader:
            all_labels.update(targets.numpy().tolist() 
                            if hasattr(targets, 'numpy') else targets.tolist())
            if len(all_labels) > 50:
                break
        self.num_classes = max(len(all_labels), self.num_classes)
        
        self._print(f"SalUn unlearning for {self.epochs} epochs "
                     f"(threshold={self.threshold}, classes={self.num_classes})")
        
        # Step 1: Compute saliency mask
        masks = self._compute_saliency_mask(forget_loader)
        
        # Step 2: Unlearning with masked gradient updates  
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=0.9,
            weight_decay=self.weight_decay
        )
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(self.epochs):
            self.model.train()
            total_forget_loss = 0.0
            total_retain_loss = 0.0
            
            forget_iter = iter(forget_loader)
            retain_iter = iter(retain_loader)
            steps = max(len(forget_loader), len(retain_loader))
            
            for step in range(steps):
                optimizer.zero_grad()
                
                # === FORGET: Random labels + masked gradients ===
                try:
                    f_inputs, f_targets = next(forget_iter)
                except StopIteration:
                    forget_iter = iter(forget_loader)
                    f_inputs, f_targets = next(forget_iter)
                
                f_inputs = f_inputs.to(self.device)
                f_targets = f_targets.to(self.device)
                
                # Generate random labels
                random_targets = self._generate_random_labels(f_targets)
                
                f_outputs = self.model(f_inputs)
                forget_loss = self.alpha * criterion(f_outputs, random_targets)
                forget_loss.backward()
                
                # Apply saliency mask: zero out gradients for non-salient weights
                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        if param.requires_grad and param.grad is not None:
                            if name in masks:
                                param.grad.data *= masks[name]
                
                total_forget_loss += forget_loss.item()
                
                # === RETAIN: Standard CE, no masking ===
                try:
                    r_inputs, r_targets = next(retain_iter)
                except StopIteration:
                    retain_iter = iter(retain_loader)
                    r_inputs, r_targets = next(retain_iter)
                
                r_inputs = r_inputs.to(self.device)
                r_targets = r_targets.to(self.device)
                
                r_outputs = self.model(r_inputs)
                retain_loss = self.beta * criterion(r_outputs, r_targets)
                retain_loss.backward()
                
                total_retain_loss += retain_loss.item()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0
                )
                
                optimizer.step()
            
            # Log progress
            if self.verbose and (epoch + 1) % max(1, self.epochs // 5) == 0:
                forget_metrics = self.evaluate(forget_loader)
                retain_metrics = self.evaluate(retain_loader)
                self._print(
                    f"  Epoch {epoch+1}/{self.epochs}: "
                    f"Forget={forget_metrics['accuracy']:.2f}%, "
                    f"Retain={retain_metrics['accuracy']:.2f}%"
                )
        
        return self.model
