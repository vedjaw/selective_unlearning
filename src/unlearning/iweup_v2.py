"""
IWEUP v2.1: Balanced MIA-Focused Unlearning Method.

Key fix: Target TRULY RANDOM predictions (~10% accuracy), not 0%.

The problem with v2.0:
- MIA was excellent (0.003) but forget accuracy was 0.1%
- 0.1% means the model learned "never predict this class" (information leak)
- We want 10% accuracy (random guessing on 10 classes)

Solution:
- Add UNIFORMITY LOSS that explicitly targets uniform distribution
- Reduce aggressive confidence calibration
- Balance entropy with uniformity to achieve true randomness
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
from tqdm import tqdm
import numpy as np

from .base import BaseUnlearner, TrainingMixin


class IWEUPv2Unlearner(BaseUnlearner, TrainingMixin):
    """
    IWEUP v2.1: Balanced MIA-Focused Unlearning.
    
    Goal: Achieve TRUE randomness (10% forget accuracy) while fooling MIA.
    
    Key insight: 
    - 0% accuracy = learned to avoid = information leak
    - 10% accuracy = truly random = no information
    - 100% accuracy = fully remembers = no forgetting
    
    Loss components:
    1. UNIFORMITY LOSS: Push predictions toward uniform (1/K for K classes)
    2. Soft entropy: Encourage spreading probability mass
    3. MIA-fooling: Match non-member confidence patterns
    4. Retain preservation: Maintain accuracy on retained data
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        verbose: bool = True,
        epochs: int = 15,
        lr: float = 0.001,
        # Loss weights - REBALANCED for uniformity
        uniformity_weight: float = 5.0,  # NEW: Strong push toward uniform
        entropy_weight: float = 1.0,  # Reduced from 2.0
        mia_weight: float = 1.0,  # Reduced from 2.0
        retain_weight: float = 5.0,
        # Temperature for softer predictions
        temperature: float = 1.5,  # Reduced from 2.0
        # Safety params
        min_retain_ratio: float = 0.85,
        max_grad_norm: float = 1.0,
    ):
        super().__init__(model, device, verbose)
        self.epochs = epochs
        self.lr = lr
        self.uniformity_weight = uniformity_weight
        self.entropy_weight = entropy_weight
        self.mia_weight = mia_weight
        self.retain_weight = retain_weight
        self.temperature = temperature
        self.min_retain_ratio = min_retain_ratio
        self.max_grad_norm = max_grad_norm
        
        self.num_classes = None
        self._sample_level = False  # Auto-detected
    
    def _detect_num_classes(self, loader: DataLoader) -> int:
        """Detect number of classes from data loader."""
        all_labels = set()
        for _, targets in loader:
            all_labels.update(targets.cpu().numpy().tolist())
            if len(all_labels) > 50:
                break
        return max(all_labels) + 1
    
    def _detect_forget_mode(self, forget_loader: DataLoader, retain_loader: DataLoader) -> bool:
        """
        Auto-detect if this is sample-level or class-level forgetting.
        
        Sample-level: forget set contains many classes (random samples).
        Class-level: forget set contains 1-2 classes.
        
        Returns True if sample-level.
        """
        forget_labels = set()
        for _, targets in forget_loader:
            forget_labels.update(targets.cpu().numpy().tolist())
            if len(forget_labels) > 5:
                break
        
        retain_labels = set()
        for _, targets in retain_loader:
            retain_labels.update(targets.cpu().numpy().tolist())
            if len(retain_labels) > 5:
                break
        
        # If forget set has most of the same classes as retain → sample-level
        overlap_ratio = len(forget_labels & retain_labels) / max(len(forget_labels), 1)
        is_sample_level = overlap_ratio > 0.5 and len(forget_labels) > 3
        
        return is_sample_level
    
    def _uniformity_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Push predictions toward uniform distribution.
        Used for CLASS-LEVEL forgetting.
        """
        probs = F.softmax(logits / self.temperature, dim=1)
        uniform = torch.ones_like(probs) / self.num_classes
        
        kl_div = F.kl_div(
            F.log_softmax(logits / self.temperature, dim=1),
            uniform,
            reduction='batchmean'
        )
        
        return kl_div
    
    def _soft_entropy_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """Soft entropy maximization toward target entropy."""
        probs = F.softmax(logits / self.temperature, dim=1)
        log_probs = F.log_softmax(logits / self.temperature, dim=1)
        entropy = -(probs * log_probs).sum(dim=1)
        
        target_entropy = np.log(self.num_classes)
        entropy_loss = (entropy - target_entropy).abs().mean()
        
        return entropy_loss
    
    def _mia_fooling_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """MIA-fooling: make confidence on true label look like non-member."""
        probs = F.softmax(logits / self.temperature, dim=1)
        
        batch_indices = torch.arange(logits.size(0), device=self.device)
        true_class_conf = probs[batch_indices, labels]
        
        target_conf = 1.0 / self.num_classes
        mia_loss = (true_class_conf - target_conf).abs().mean()
        
        return mia_loss
    
    def _confidence_reduction_loss(self, logits: torch.Tensor, labels: torch.Tensor, 
                                    smooth_factor: float = 0.7) -> torch.Tensor:
        """
        Confidence reduction for SAMPLE-LEVEL forgetting.
        
        Label smoothing toward soft targets. Higher smooth_factor = less smoothing
        (gentler). Lower = more aggressive confidence reduction.
        
        Args:
            smooth_factor: confidence on true class in soft target (0.7=aggressive, 0.9=gentle)
        """
        batch_size = logits.size(0)
        smooth_targets = torch.full_like(
            F.softmax(logits, dim=1),
            (1.0 - smooth_factor) / (self.num_classes - 1)
        )
        smooth_targets.scatter_(1, labels.unsqueeze(1), smooth_factor)
        
        log_probs = F.log_softmax(logits / self.temperature, dim=1)
        loss = F.kl_div(log_probs, smooth_targets, reduction='batchmean')
        
        return loss
    
    def _memorization_loss(self, logits: torch.Tensor, labels: torch.Tensor,
                           threshold: float = 0.8) -> torch.Tensor:
        """
        Anti-memorization loss: penalizes high confidence on true class.
        
        Args:
            threshold: confidence above this is penalized (0.8=aggressive, 0.95=gentle)
        """
        probs = F.softmax(logits, dim=1)
        batch_indices = torch.arange(logits.size(0), device=self.device)
        true_conf = probs[batch_indices, labels]
        
        excess = F.relu(true_conf - threshold)
        
        return excess.mean()
    
    def _compute_class_level_forget_loss(self, outputs, labels, effective_uniformity_weight):
        """Compute forget loss for CLASS-LEVEL forgetting."""
        uniform_loss = self._uniformity_loss(outputs)
        entropy_loss = self._soft_entropy_loss(outputs)
        mia_loss = self._mia_fooling_loss(outputs, labels)
        
        forget_loss = (
            effective_uniformity_weight * uniform_loss +
            self.entropy_weight * entropy_loss +
            self.mia_weight * mia_loss
        )
        return forget_loss, uniform_loss.item(), entropy_loss.item(), mia_loss.item()
    
    def _compute_sample_level_forget_loss(self, outputs, labels):
        """
        Compute forget loss for SAMPLE-LEVEL forgetting.
        
        Strategy: reduce memorization on specific samples without
        destroying the model's class-level knowledge.
        
        Components:
        1. Confidence reduction (label smoothing toward soft targets)
        2. Anti-memorization (penalize very high confidence)
        3. MIA fooling (reduce distinguishability from non-members)
        """
        conf_loss = self._confidence_reduction_loss(outputs, labels)
        memo_loss = self._memorization_loss(outputs, labels)
        mia_loss = self._mia_fooling_loss(outputs, labels)
        
        # Weights: prioritize confidence reduction
        forget_loss = (
            3.0 * conf_loss +
            2.0 * memo_loss +
            1.0 * mia_loss
        )
        return forget_loss, conf_loss.item(), memo_loss.item(), mia_loss.item()
    
    def unlearn(
        self,
        forget_loader: DataLoader,
        retain_loader: DataLoader,
        test_loader: DataLoader = None,
        **kwargs
    ) -> nn.Module:
        """
        Perform balanced MIA-focused unlearning.
        
        Auto-detects class-level vs sample-level forgetting:
        - Class-level: push forget class toward uniform predictions
        - Sample-level: reduce memorization on specific samples
        """
        self.num_classes = self._detect_num_classes(retain_loader)
        self._sample_level = self._detect_forget_mode(forget_loader, retain_loader)
        
        self._print(f"Detected {self.num_classes} classes")
        
        if self._sample_level:
            self._print("MODE: Sample-level forgetting (random samples)")
            self._print("Strategy: Confidence reduction + anti-memorization")
            # Fixed settings — tuned for 1% (best result)
            effective_epochs = min(self.epochs, 10)
            effective_lr = self.lr * 0.5
            effective_uniformity_weight = 0  # Not used
        else:
            self._print(f"MODE: Class-level forgetting")
            self._print(f"Target forget accuracy: ~{100/self.num_classes:.1f}% (random)")
            
            # === Adaptive hyperparameters based on num_classes ===
            if self.num_classes > 50:
                effective_uniformity_weight = self.uniformity_weight * 3.0
                effective_epochs = max(self.epochs, 30)
                effective_lr = self.lr * 2.0
                early_stop_tolerance = 2.0
                self._print(f"CIFAR-100 mode: uniformity={effective_uniformity_weight}, "
                           f"epochs={effective_epochs}, lr={effective_lr}")
            elif self.num_classes > 10:
                effective_uniformity_weight = self.uniformity_weight * 2.0
                effective_epochs = max(self.epochs, 20)
                effective_lr = self.lr * 1.5
                early_stop_tolerance = 3.0
            else:
                effective_uniformity_weight = self.uniformity_weight
                effective_epochs = self.epochs
                effective_lr = self.lr
                early_stop_tolerance = 5.0
        
        # Initial accuracy
        initial_retain_acc = self.evaluate(retain_loader)['accuracy']
        initial_forget_acc = self.evaluate(forget_loader)['accuracy']
        self._print(f"Initial: Retain={initial_retain_acc:.2f}%, Forget={initial_forget_acc:.2f}%")
        
        self._print(f"IWEUP v2.1 unlearning for {effective_epochs} epochs...")
        
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=effective_lr,
            momentum=0.9,
            weight_decay=5e-4
        )
        retain_criterion = nn.CrossEntropyLoss()
        
        for epoch in range(effective_epochs):
            self.model.train()
            
            forget_iter = iter(forget_loader)
            retain_iter = iter(retain_loader)
            
            num_batches = max(len(forget_loader), len(retain_loader))
            
            epoch_l1 = 0.0
            epoch_l2 = 0.0
            epoch_l3 = 0.0
            
            for batch_idx in range(num_batches):
                # === FORGET STEP ===
                optimizer.zero_grad()
                
                try:
                    inputs, labels = next(forget_iter)
                except StopIteration:
                    forget_iter = iter(forget_loader)
                    inputs, labels = next(forget_iter)
                
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(inputs)
                
                if self._sample_level:
                    forget_loss, l1, l2, l3 = self._compute_sample_level_forget_loss(
                        outputs, labels
                    )
                else:
                    forget_loss, l1, l2, l3 = self._compute_class_level_forget_loss(
                        outputs, labels, effective_uniformity_weight
                    )
                
                forget_loss.backward()
                epoch_l1 += l1
                epoch_l2 += l2
                epoch_l3 += l3
                
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                optimizer.step()
                
                # === RETAIN STEP ===
                optimizer.zero_grad()
                
                try:
                    retain_inputs, retain_targets = next(retain_iter)
                except StopIteration:
                    retain_iter = iter(retain_loader)
                    retain_inputs, retain_targets = next(retain_iter)
                
                retain_inputs = retain_inputs.to(self.device)
                retain_targets = retain_targets.to(self.device)
                
                retain_outputs = self.model(retain_inputs)
                retain_loss = self.retain_weight * retain_criterion(
                    retain_outputs, retain_targets
                )
                retain_loss.backward()
                
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                optimizer.step()
            
            # Check metrics for early stopping
            current_retain_acc = self.evaluate(retain_loader)['accuracy']
            current_forget_acc = self.evaluate(forget_loader)['accuracy']
            
            # Stop if retain drops too much
            if current_retain_acc < initial_retain_acc * self.min_retain_ratio:
                self._print(f"  Epoch {epoch+1}: Retain dropped to "
                           f"{current_retain_acc:.2f}%, stopping")
                break
            
            if self._sample_level:
                # Sample-level: stop based on MIA-like metric (confidence drop)
                if self.verbose and (epoch + 1) % max(1, effective_epochs // 5) == 0:
                    self._print(f"  Epoch {epoch+1}/{effective_epochs}: "
                               f"Forget={current_forget_acc:.2f}%, "
                               f"Retain={current_retain_acc:.2f}%, "
                               f"ConfRed={epoch_l1/num_batches:.4f}")
            else:
                # Class-level: stop if we're close to target
                target_acc = 100.0 / self.num_classes
                min_retain = 90.0 if self.num_classes <= 10 else 80.0
                if abs(current_forget_acc - target_acc) < early_stop_tolerance and current_retain_acc > min_retain:
                    self._print(f"  Epoch {epoch+1}: Target reached! "
                               f"Forget={current_forget_acc:.2f}% (target ~{target_acc:.1f}%), "
                               f"Retain={current_retain_acc:.2f}%")
                    break
                
                if self.verbose and (epoch + 1) % max(1, effective_epochs // 5) == 0:
                    self._print(f"  Epoch {epoch+1}/{effective_epochs}: "
                               f"Forget={current_forget_acc:.2f}%, "
                               f"Retain={current_retain_acc:.2f}%, "
                               f"Uniform={epoch_l1/num_batches:.4f}")
        
        # Final report
        final_forget = self.evaluate(forget_loader)['accuracy']
        final_retain = self.evaluate(retain_loader)['accuracy']
        if self._sample_level:
            self._print(f"Final: Forget={final_forget:.2f}%, Retain={final_retain:.2f}%")
        else:
            self._print(f"Final: Forget={final_forget:.2f}% (target ~{100/self.num_classes:.1f}%), "
                       f"Retain={final_retain:.2f}%")
        
        return self.model
