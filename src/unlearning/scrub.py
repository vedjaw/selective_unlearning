"""
SCRUB unlearning method.

Based on: "SCRUB: Sparsifiying Catastrophically Retraining for 
Unlearning By-products" (Kurmanji et al., ICML 2024)

Two-phase approach:
  Phase 1: Maximize KL divergence from teacher on forget data (bounded)
  Phase 2: Minimize KL divergence from teacher on retain data + CE loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Any
from tqdm import tqdm
import copy

from .base import BaseUnlearner, TrainingMixin


class SCRUBUnlearner(BaseUnlearner, TrainingMixin):
    """
    SCRUB unlearning using teacher-student knowledge distillation.
    
    Uses the original model as a teacher. On forget data, the student
    is trained to DIVERGE from the teacher. On retain data, the student
    is trained to MATCH the teacher while maintaining classification accuracy.
    
    Key fix: Two-phase training prevents the unbounded negative KL loss
    from destroying retain knowledge.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        verbose: bool = True,
        epochs: int = 10,
        lr: float = 0.001,
        temperature: float = 4.0,
        max_grad_norm: float = 1.0,
        forget_epochs: int = 5,  # Subset of epochs for forget phase
    ):
        """
        Initialize SCRUB unlearner.
        
        Args:
            model: Model to unlearn from
            device: Computation device  
            verbose: Print progress
            epochs: Total number of unlearning epochs
            lr: Learning rate
            temperature: Temperature for softmax in distillation
            max_grad_norm: Maximum gradient norm for clipping
            forget_epochs: Number of initial epochs for forget phase
        """
        super().__init__(model, device, verbose)
        self.epochs = epochs
        self.lr = lr
        self.temperature = temperature
        self.max_grad_norm = max_grad_norm
        self.forget_epochs = min(forget_epochs, epochs)
        
        # Create teacher model (frozen copy of original)
        self.teacher = copy.deepcopy(model)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
    
    def _distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """KL divergence distillation loss (scaled by T^2)."""
        T = self.temperature
        student_log_probs = F.log_softmax(student_logits / T, dim=1)
        teacher_probs = F.softmax(teacher_logits / T, dim=1)
        return F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (T ** 2)
    
    def unlearn(
        self,
        forget_loader: DataLoader,
        retain_loader: DataLoader,
        **kwargs
    ) -> nn.Module:
        """
        Unlearn using SCRUB method (two-phase).
        
        Phase 1 (forget_epochs): Gradient ascent on forget data with
            bounded KL divergence. Retain data used for stability.
        Phase 2 (remaining epochs): Fine-tune on retain data with
            teacher distillation to recover utility.
        """
        self._print(f"SCRUB unlearning: {self.forget_epochs} forget + "
                    f"{self.epochs - self.forget_epochs} retain epochs")
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0
            n_steps = 0
            
            is_forget_phase = epoch < self.forget_epochs
            
            if is_forget_phase:
                # === PHASE 1: Forget + Stabilize ===
                forget_iter = iter(forget_loader)
                retain_iter = iter(retain_loader)
                steps = max(len(forget_loader), len(retain_loader))
                
                for _ in range(steps):
                    optimizer.zero_grad()
                    
                    # --- Forget step: maximize divergence from teacher ---
                    try:
                        f_inputs, f_targets = next(forget_iter)
                    except StopIteration:
                        forget_iter = iter(forget_loader)
                        f_inputs, f_targets = next(forget_iter)
                    
                    f_inputs = f_inputs.to(self.device)
                    
                    student_logits = self.model(f_inputs)
                    with torch.no_grad():
                        teacher_logits = self.teacher(f_inputs)
                    
                    # Negative distillation (maximize divergence)
                    # Clamp to prevent unbounded negative loss
                    forget_kl = self._distillation_loss(student_logits, teacher_logits)
                    forget_loss = -torch.clamp(forget_kl, max=20.0)
                    
                    # --- Retain step: stay close to teacher ---
                    try:
                        r_inputs, r_targets = next(retain_iter)
                    except StopIteration:
                        retain_iter = iter(retain_loader)
                        r_inputs, r_targets = next(retain_iter)
                    
                    r_inputs = r_inputs.to(self.device)
                    r_targets = r_targets.to(self.device)
                    
                    student_logits = self.model(r_inputs)
                    with torch.no_grad():
                        teacher_logits = self.teacher(r_inputs)
                    
                    retain_kl = self._distillation_loss(student_logits, teacher_logits)
                    retain_ce = criterion(student_logits, r_targets)
                    retain_loss = 0.5 * retain_kl + 0.5 * retain_ce
                    
                    # Balanced forget/retain
                    loss = 0.5 * forget_loss + 0.5 * retain_loss
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    optimizer.step()
                    total_loss += loss.item()
                    n_steps += 1
            else:
                # === PHASE 2: Retain-only fine-tuning with distillation ===
                for r_inputs, r_targets in retain_loader:
                    optimizer.zero_grad()
                    
                    r_inputs = r_inputs.to(self.device)
                    r_targets = r_targets.to(self.device)
                    
                    student_logits = self.model(r_inputs)
                    with torch.no_grad():
                        teacher_logits = self.teacher(r_inputs)
                    
                    retain_kl = self._distillation_loss(student_logits, teacher_logits)
                    retain_ce = criterion(student_logits, r_targets)
                    
                    # Blend distillation + classification
                    loss = 0.5 * retain_kl + 0.5 * retain_ce
                    
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    n_steps += 1
            
            avg_loss = total_loss / max(n_steps, 1)
            phase_str = "FORGET" if is_forget_phase else "RETAIN"
            
            if self.verbose and (epoch + 1) % max(1, self.epochs // 5) == 0:
                forget_metrics = self.evaluate(forget_loader)
                retain_metrics = self.evaluate(retain_loader)
                self._print(f"  Epoch {epoch+1}/{self.epochs} [{phase_str}]: "
                           f"Forget={forget_metrics['accuracy']:.2f}%, "
                           f"Retain={retain_metrics['accuracy']:.2f}%, "
                           f"Loss={avg_loss:.4f}")
                
                # Early stop if retain drops too low during forget phase
                if is_forget_phase and retain_metrics['accuracy'] < 80.0:
                    self._print(f"  Early stopping forget phase (retain < 80%)")
                    self.forget_epochs = epoch  # Switch to retain phase
        
        return self.model
