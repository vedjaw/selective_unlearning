"""
Bad Teacher unlearning method.

Based on: "Machine Unlearning of Features and Labels" 
(Chundawat et al., NeurIPS 2023)

Uses a randomly initialized 'bad teacher' to provide incorrect
guidance on forget samples, while a 'good teacher' (original model)
maintains performance on retain samples via knowledge distillation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Any
from tqdm import tqdm
import copy

from .base import BaseUnlearner, TrainingMixin


class BadTeacherUnlearner(BaseUnlearner, TrainingMixin):
    """
    Bad Teacher unlearning via incompetent teacher distillation.
    
    Key idea: Use a randomly initialized model as a 'bad teacher'
    to guide predictions on forget samples toward random outputs,
    while using the original model as a 'good teacher' for retain samples.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        verbose: bool = True,
        epochs: int = 10,
        lr: float = 0.001,
        temperature: float = 4.0,
        alpha: float = 0.5,  # Balance between bad/good teacher
    ):
        """
        Args:
            model: Model to unlearn from
            device: Computation device
            verbose: Print progress
            epochs: Number of unlearning epochs
            lr: Learning rate
            temperature: Distillation temperature
            alpha: Weight for retain distillation (1-alpha for forget)
        """
        super().__init__(model, device, verbose)
        self.epochs = epochs
        self.lr = lr
        self.temperature = temperature
        self.alpha = alpha
        
        # Good teacher: frozen copy of original model
        self.good_teacher = copy.deepcopy(model)
        self.good_teacher.eval()
        for param in self.good_teacher.parameters():
            param.requires_grad = False
        
        # Bad teacher: randomly initialized model with same architecture
        self.bad_teacher = copy.deepcopy(model)
        def reinit(m):
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        self.bad_teacher.apply(reinit)
        self.bad_teacher.eval()
        for param in self.bad_teacher.parameters():
            param.requires_grad = False
    
    def _distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """KL divergence distillation loss."""
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)
        return F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (self.temperature ** 2)
    
    def unlearn(
        self,
        forget_loader: DataLoader,
        retain_loader: DataLoader,
        **kwargs
    ) -> nn.Module:
        """
        Unlearn using Bad Teacher method.
        
        On forget samples: distill from bad teacher (random outputs)
        On retain samples: distill from good teacher (original outputs) + CE loss
        """
        self._print(f"Bad Teacher unlearning for {self.epochs} epochs...")
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
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
                
                # === Forget: distill from BAD teacher ===
                try:
                    f_inputs, f_targets = next(forget_iter)
                except StopIteration:
                    forget_iter = iter(forget_loader)
                    f_inputs, f_targets = next(forget_iter)
                
                f_inputs = f_inputs.to(self.device)
                student_logits = self.model(f_inputs)
                
                with torch.no_grad():
                    bad_teacher_logits = self.bad_teacher(f_inputs)
                
                forget_loss = self._distillation_loss(student_logits, bad_teacher_logits)
                loss += (1 - self.alpha) * forget_loss
                
                # === Retain: distill from GOOD teacher + CE loss ===
                try:
                    r_inputs, r_targets = next(retain_iter)
                except StopIteration:
                    retain_iter = iter(retain_loader)
                    r_inputs, r_targets = next(retain_iter)
                
                r_inputs = r_inputs.to(self.device)
                r_targets = r_targets.to(self.device)
                
                student_logits = self.model(r_inputs)
                
                with torch.no_grad():
                    good_teacher_logits = self.good_teacher(r_inputs)
                
                retain_distill = self._distillation_loss(student_logits, good_teacher_logits)
                retain_ce = criterion(student_logits, r_targets)
                
                retain_loss = 0.5 * retain_distill + 0.5 * retain_ce
                loss += self.alpha * retain_loss
                
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
