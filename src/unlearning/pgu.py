"""
Projected Gradient Unlearning (PGU).

Based on: Hoang et al. (2024), "Learn to Unlearn for Deep Neural Networks: 
Minimizing Unlearning Interference with Gradient Projection"

Fixed version with:
- Lower learning rate for stability
- Gradient clipping to prevent explosion
- Early stopping when retain drops
- Interleaved retain training
- CPU fallback for SVD
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Optional
from tqdm import tqdm
import copy

from .base import BaseUnlearner, TrainingMixin


class PGUUnlearner(BaseUnlearner, TrainingMixin):
    """
    Projected Gradient Unlearning (PGU) - Fixed Version.
    
    Performs entropy-based unlearning while projecting gradients onto
    the orthogonal subspace of the retain set's gradient space.
    This minimizes interference with retained knowledge.
    
    Key improvements:
    - Uses entropy maximization instead of gradient ascent
    - Lower learning rate (0.0001 vs 0.01)
    - Gradient clipping for stability
    - Early stopping when retain accuracy drops
    - Interleaved retain training every batch
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        verbose: bool = True,
        epochs: int = 10,
        lr: float = 0.0001,  # Much lower LR
        project_dim: int = 100,
        weight_decay: float = 5e-4,
        retain_weight: float = 5.0,  # Strong retain weight
        min_retain_ratio: float = 0.7,  # Stop if retain drops below 70% of original
    ):
        """
        Initialize PGU unlearner.
        
        Args:
            model: Model to unlearn from
            device: Computation device
            verbose: Print progress
            epochs: Number of unlearning epochs
            lr: Learning rate (keep LOW for stability)
            project_dim: Dimension for gradient projection basis
            weight_decay: Weight decay for optimizer
            retain_weight: Weight for retain loss
            min_retain_ratio: Stop if retain acc drops below this ratio of initial
        """
        super().__init__(model, device, verbose)
        self.epochs = epochs
        self.lr = lr
        self.project_dim = project_dim
        self.weight_decay = weight_decay
        self.retain_weight = retain_weight
        self.min_retain_ratio = min_retain_ratio
        
        # Cache for gradient subspace basis
        self.retain_basis = None
    
    def _compute_gradient_subspace(
        self,
        loader: DataLoader,
        num_batches: int = 50
    ) -> torch.Tensor:
        """
        Compute the gradient subspace basis for retain set.
        
        Uses random projection to find an approximate basis of the
        gradient space spanned by the retain samples.
        
        Args:
            loader: Retain set data loader
            num_batches: Number of batches to use
            
        Returns:
            Orthonormal basis matrix (num_params x project_dim)
        """
        self._print("Computing retain gradient subspace...")
        
        # Collect gradients
        gradients = []
        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        batch_count = 0
        
        for inputs, targets in loader:
            if batch_count >= num_batches:
                break
                
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.model.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Flatten all gradients into single vector with clipping
            grad_vec = []
            for param in self.model.parameters():
                if param.grad is not None:
                    g = param.grad.data.flatten()
                    # Clip gradients
                    g = torch.clamp(g, -10.0, 10.0)
                    grad_vec.append(g)
            grad_vec = torch.cat(grad_vec)
            
            # Handle NaN/Inf
            if torch.isnan(grad_vec).any() or torch.isinf(grad_vec).any():
                grad_vec = torch.nan_to_num(grad_vec, nan=0.0, posinf=0.0, neginf=0.0)
            
            gradients.append(grad_vec)
            batch_count += 1
        
        # Stack gradients: (num_batches x num_params)
        G = torch.stack(gradients)
        
        # Clean gradient matrix
        G = torch.nan_to_num(G, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Compute SVD with fallbacks
        try:
            U, S, Vh = torch.linalg.svd(G, full_matrices=False)
        except RuntimeError as e:
            # GPU SVD failed, try CPU fallback
            self._print(f"  GPU SVD failed, falling back to CPU: {str(e)[:50]}...")
            try:
                G_cpu = G.cpu()
                U, S, Vh = torch.linalg.svd(G_cpu, full_matrices=False)
                Vh = Vh.to(self.device)
            except RuntimeError as e2:
                # Complete failure, return random orthonormal basis
                self._print(f"  CPU SVD also failed. Using random orthonormal basis.")
                num_params = G.shape[1]
                k = min(self.project_dim, batch_count)
                basis = torch.randn(num_params, k, device=self.device)
                basis, _ = torch.linalg.qr(basis)
                return basis
        
        # Take top project_dim components
        k = min(self.project_dim, Vh.shape[0])
        basis = Vh[:k].T  # (num_params x k)
        
        self._print(f"Computed gradient basis with {k} components")
        return basis
    
    def _project_gradient_orthogonal(
        self,
        grad_vec: torch.Tensor,
        basis: torch.Tensor
    ) -> torch.Tensor:
        """
        Project gradient onto the orthogonal complement of the retain subspace.
        
        Args:
            grad_vec: Gradient vector to project
            basis: Retain gradient subspace basis (num_params x k)
            
        Returns:
            Projected gradient vector (orthogonal to retain subspace)
        """
        # Project gradient onto basis: proj = basis @ (basis.T @ grad)
        projection = basis @ (basis.T @ grad_vec)
        
        # Orthogonal component: grad - proj
        return grad_vec - projection
    
    def _apply_gradient_to_model(self, grad_vec: torch.Tensor):
        """Apply a flattened gradient vector to model parameters."""
        idx = 0
        for param in self.model.parameters():
            numel = param.numel()
            param.grad = grad_vec[idx:idx+numel].reshape(param.shape)
            idx += numel
    
    def _get_model_gradient_vector(self) -> torch.Tensor:
        """Get flattened gradient vector from model."""
        grad_vec = []
        for param in self.model.parameters():
            if param.grad is not None:
                grad_vec.append(param.grad.data.flatten())
            else:
                grad_vec.append(torch.zeros(param.numel(), device=self.device))
        return torch.cat(grad_vec)
    
    def _entropy_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute entropy loss (negative entropy for maximization)."""
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)
        entropy = -(probs * log_probs).sum(dim=1).mean()
        return -entropy  # Minimize negative entropy = maximize entropy
    
    def unlearn(
        self,
        forget_loader: DataLoader,
        retain_loader: DataLoader,
        **kwargs
    ) -> nn.Module:
        """
        Unlearn using Projected Gradient method with entropy maximization.
        
        Args:
            forget_loader: DataLoader for samples to forget
            retain_loader: DataLoader for samples to retain
            
        Returns:
            The unlearned model
        """
        # Compute retain gradient subspace
        self.retain_basis = self._compute_gradient_subspace(retain_loader)
        
        # Get initial retain accuracy for early stopping
        initial_retain_acc = self.evaluate(retain_loader)['accuracy']
        self._print(f"Initial retain accuracy: {initial_retain_acc:.2f}%")
        
        self._print(f"PGU unlearning for {self.epochs} epochs...")
        
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=0.9,
            weight_decay=self.weight_decay
        )
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(self.epochs):
            self.model.train()
            
            forget_iter = iter(forget_loader)
            retain_iter = iter(retain_loader)
            
            steps = max(len(forget_loader), len(retain_loader))
            
            for step in range(steps):
                optimizer.zero_grad()
                
                # === FORGET: Entropy maximization with projection ===
                try:
                    forget_inputs, _ = next(forget_iter)
                except StopIteration:
                    forget_iter = iter(forget_loader)
                    forget_inputs, _ = next(forget_iter)
                
                forget_inputs = forget_inputs.to(self.device)
                forget_outputs = self.model(forget_inputs)
                entropy_loss = self._entropy_loss(forget_outputs)
                entropy_loss.backward()
                
                # Get gradient and project onto orthogonal complement
                grad_vec = self._get_model_gradient_vector()
                grad_vec = torch.clamp(grad_vec, -1.0, 1.0)  # Clip
                proj_grad = self._project_gradient_orthogonal(grad_vec, self.retain_basis)
                
                # Apply projected gradient
                self._apply_gradient_to_model(proj_grad * 0.1)  # Scale down
                optimizer.step()
                
                # === RETAIN: Standard training step ===
                optimizer.zero_grad()
                
                try:
                    retain_inputs, retain_targets = next(retain_iter)
                except StopIteration:
                    retain_iter = iter(retain_loader)
                    retain_inputs, retain_targets = next(retain_iter)
                
                retain_inputs = retain_inputs.to(self.device)
                retain_targets = retain_targets.to(self.device)
                
                retain_outputs = self.model(retain_inputs)
                retain_loss = criterion(retain_outputs, retain_targets)
                (self.retain_weight * retain_loss).backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
            
            # Early stopping check
            current_retain_acc = self.evaluate(retain_loader)['accuracy']
            if current_retain_acc < initial_retain_acc * self.min_retain_ratio:
                self._print(f"  Epoch {epoch+1}: Retain dropped to {current_retain_acc:.2f}%, stopping early")
                break
            
            if self.verbose and (epoch + 1) % max(1, self.epochs // 5) == 0:
                forget_metrics = self.evaluate(forget_loader)
                retain_metrics = self.evaluate(retain_loader)
                self._print(f"  Epoch {epoch+1}/{self.epochs}: "
                           f"Forget Acc={forget_metrics['accuracy']:.2f}%, "
                           f"Retain Acc={retain_metrics['accuracy']:.2f}%")
        
        return self.model
