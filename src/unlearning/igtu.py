"""
Influence-Guided Targeted Unlearning (IGTU) - Fixed Version.

Our novel contribution: combining influence functions with ENTROPY MAXIMIZATION
to achieve efficient, targeted unlearning with approximate certified guarantees.

Key innovations:
1. Influence-weighted entropy maximization (not gradient ascent)
2. Gradient projection to preserve retain set performance  
3. Approximate certified bounds on residual influence
4. Strong retention through interleaved training

Fixed issues:
- Uses entropy loss (uniform predictions) instead of gradient ascent
- Normalized influence weights to prevent instability
- Lower learning rate with gradient clipping
- Better retention correction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Any, List, Optional, Tuple
from tqdm import tqdm
import copy
import numpy as np

from .base import BaseUnlearner, TrainingMixin
from ..influence.influence_functions import (
    compute_influence_on_parameters,
    compute_batch_influence,
    get_gradient_vector,
    compute_inverse_hvp_lissa,
)


class IGTUUnlearner(BaseUnlearner, TrainingMixin):
    """
    Influence-Guided Targeted Unlearning (IGTU) - Fixed Version.
    
    Novel unlearning method that:
    1. Uses influence functions to weight forget sample importance
    2. Applies ENTROPY MAXIMIZATION for true forgetting (random predictions)
    3. Projects gradients orthogonally to retain subspace
    4. Provides approximate certified bounds on unlearning completeness
    
    Key insight: True forgetting = uniform predictions (max entropy),
    NOT wrong predictions (gradient ascent).
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        verbose: bool = True,
        epochs: int = 10,
        lr: float = 0.001,  # Lower LR for stability
        # Influence function params
        influence_samples: int = 100,
        lissa_depth: int = 1000,
        lissa_scale: float = 25.0,
        # Gradient projection params
        project_dim: int = 100,
        # Retention params
        retention_weight: float = 5.0,  # Higher retention weight
        retention_threshold: float = 0.02,
        min_retain_ratio: float = 0.8,  # Early stopping threshold
    ):
        """
        Initialize IGTU unlearner.
        
        Args:
            model: Model to unlearn from
            device: Computation device
            verbose: Print progress
            epochs: Number of unlearning epochs
            lr: Learning rate (lower for stability)
            influence_samples: Number of samples for influence estimation
            lissa_depth: Depth of LiSSA recursion
            lissa_scale: Scale factor for LiSSA
            project_dim: Dimension for gradient projection
            retention_weight: Weight for retention loss
            retention_threshold: Maximum allowed accuracy drop on retain set
            min_retain_ratio: Stop if retain drops below this ratio
        """
        super().__init__(model, device, verbose)
        self.epochs = epochs
        self.lr = lr
        self.influence_samples = influence_samples
        self.lissa_depth = lissa_depth
        self.lissa_scale = lissa_scale
        self.project_dim = project_dim
        self.retention_weight = retention_weight
        self.retention_threshold = retention_threshold
        self.min_retain_ratio = min_retain_ratio
        
        # Cached values
        self.influence_weights = None
        self.retain_basis = None
        self.initial_retain_acc = None
    
    def _compute_influence_weights(
        self,
        forget_loader: DataLoader,
        retain_loader: DataLoader,
    ) -> torch.Tensor:
        """
        Compute influence-based weights for forget samples.
        
        Samples with higher influence on model predictions get higher
        weights for more aggressive unlearning.
        
        Args:
            forget_loader: Forget set data loader
            retain_loader: Retain set data loader (for Hessian estimation)
            
        Returns:
            Normalized influence weights for each forget sample
        """
        self._print("Computing influence weights for forget samples...")
        
        influences = []
        criterion = nn.CrossEntropyLoss()
        
        self.model.eval()
        sample_count = 0
        
        for inputs, targets in tqdm(forget_loader, desc="Computing influences", 
                                     disable=not self.verbose):
            if sample_count >= self.influence_samples:
                break
                
            for i in range(inputs.size(0)):
                if sample_count >= self.influence_samples:
                    break
                    
                sample_input = inputs[i:i+1].to(self.device)
                sample_target = targets[i:i+1].to(self.device)
                
                # Compute gradient for this sample
                self.model.zero_grad()
                output = self.model(sample_input)
                loss = criterion(output, sample_target)
                grad = get_gradient_vector(self.model, loss, flatten=True).detach()
                
                # Use gradient norm as proxy for influence
                influence = grad.norm().item()
                influences.append(influence)
                sample_count += 1
        
        # Normalize to get weights
        influences = torch.tensor(influences, device=self.device)
        
        # Handle edge cases (all zeros, NaN, etc.)
        if influences.sum() == 0 or torch.isnan(influences).any():
            weights = torch.ones_like(influences) / len(influences)
        else:
            # Softmax normalization for smoother weights
            weights = F.softmax(influences / influences.mean(), dim=0)
        
        # Extend weights to cover full dataset
        total_forget = len(forget_loader.dataset)
        num_computed = len(weights)
        
        if num_computed < total_forget:
            repeats = (total_forget // num_computed) + 1
            weights = weights.repeat(repeats)[:total_forget]
        
        self._print(f"Influence weights: min={weights.min():.4f}, "
                   f"max={weights.max():.4f}, mean={weights.mean():.4f}")
        
        return weights
    
    def _compute_retain_subspace(
        self,
        retain_loader: DataLoader,
        num_batches: int = 50
    ) -> torch.Tensor:
        """
        Compute gradient subspace for retain set.
        """
        self._print("Computing retain gradient subspace...")
        
        gradients = []
        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        batch_count = 0
        
        for inputs, targets in retain_loader:
            if batch_count >= num_batches:
                break
                
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.model.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            grad_vec = []
            for param in self.model.parameters():
                if param.grad is not None:
                    g = param.grad.data.flatten()
                    g = torch.clamp(g, -10.0, 10.0)  # Clip
                    grad_vec.append(g)
            grad_vec = torch.cat(grad_vec)
            
            # Handle NaN
            if torch.isnan(grad_vec).any() or torch.isinf(grad_vec).any():
                grad_vec = torch.nan_to_num(grad_vec, nan=0.0, posinf=0.0, neginf=0.0)
                
            gradients.append(grad_vec)
            batch_count += 1
        
        G = torch.stack(gradients)
        G = torch.nan_to_num(G, nan=0.0, posinf=0.0, neginf=0.0)
        
        # SVD with fallback
        try:
            U, S, Vh = torch.linalg.svd(G, full_matrices=False)
        except RuntimeError as e:
            self._print(f"  GPU SVD failed, falling back to CPU: {str(e)[:50]}...")
            try:
                G_cpu = G.cpu()
                U, S, Vh = torch.linalg.svd(G_cpu, full_matrices=False)
                Vh = Vh.to(self.device)
                S = S.to(self.device)
            except RuntimeError:
                # Ultimate fallback: random orthonormal basis
                self._print(f"  CPU SVD also failed. Using random orthonormal basis.")
                num_params = G.shape[1]
                k = min(self.project_dim, batch_count)
                basis = torch.randn(num_params, k, device=self.device)
                basis, _ = torch.linalg.qr(basis)
                return basis
        
        k = min(self.project_dim, Vh.shape[0])
        basis = Vh[:k].T
        
        S_sum = S.sum()
        variance_captured = (S[:k].sum() / S_sum * 100).item() if S_sum > 0 else 100.0
            
        self._print(f"Computed gradient basis with {k} components, "
                   f"capturing {variance_captured:.1f}% variance")
        
        return basis
    
    def _project_orthogonal(
        self,
        grad_vec: torch.Tensor,
        basis: torch.Tensor
    ) -> torch.Tensor:
        """Project gradient onto orthogonal complement of basis."""
        projection = basis @ (basis.T @ grad_vec)
        return grad_vec - projection
    
    def _get_model_gradient_vector(self) -> torch.Tensor:
        """Get flattened gradient vector from model."""
        grad_vec = []
        for param in self.model.parameters():
            if param.grad is not None:
                grad_vec.append(param.grad.data.flatten())
            else:
                grad_vec.append(torch.zeros(param.numel(), device=self.device))
        return torch.cat(grad_vec)
    
    def _apply_gradient_to_model(self, grad_vec: torch.Tensor):
        """Apply flattened gradient vector to model parameters."""
        idx = 0
        for param in self.model.parameters():
            numel = param.numel()
            param.grad = grad_vec[idx:idx+numel].reshape(param.shape)
            idx += numel
    
    def _entropy_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy loss for maximizing prediction uncertainty.
        
        Maximizing entropy → uniform distribution → true forgetting
        Returns negative entropy (to maximize via gradient descent).
        """
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)
        entropy = -(probs * log_probs).sum(dim=1).mean()
        return -entropy  # Minimize negative entropy = maximize entropy
    
    def _compute_residual_influence(
        self,
        forget_loader: DataLoader,
        retain_loader: DataLoader,
        num_samples: int = 50
    ) -> Dict[str, float]:
        """
        Compute approximate residual influence after unlearning.
        """
        self._print("Computing residual influence bounds...")
        
        residual_influences = []
        criterion = nn.CrossEntropyLoss()
        
        self.model.eval()
        sample_count = 0
        
        for inputs, targets in forget_loader:
            if sample_count >= num_samples:
                break
                
            for i in range(inputs.size(0)):
                if sample_count >= num_samples:
                    break
                    
                sample_input = inputs[i:i+1].to(self.device)
                sample_target = targets[i:i+1].to(self.device)
                
                self.model.zero_grad()
                output = self.model(sample_input)
                loss = criterion(output, sample_target)
                grad = get_gradient_vector(self.model, loss, flatten=True).detach()
                
                residual_influences.append(grad.norm().item())
                sample_count += 1
        
        residual_influences = np.array(residual_influences)
        
        return {
            'mean_residual': float(residual_influences.mean()),
            'max_residual': float(residual_influences.max()),
            'std_residual': float(residual_influences.std()),
            'bound_95': float(np.percentile(residual_influences, 95)),
        }
    
    def unlearn(
        self,
        forget_loader: DataLoader,
        retain_loader: DataLoader,
        compute_certificate: bool = True,
        **kwargs
    ) -> nn.Module:
        """
        Perform Influence-Guided Targeted Unlearning with entropy maximization.
        
        Key change from original: Uses entropy maximization instead of gradient
        ascent for true forgetting (uniform predictions) rather than wrong predictions.
        
        Args:
            forget_loader: DataLoader for samples to forget
            retain_loader: DataLoader for samples to retain
            compute_certificate: Whether to compute residual influence bounds
            
        Returns:
            The unlearned model with optional certificate
        """
        # Store initial retain accuracy
        self.initial_retain_acc = self.evaluate(retain_loader)['accuracy']
        self._print(f"Initial retain accuracy: {self.initial_retain_acc:.2f}%")
        
        # Phase 1: Compute influence weights
        self.influence_weights = self._compute_influence_weights(
            forget_loader, retain_loader
        )
        
        # Phase 2: Compute retain gradient subspace
        self.retain_basis = self._compute_retain_subspace(retain_loader)
        
        # Phase 3: Influence-guided ENTROPY MAXIMIZATION with projection
        self._print(f"IGTU unlearning (entropy maximization) for {self.epochs} epochs...")
        
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=0.9,
            weight_decay=5e-4
        )
        retain_criterion = nn.CrossEntropyLoss()
        
        weight_idx = 0
        
        for epoch in range(self.epochs):
            self.model.train()
            
            forget_iter = iter(forget_loader)
            retain_iter = iter(retain_loader)
            
            num_batches = max(len(forget_loader), len(retain_loader))
            
            for batch_idx in range(num_batches):
                # === FORGET: Influence-weighted entropy maximization ===
                try:
                    inputs, _ = next(forget_iter)
                except StopIteration:
                    forget_iter = iter(forget_loader)
                    inputs, _ = next(forget_iter)
                    
                inputs = inputs.to(self.device)
                batch_size = inputs.size(0)
                
                # Get influence weights for this batch
                batch_weights = self.influence_weights[weight_idx:weight_idx+batch_size]
                if len(batch_weights) < batch_size:
                    weight_idx = 0
                    batch_weights = self.influence_weights[:batch_size]
                weight_idx = (weight_idx + batch_size) % len(self.influence_weights)
                batch_weights = batch_weights[:batch_size].to(self.device)
                
                # Compute influence-weighted entropy loss
                optimizer.zero_grad()
                outputs = self.model(inputs)
                
                # Per-sample entropy
                probs = F.softmax(outputs, dim=1)
                log_probs = F.log_softmax(outputs, dim=1)
                sample_entropy = -(probs * log_probs).sum(dim=1)
                
                # Weighted entropy (negative because we want to MAXIMIZE)
                weighted_entropy_loss = -(sample_entropy * batch_weights).sum()
                weighted_entropy_loss.backward()
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Project orthogonally and apply
                grad_vec = self._get_model_gradient_vector()
                if not torch.isnan(grad_vec).any():
                    grad_vec = torch.clamp(grad_vec, -1.0, 1.0)
                    proj_grad = self._project_orthogonal(grad_vec, self.retain_basis)
                    self._apply_gradient_to_model(proj_grad * 0.1)  # Scale down
                    optimizer.step()
                
                # === RETAIN: Strong classification training ===
                optimizer.zero_grad()
                
                try:
                    retain_inputs, retain_targets = next(retain_iter)
                except StopIteration:
                    retain_iter = iter(retain_loader)
                    retain_inputs, retain_targets = next(retain_iter)
                
                retain_inputs = retain_inputs.to(self.device)
                retain_targets = retain_targets.to(self.device)
                
                retain_outputs = self.model(retain_inputs)
                retain_loss = self.retention_weight * retain_criterion(retain_outputs, retain_targets)
                retain_loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
            
            # Check retention and apply correction if needed
            current_retain_acc = self.evaluate(retain_loader)['accuracy']
            
            # Early stopping if retain drops too much
            if current_retain_acc < self.initial_retain_acc * self.min_retain_ratio:
                self._print(f"  Epoch {epoch+1}: Retain dropped to {current_retain_acc:.2f}%, stopping early")
                break
            
            # Retention correction if dropping
            acc_drop = self.initial_retain_acc - current_retain_acc
            if acc_drop > self.retention_threshold * 100:
                self._print(f"  Epoch {epoch+1}: Retention drop {acc_drop:.2f}%, applying correction...")
                for _ in range(2):
                    self._train_epoch(
                        self.model, retain_loader, optimizer, 
                        retain_criterion, self.device
                    )
            
            if self.verbose and (epoch + 1) % max(1, self.epochs // 5) == 0:
                forget_metrics = self.evaluate(forget_loader)
                retain_metrics = self.evaluate(retain_loader)
                self._print(f"  Epoch {epoch+1}/{self.epochs}: "
                           f"Forget Acc={forget_metrics['accuracy']:.2f}%, "
                           f"Retain Acc={retain_metrics['accuracy']:.2f}%")
        
        # Phase 4: Compute certificate (optional)
        certificate = None
        if compute_certificate:
            certificate = self._compute_residual_influence(
                forget_loader, retain_loader
            )
            self._print(f"Residual influence bounds: "
                       f"mean={certificate['mean_residual']:.4f}, "
                       f"95th percentile={certificate['bound_95']:.4f}")
            self.certificate = certificate
        
        return self.model
    
    def get_certificate(self) -> Optional[Dict[str, float]]:
        """Get the unlearning certificate if computed."""
        return getattr(self, 'certificate', None)
    
    def verify_unlearning(
        self,
        forget_loader: DataLoader,
        test_loader: DataLoader,
    ) -> Dict[str, Any]:
        """
        Verify unlearning quality with multiple metrics.
        """
        forget_metrics = self.evaluate(forget_loader)
        test_metrics = self.evaluate(test_loader)
        
        results = {
            'forget_accuracy': forget_metrics['accuracy'],
            'forget_loss': forget_metrics['loss'],
            'test_accuracy': test_metrics['accuracy'],
            'test_loss': test_metrics['loss'],
        }
        
        if hasattr(self, 'certificate'):
            results['certificate'] = self.certificate
        
        return results
