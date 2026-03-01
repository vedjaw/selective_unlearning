"""
RINE: Redundant Information Neural Estimator.

Adapted from the EACL 2026 paper:
"Auditing Language Model Unlearning via Information Decomposition"
(https://github.com/UKPLab/eacl2026-auditing-unlearning)

Estimates the Partial Information Decomposition (PID) between
a base model and an unlearned model with respect to forget data:

- I∩ (Residual Knowledge): shared info that persists after unlearning
- I^B_uniq (Unlearned Knowledge): info successfully removed

The estimator is architecture-agnostic: it operates on extracted
representation vectors (Z_base, Z_unlearned) and binary labels Y.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Optional


class LogisticProbe(nn.Module):
    """Linear probe (logistic regression) mapping representation → 1 logit."""
    
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class RINE:
    """
    Redundant Information Neural Estimator.
    
    Uses two logistic probes (one per model) with a Lagrangian consistency
    constraint to estimate the redundant (shared) mutual information
    between the base and unlearned model representations about the
    forget/retain label Y.
    
    Key equation (Eq 12 in paper):
        L = 0.5 * (CE_base + CE_unlearned) + β * ||p_base - p_unlearned||₁
    
    The consistency term (β) forces both probes to agree, isolating
    the redundant (shared) information component.
    
    Args:
        dim_base: Dimension of base model representations
        dim_unlearned: Dimension of unlearned model representations
        lr: Learning rate for probe optimization
        device: Computation device
    """
    
    def __init__(
        self,
        dim_base: int,
        dim_unlearned: int,
        lr: float = 1e-3,
        device: str = 'cuda',
    ):
        self.f1 = LogisticProbe(dim_base).to(device)
        self.f2 = LogisticProbe(dim_unlearned).to(device)
        self.optimizer = optim.Adam(
            list(self.f1.parameters()) + list(self.f2.parameters()), lr=lr
        )
        self.device = device
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')
    
    def train_estimator(
        self,
        Z_base: np.ndarray,
        Z_unlearned: np.ndarray,
        Y: np.ndarray,
        beta: float = 5.0,
        epochs: int = 1000,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        Train RINE to estimate information decomposition.
        
        Args:
            Z_base: Base model representations, shape (N, D_base)
            Z_unlearned: Unlearned model representations, shape (N, D_unl)
            Y: Binary labels (1=forget, 0=retain), shape (N,)
            beta: Lagrangian multiplier for consistency constraint
            epochs: Number of optimization steps
            verbose: Print progress
            
        Returns:
            Dictionary with Residual Knowledge, Unlearned Knowledge, etc.
        """
        self.f1.train()
        self.f2.train()
        
        Z_b = torch.tensor(Z_base, dtype=torch.float32).to(self.device)
        Z_u = torch.tensor(Z_unlearned, dtype=torch.float32).to(self.device)
        Y_t = torch.tensor(Y, dtype=torch.float32).unsqueeze(1).to(self.device)
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            logits_1 = self.f1(Z_b)
            logits_2 = self.f2(Z_u)
            
            # Prediction losses
            loss_1 = self.bce(logits_1, Y_t)
            loss_2 = self.bce(logits_2, Y_t)
            
            # Consistency constraint: force probes to agree
            probs_1 = torch.sigmoid(logits_1)
            probs_2 = torch.sigmoid(logits_2)
            dist_loss = torch.mean(torch.abs(probs_1 - probs_2))
            
            # Lagrangian loss (Eq 12)
            total_loss = 0.5 * (loss_1 + loss_2) + beta * dist_loss
            
            total_loss.backward()
            self.optimizer.step()
            
            if verbose and (epoch + 1) % 200 == 0:
                metrics = self._compute_metrics(Z_b, Z_u, Y_t)
                print(f"  RINE epoch {epoch+1}/{epochs}: "
                      f"I∩={metrics['Residual Knowledge']:.4f} bits, "
                      f"I^B_uniq={metrics['Unlearned Knowledge']:.4f} bits, "
                      f"loss={total_loss.item():.4f}")
        
        return self._compute_metrics(Z_b, Z_u, Y_t)
    
    def _compute_metrics(
        self,
        Z_b: torch.Tensor,
        Z_u: torch.Tensor,
        Y: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute PID metrics from trained probes."""
        self.f1.eval()
        self.f2.eval()
        
        with torch.no_grad():
            logits_1 = self.f1(Z_b)
            logits_2 = self.f2(Z_u)
            
            ce_1 = self.bce(logits_1, Y).item()
            ce_2 = self.bce(logits_2, Y).item()
            
            # Average cross-entropy under consistency constraint
            L_cap = 0.5 * (ce_1 + ce_2)
        
        # H(Y): binary entropy of labels (nats)
        p = Y.mean().item()
        if 0 < p < 1:
            H_Y = -(p * np.log(p) + (1 - p) * np.log(1 - p))
        else:
            H_Y = 0.0
        
        # Convert nats → bits
        log2 = np.log(2)
        
        I_cap = (H_Y - L_cap) / log2         # Residual Knowledge (I∩)
        I_base = (H_Y - ce_1) / log2          # Total info in base model
        I_unlearned = (H_Y - ce_2) / log2     # Total info in unlearned model
        I_uniq_B = I_base - I_cap             # Unlearned Knowledge
        
        return {
            "Residual Knowledge": max(0.0, I_cap),
            "Unlearned Knowledge": max(0.0, I_uniq_B),
            "Total Base Info": I_base,
            "Total Unlearned Info": I_unlearned,
            "logits_base": logits_1.cpu().numpy(),
            "logits_unlearned": logits_2.cpu().numpy(),
        }
    
    def compute_risk_scores(
        self,
        metrics: Dict,
        Y: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute per-sample risk scores for inference-time abstention.
        
        Risk = Average_Prob × Agreement
        High risk on forget samples → residual memory leak.
        
        Args:
            metrics: Output from train_estimator
            Y: Binary labels (1=forget, 0=retain)
            
        Returns:
            Dictionary with risk scores and summary statistics
        """
        logits_b = metrics['logits_base'].flatten()
        logits_u = metrics['logits_unlearned'].flatten()
        
        p1 = 1.0 / (1.0 + np.exp(-logits_b))  # sigmoid
        p2 = 1.0 / (1.0 + np.exp(-logits_u))
        
        # Risk = average probability × agreement
        risk_scores = 0.5 * (p1 + p2) * (1.0 - np.abs(p1 - p2))
        
        forget_mask = (Y == 1)
        retain_mask = (Y == 0)
        
        return {
            "risk_scores": risk_scores,
            "avg_risk_forget": float(risk_scores[forget_mask].mean()) if forget_mask.any() else 0.0,
            "avg_risk_retain": float(risk_scores[retain_mask].mean()) if retain_mask.any() else 0.0,
            "max_risk_forget": float(risk_scores[forget_mask].max()) if forget_mask.any() else 0.0,
        }
