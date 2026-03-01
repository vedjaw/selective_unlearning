"""
Difficulty Estimation using Residual Knowledge (I∩).

Computes per-sample or per-subset I∩ scores to rank forget set
samples by difficulty:
    - Low I∩ → easy-to-forget (model releases information readily)
    - High I∩ → hard-to-forget (model retains deep memorization)

This provides a theoretically grounded difficulty measure based
on information retention, not just classification accuracy.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset, TensorDataset
from typing import Dict, List, Tuple, Optional
import copy

from .rine import RINE
from .activation_extractor import ResNetActivationExtractor


def compute_per_subset_difficulty(
    base_model: nn.Module,
    unlearned_model: nn.Module,
    forget_dataset,
    retain_loader: DataLoader,
    layer: str = 'avgpool',
    device: str = 'cuda',
    n_subsets: int = 10,
    subset_size: int = 50,
    rine_epochs: int = 500,
    rine_beta: float = 5.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate difficulty via subset-level I∩ scoring.
    
    Splits the forget set into n_subsets random subsets and computes
    I∩ for each. Samples in high-I∩ subsets are "hard to forget".
    
    Args:
        base_model: Original trained model
        unlearned_model: Model after unlearning
        forget_dataset: Full forget dataset (indexable)
        retain_loader: DataLoader for retain set
        layer: ResNet layer for activations
        device: Computation device
        n_subsets: Number of subsets to evaluate
        subset_size: Samples per subset
        rine_epochs: RINE optimization steps
        rine_beta: Lagrangian multiplier
        seed: Random seed
        
    Returns:
        Tuple of (sample_indices, i_cap_scores):
        - sample_indices: indices into forget_dataset, sorted by I∩
        - i_cap_scores: per-sample I∩ scores (averaged over subsets 
          containing that sample)
    """
    np.random.seed(seed)
    
    n_forget = len(forget_dataset)
    per_sample_scores = np.zeros(n_forget)
    per_sample_counts = np.zeros(n_forget)
    
    # Extract retain activations once (shared across all subsets)
    base_extractor = ResNetActivationExtractor(base_model, device)
    unl_extractor = ResNetActivationExtractor(unlearned_model, device)
    
    Z_base_retain, _ = base_extractor.extract(retain_loader, layer)
    Z_unl_retain, _ = unl_extractor.extract(retain_loader, layer)
    
    # Subsample retain to match subset_size for balanced RINE
    retain_indices = np.random.choice(
        len(Z_base_retain), 
        size=min(subset_size, len(Z_base_retain)), 
        replace=False
    )
    Z_base_retain_sub = Z_base_retain[retain_indices]
    Z_unl_retain_sub = Z_unl_retain[retain_indices]
    
    for i in range(n_subsets):
        # Random subset of forget samples
        subset_indices = np.random.choice(
            n_forget, size=min(subset_size, n_forget), replace=False
        )
        
        # Create subset dataloader
        subset = Subset(forget_dataset, subset_indices)
        subset_loader = DataLoader(subset, batch_size=128, shuffle=False)
        
        # Extract activations for this subset
        Z_base_forget, _ = base_extractor.extract(subset_loader, layer)
        Z_unl_forget, _ = unl_extractor.extract(subset_loader, layer)
        
        # Combine with retain subset
        Z_base = np.concatenate([Z_base_forget, Z_base_retain_sub])
        Z_unl = np.concatenate([Z_unl_forget, Z_unl_retain_sub])
        Y = np.concatenate([
            np.ones(len(Z_base_forget)), 
            np.zeros(len(Z_base_retain_sub))
        ])
        
        # Run RINE
        dim = Z_base.shape[1]
        rine = RINE(dim, dim, device=device)
        metrics = rine.train_estimator(
            Z_base, Z_unl, Y, 
            beta=rine_beta, epochs=rine_epochs, verbose=False
        )
        
        # Assign I∩ score to all samples in this subset
        i_cap = metrics['Residual Knowledge']
        for idx in subset_indices:
            per_sample_scores[idx] += i_cap
            per_sample_counts[idx] += 1
    
    # Average scores
    mask = per_sample_counts > 0
    per_sample_scores[mask] /= per_sample_counts[mask]
    
    # Sort by difficulty (ascending I∩)
    sorted_indices = np.argsort(per_sample_scores)
    
    return sorted_indices, per_sample_scores


def create_difficulty_splits(
    forget_dataset,
    per_sample_scores: np.ndarray,
    split_size: int,
    seed: int = 42,
) -> Dict[str, List[int]]:
    """
    Create three forget sets: easy, hard, and random.
    
    Args:
        forget_dataset: Full forget dataset
        per_sample_scores: Per-sample I∩ scores from compute_per_subset_difficulty
        split_size: Number of samples in each split
        seed: Random seed for random split
        
    Returns:
        Dictionary with 'easy', 'hard', 'random' keys mapping to 
        lists of indices into forget_dataset
    """
    np.random.seed(seed)
    n = len(forget_dataset)
    
    sorted_indices = np.argsort(per_sample_scores)
    
    # Easy: lowest I∩ samples
    easy_indices = sorted_indices[:split_size].tolist()
    
    # Hard: highest I∩ samples
    hard_indices = sorted_indices[-split_size:].tolist()
    
    # Random: uniform random sample
    random_indices = np.random.choice(n, size=split_size, replace=False).tolist()
    
    return {
        'easy': easy_indices,
        'hard': hard_indices,
        'random': random_indices,
    }


def compute_parameter_distance(
    model_a: nn.Module,
    model_b: nn.Module,
) -> float:
    """
    Compute L2 parameter distance between two models.
    
    ||θ_a - θ_b||₂
    
    This measures how far the unlearned model has drifted from
    retrain-from-scratch in parameter space.
    """
    distance = 0.0
    for (name_a, param_a), (name_b, param_b) in zip(
        model_a.named_parameters(), model_b.named_parameters()
    ):
        assert name_a == name_b, f"Parameter mismatch: {name_a} vs {name_b}"
        distance += torch.sum((param_a - param_b) ** 2).item()
    
    return float(np.sqrt(distance))


def compute_representation_shift(
    Z_unlearned: np.ndarray,
    Z_retrained: np.ndarray,
) -> float:
    """
    Compute average L2 representation shift between unlearned and retrained.
    
    ||Z_unlearned - Z_retrained||₂ averaged over samples.
    """
    diff = Z_unlearned - Z_retrained
    per_sample_dist = np.linalg.norm(diff, axis=1)
    return float(per_sample_dist.mean())
