"""
Evaluation metrics for machine unlearning.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np


@dataclass
class UnlearningMetrics:
    """Container for unlearning evaluation metrics."""
    
    # Accuracy metrics
    forget_accuracy: float = 0.0
    retain_accuracy: float = 0.0
    test_accuracy: float = 0.0
    
    # Loss metrics
    forget_loss: float = 0.0
    retain_loss: float = 0.0
    test_loss: float = 0.0
    
    # MIA metrics
    mia_auc: float = 0.5  # AUC-ROC for membership inference
    mia_accuracy: float = 0.5  # Accuracy of MIA
    
    # Unlearning specific
    unlearning_accuracy: float = 0.0  # How much accuracy dropped on forget set
    remaining_accuracy: float = 0.0  # How much accuracy retained on retain set
    
    # Time metrics
    unlearning_time: float = 0.0
    
    # Certificate metrics (for IGTU)
    residual_influence_mean: Optional[float] = None
    residual_influence_bound: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'forget_accuracy': self.forget_accuracy,
            'retain_accuracy': self.retain_accuracy,
            'test_accuracy': self.test_accuracy,
            'forget_loss': self.forget_loss,
            'retain_loss': self.retain_loss,
            'test_loss': self.test_loss,
            'mia_auc': self.mia_auc,
            'mia_accuracy': self.mia_accuracy,
            'unlearning_accuracy': self.unlearning_accuracy,
            'remaining_accuracy': self.remaining_accuracy,
            'unlearning_time': self.unlearning_time,
            'residual_influence_mean': self.residual_influence_mean,
            'residual_influence_bound': self.residual_influence_bound,
        }


@torch.no_grad()
def compute_accuracy(
    model: nn.Module,
    loader: DataLoader,
    device: str = 'cuda'
) -> Tuple[float, float]:
    """
    Compute accuracy and loss on a dataset.
    
    Args:
        model: Neural network model
        loader: Data loader
        device: Computation device
        
    Returns:
        Tuple of (accuracy, average_loss)
    """
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / total
    
    return accuracy, avg_loss


@torch.no_grad()
def compute_confidences(
    model: nn.Module,
    loader: DataLoader,
    device: str = 'cuda'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute prediction confidences for MIA.
    
    Args:
        model: Neural network model
        loader: Data loader
        device: Computation device
        
    Returns:
        Tuple of (confidences, predictions, targets)
    """
    model.eval()
    all_confidences = []
    all_predictions = []
    all_targets = []
    
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        
        # Get softmax probabilities
        probs = torch.softmax(outputs, dim=1)
        
        # Get confidence (max probability)
        confidences, predictions = probs.max(dim=1)
        
        all_confidences.append(confidences.cpu().numpy())
        all_predictions.append(predictions.cpu().numpy())
        all_targets.append(targets.cpu().numpy())
    
    return (
        np.concatenate(all_confidences),
        np.concatenate(all_predictions),
        np.concatenate(all_targets)
    )


@torch.no_grad()
def compute_losses(
    model: nn.Module,
    loader: DataLoader,
    device: str = 'cuda'
) -> np.ndarray:
    """
    Compute per-sample losses for MIA.
    
    Args:
        model: Neural network model
        loader: Data loader
        device: Computation device
        
    Returns:
        Array of per-sample losses
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='none')
    all_losses = []
    
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        losses = criterion(outputs, targets)
        all_losses.append(losses.cpu().numpy())
    
    return np.concatenate(all_losses)


def compute_unlearning_metrics(
    model: nn.Module,
    original_model: nn.Module,
    forget_loader: DataLoader,
    retain_loader: DataLoader,
    test_loader: DataLoader,
    device: str = 'cuda',
    mia_attack: Optional['MembershipInferenceAttack'] = None,
    certificate: Optional[Dict] = None,
) -> UnlearningMetrics:
    """
    Compute comprehensive unlearning evaluation metrics.
    
    Args:
        model: Unlearned model
        original_model: Original model (before unlearning)
        forget_loader: Forget set data loader
        retain_loader: Retain set data loader
        test_loader: Test set data loader
        device: Computation device
        mia_attack: Optional MIA attack for privacy evaluation
        certificate: Optional certificate from IGTU
        
    Returns:
        UnlearningMetrics object with all metrics
    """
    metrics = UnlearningMetrics()
    
    # Compute accuracy metrics
    metrics.forget_accuracy, metrics.forget_loss = compute_accuracy(
        model, forget_loader, device
    )
    metrics.retain_accuracy, metrics.retain_loss = compute_accuracy(
        model, retain_loader, device
    )
    metrics.test_accuracy, metrics.test_loss = compute_accuracy(
        model, test_loader, device
    )
    
    # Compute original model metrics for comparison
    orig_forget_acc, _ = compute_accuracy(original_model, forget_loader, device)
    orig_retain_acc, _ = compute_accuracy(original_model, retain_loader, device)
    
    # Unlearning accuracy: how much accuracy dropped on forget set
    # Higher is better (more forgetting)
    metrics.unlearning_accuracy = orig_forget_acc - metrics.forget_accuracy
    
    # Remaining accuracy: how much accuracy is retained on retain set
    # Higher is better (less catastrophic forgetting)
    metrics.remaining_accuracy = metrics.retain_accuracy / orig_retain_acc * 100
    
    # MIA metrics
    if mia_attack is not None:
        mia_results = mia_attack.attack(model, forget_loader)
        metrics.mia_auc = mia_results['auc']
        metrics.mia_accuracy = mia_results['accuracy']
    
    # Certificate metrics
    if certificate is not None:
        metrics.residual_influence_mean = certificate.get('mean_residual')
        metrics.residual_influence_bound = certificate.get('bound_95')
    
    return metrics


def compare_to_retrained(
    unlearned_model: nn.Module,
    retrained_model: nn.Module,
    test_loader: DataLoader,
    device: str = 'cuda',
) -> Dict[str, float]:
    """
    Compare unlearned model to gold-standard retrained model.
    
    Args:
        unlearned_model: Model after unlearning
        retrained_model: Model retrained from scratch without forget data
        test_loader: Test set data loader
        device: Computation device
        
    Returns:
        Dictionary with comparison metrics
    """
    unlearned_acc, unlearned_loss = compute_accuracy(
        unlearned_model, test_loader, device
    )
    retrained_acc, retrained_loss = compute_accuracy(
        retrained_model, test_loader, device
    )
    
    # Compare output distributions
    unlearned_model.eval()
    retrained_model.eval()
    
    kl_divergences = []
    
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            
            unlearned_probs = torch.softmax(unlearned_model(inputs), dim=1)
            retrained_probs = torch.softmax(retrained_model(inputs), dim=1)
            
            # KL divergence between output distributions
            kl = torch.sum(
                retrained_probs * (torch.log(retrained_probs + 1e-10) - 
                                   torch.log(unlearned_probs + 1e-10)),
                dim=1
            )
            kl_divergences.extend(kl.cpu().numpy())
    
    return {
        'unlearned_accuracy': unlearned_acc,
        'retrained_accuracy': retrained_acc,
        'accuracy_gap': retrained_acc - unlearned_acc,
        'unlearned_loss': unlearned_loss,
        'retrained_loss': retrained_loss,
        'loss_gap': unlearned_loss - retrained_loss,
        'mean_kl_divergence': float(np.mean(kl_divergences)),
        'max_kl_divergence': float(np.max(kl_divergences)),
    }
