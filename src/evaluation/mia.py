"""
Membership Inference Attack (MIA) for unlearning evaluation.

Implements two proper evaluation approaches:
1. Shadow-model MIA: Train an attack classifier on member vs non-member
   confidence patterns, then evaluate on forget set.
2. Population-level MIA: Compare distributional statistics of forget
   set confidence to a reference population (test set).

For unlearning, the key metric is whether an attacker can distinguish
samples in the forget set from samples the model never saw.

Correct interpretation:
  MIA ~0.5 → attacker can't tell → good forgetting
  MIA >> 0.5 → attacker can tell sample was trained on → poor forgetting
  MIA << 0.5 → attacker can tell something is "off" → over-forgetting
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import numpy as np
from typing import Dict, Tuple, Optional, List


def _extract_confidence_features(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract per-sample confidence features from model predictions.
    
    Returns:
        features: Array of shape (n_samples, 4) with:
          [max_prob, entropy, correct_class_prob, loss]
        correctness: Binary array indicating correct predictions
    """
    model.eval()
    all_features = []
    all_correct = []
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            
            # Handle numerical issues
            logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
            
            probs = torch.softmax(logits, dim=1)
            
            # Feature 1: Max confidence
            max_prob, preds = probs.max(dim=1)
            
            # Feature 2: Prediction entropy (higher = less confident)
            log_probs = torch.log(probs + 1e-10)
            entropy = -(probs * log_probs).sum(dim=1)
            
            # Feature 3: True class probability
            true_class_prob = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            
            # Feature 4: Cross-entropy loss per sample
            loss_fn = nn.CrossEntropyLoss(reduction='none')
            per_sample_loss = loss_fn(logits, targets)
            per_sample_loss = torch.clamp(per_sample_loss, 0, 50)
            
            # Correctness
            correct = preds.eq(targets).float()
            
            batch_features = torch.stack([
                max_prob, entropy, true_class_prob, per_sample_loss
            ], dim=1)
            
            all_features.append(batch_features.cpu().numpy())
            all_correct.append(correct.cpu().numpy())
    
    features = np.vstack(all_features)
    correctness = np.concatenate(all_correct)
    
    # Clean up any NaN/Inf
    features = np.nan_to_num(features, nan=0.5, posinf=50.0, neginf=0.0)
    
    return features, correctness


def shadow_model_mia(
    model: nn.Module,
    forget_loader: DataLoader,
    retain_loader: DataLoader,
    test_loader: DataLoader,
    device: str = 'cuda',
) -> Dict[str, float]:
    """
    Proper shadow-model MIA evaluation for unlearning.
    
    Training phase:
      - Members = retain_loader samples (model trained on these)
      - Non-members = test_loader samples (model never trained on these)
      - Train logistic regression on their confidence features
    
    Evaluation phase:
      - Run the trained attacker on forget_loader samples
      - Good unlearning: forget samples look like non-members (MIA ~0.5)
    
    Returns:
        Dict with:
          'mia_accuracy': Attacker accuracy on forget set (0.5 = ideal)
          'mia_auc': AUC of attacker on forget set
          'forget_member_ratio': Fraction of forget samples classified as members
          'retain_member_ratio': Fraction of retain samples classified as members (sanity check)
          'test_member_ratio': Fraction of test samples classified as non-members (sanity check)
    """
    # === Step 1: Extract features ===
    retain_features, _ = _extract_confidence_features(model, retain_loader, device)
    test_features, _ = _extract_confidence_features(model, test_loader, device)
    forget_features, _ = _extract_confidence_features(model, forget_loader, device)
    
    # === Step 2: Balance member/non-member sets for training ===
    n_retain = len(retain_features)
    n_test = len(test_features)
    n_train = min(n_retain, n_test)
    
    # Subsample to balance
    rng = np.random.RandomState(42)
    retain_idx = rng.choice(n_retain, n_train, replace=False)
    test_idx = rng.choice(n_test, n_train, replace=False)
    
    X_train = np.vstack([retain_features[retain_idx], test_features[test_idx]])
    y_train = np.concatenate([np.ones(n_train), np.zeros(n_train)])
    
    # === Step 3: Train attack classifier ===
    attack_model = LogisticRegression(
        max_iter=2000, 
        random_state=42,
        C=1.0,
        solver='lbfgs'
    )
    attack_model.fit(X_train, y_train)
    
    # === Step 4: Evaluate on all sets ===
    # Forget set — the key metric
    forget_preds = attack_model.predict(forget_features)
    forget_proba = attack_model.predict_proba(forget_features)[:, 1]
    forget_member_ratio = forget_preds.mean()
    
    # Sanity check: retain should be classified as members
    retain_preds = attack_model.predict(retain_features)
    retain_member_ratio = retain_preds.mean()
    
    # Sanity check: test should be classified as non-members  
    test_preds = attack_model.predict(test_features)
    test_member_ratio = test_preds.mean()
    
    # === Step 5: Compute attack accuracy on forget set ===
    # If forget samples were truly unlearned, attacker should guess ~50%
    # We measure: how many forget samples does the attacker think are members?
    # Lower = better unlearning
    
    # AUC: how well can the attacker rank forget samples as members?
    # For forget set, "ground truth" is ambiguous post-unlearning
    # We use member_ratio as the primary metric
    
    # Cross-validated accuracy of the attack model itself (measures attack quality)
    try:
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        cv_scores = []
        for train_i, val_i in cv.split(X_train, y_train):
            cv_model = LogisticRegression(max_iter=2000, random_state=42)
            cv_model.fit(X_train[train_i], y_train[train_i])
            cv_scores.append(cv_model.score(X_train[val_i], y_train[val_i]))
        attack_quality = np.mean(cv_scores)
    except:
        attack_quality = 0.5
    
    return {
        'forget_member_ratio': float(forget_member_ratio),
        'retain_member_ratio': float(retain_member_ratio),
        'test_member_ratio': float(test_member_ratio),
        'attack_quality': float(attack_quality),
        'forget_mean_confidence': float(forget_proba.mean()),
    }


def population_mia(
    model: nn.Module,
    forget_loader: DataLoader,
    test_loader: DataLoader,
    device: str = 'cuda',
) -> Dict[str, float]:
    """
    Population-level MIA: compares distributions without training an attacker.
    
    Computes distributional distance between forget-set confidence and
    test-set confidence. If they're similar, forgetting was successful.
    
    This is more robust than a trained attacker for small sample sizes.
    """
    forget_features, forget_correct = _extract_confidence_features(
        model, forget_loader, device
    )
    test_features, test_correct = _extract_confidence_features(
        model, test_loader, device
    )
    
    # Compare max-confidence distributions
    forget_conf = forget_features[:, 0]  # max prob
    test_conf = test_features[:, 0]
    
    # KS statistic: how different are the distributions?
    from scipy.stats import ks_2samp
    ks_stat, ks_pval = ks_2samp(forget_conf, test_conf)
    
    # Mean confidence difference
    conf_diff = abs(float(forget_conf.mean()) - float(test_conf.mean()))
    
    # Entropy comparison
    forget_entropy = forget_features[:, 1].mean()
    test_entropy = test_features[:, 1].mean()
    
    # Correctness comparison
    forget_acc = forget_correct.mean()
    test_acc = test_correct.mean()
    
    return {
        'ks_statistic': float(ks_stat),
        'ks_pvalue': float(ks_pval),
        'confidence_gap': conf_diff,
        'forget_mean_conf': float(forget_conf.mean()),
        'test_mean_conf': float(test_conf.mean()),
        'forget_entropy': float(forget_entropy),
        'test_entropy': float(test_entropy),
        'forget_accuracy': float(forget_acc * 100),
        'test_accuracy': float(test_acc * 100),
    }


def comprehensive_mia_evaluation(
    model: nn.Module,
    forget_loader: DataLoader,
    retain_loader: DataLoader,
    test_loader: DataLoader,
    device: str = 'cuda',
) -> Dict[str, float]:
    """
    Full MIA evaluation combining shadow-model and population-level attacks.
    
    This is the recommended function for evaluating unlearning quality.
    
    Primary metric: forget_member_ratio from shadow-model MIA
      - ~0.5 = ideal (attacker can't distinguish)
      - >> 0.5 = poor forgetting (still looks like training data)
      - << 0.5 = over-forgetting (looks distinctively different from all data)
    
    Args:
        model: The unlearned model to evaluate
        forget_loader: Samples that should have been forgotten
        retain_loader: Samples still in training set
        test_loader: Held-out test samples (never in training)
        device: Computation device
    """
    # Shadow model MIA
    shadow_results = shadow_model_mia(
        model, forget_loader, retain_loader, test_loader, device
    )
    
    # Population-level MIA  
    pop_results = population_mia(model, forget_loader, test_loader, device)
    
    # Combine results
    results = {}
    for k, v in shadow_results.items():
        results[f'shadow_{k}'] = v
    for k, v in pop_results.items():
        results[f'pop_{k}'] = v
    
    # Primary metric: distance from ideal (0.5)
    member_ratio = shadow_results['forget_member_ratio']
    results['mia_score'] = member_ratio  # Primary metric
    results['mia_distance_from_ideal'] = abs(member_ratio - 0.5)
    
    return results


# Backward-compatible wrapper
def simple_mia_evaluation(
    model: nn.Module,
    forget_loader: DataLoader,
    retain_or_test_loader: DataLoader,
    device: str = 'cuda',
) -> Dict[str, float]:
    """
    Backward-compatible wrapper. Uses shadow MIA with retain=test as fallback.
    
    NOTE: For proper evaluation, use comprehensive_mia_evaluation instead.
    """
    results = shadow_model_mia(
        model, forget_loader, retain_or_test_loader, 
        retain_or_test_loader, device
    )
    return results
