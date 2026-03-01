"""
Stress-Test Protocol for Machine Unlearning Methods.

For each unlearning method, constructs three forget sets of equal size:
    1. Random — baseline (uniform random samples)
    2. Low-I∩ (Easy) — samples easy to forget
    3. High-I∩ (Hard) — samples hard to forget

Then evaluates robustness across:
    - Retain-set accuracy
    - Forget-set accuracy
    - Parameter distance to retrain-from-scratch
    - Representation shift
    - MIA detection rate
    - I∩ score

Hypothesis: If a method is robust, performance should remain stable
across all three sets. Significant degradation on the hard set 
indicates residual memory and instability.

Usage:
    python experiments/run_stress_test.py --dataset cifar10 \
        --checkpoint checkpoints/cifar10_resnet18_original.pt

This script has TWO phases:
    Phase A: Difficulty estimation (compute I∩ per sample)
    Phase B: Stress testing (run unlearning on easy/hard/random splits)
"""

import argparse
import torch
import numpy as np
import random
import json
import copy
import csv
import time
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore', category=UserWarning)

import sys
sys.path.append(str(Path(__file__).parent.parent))

from torch.utils.data import Subset, DataLoader

from src.config import DataConfig, ModelConfig
from src.data.dataset import load_dataset
from src.models.classifier import create_model
from src.unlearning import (
    FineTuneUnlearner,
    GradientAscentUnlearner,
    SCRUBUnlearner,
    IWEUPv2Unlearner,
    RetrainUnlearner,
    BadTeacherUnlearner,
    AmnesiacUnlearner,
    SalUnUnlearner,
)
from src.auditing.rine import RINE
from src.auditing.activation_extractor import (
    ResNetActivationExtractor,
    extract_for_audit,
)
from src.auditing.difficulty import (
    compute_per_subset_difficulty,
    create_difficulty_splits,
    compute_parameter_distance,
    compute_representation_shift,
)
from src.evaluation.mia import shadow_model_mia


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_num_classes(dataset: str) -> int:
    if dataset == 'cifar100':
        return 100
    return 10


def get_unlearner(method, model, device, epochs=15, lr=0.001):
    """Create unlearner by method name."""
    if method == 'finetune':
        return FineTuneUnlearner(model, device=device, epochs=epochs, lr=lr)
    elif method == 'gradient_ascent':
        return GradientAscentUnlearner(model, device=device, epochs=epochs, lr=lr * 0.5)
    elif method == 'scrub':
        return SCRUBUnlearner(model, device=device, epochs=epochs, lr=lr)
    elif method == 'iweup_v2':
        return IWEUPv2Unlearner(model, device=device, epochs=epochs, lr=lr)
    elif method == 'retrain':
        return RetrainUnlearner(model, device=device, epochs=100, lr=0.1)
    elif method == 'bad_teacher':
        return BadTeacherUnlearner(model, device=device, epochs=epochs, lr=lr)
    elif method == 'amnesiac':
        return AmnesiacUnlearner(model, device=device, epochs=epochs, lr=lr)
    elif method == 'salun':
        return SalUnUnlearner(model, device=device, epochs=epochs, lr=lr)
    else:
        raise ValueError(f"Unknown method: {method}")


def evaluate_model(model, dataloader, device):
    """Compute accuracy on a dataloader."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100.0 * correct / total


def phase_a_difficulty_estimation(
    base_model,
    dataset_obj,
    device: str,
    seed: int,
    layer: str = 'avgpool',
    n_subsets: int = 20,
    subset_size: int = 100,
):
    """
    Phase A: Compute per-sample I∩ difficulty scores.
    
    Uses a "reference unlearning" (gradient ascent, fast) to get
    initial difficulty estimates. The key insight: if a sample has
    high I∩ even when GA is applied, it's inherently hard to forget.
    """
    set_seed(seed)
    
    print("\n" + "=" * 60)
    print("PHASE A: DIFFICULTY ESTIMATION")
    print("=" * 60)
    
    forget_loader = dataset_obj.get_forget_loader(128)
    retain_loader = dataset_obj.get_retain_loader(128)
    
    # Get the underlying forget dataset for subset sampling
    forget_dataset = dataset_obj.forget_set
    
    # Run a quick reference unlearning (GA, 5 epochs)
    print("  Running reference unlearning (Gradient Ascent, 5 epochs)...")
    ref_model = copy.deepcopy(base_model)
    ref_unlearner = GradientAscentUnlearner(
        ref_model, device=device, epochs=5, lr=0.0005
    )
    ref_unlearned = ref_unlearner.unlearn(forget_loader, retain_loader)
    
    # Compute per-subset difficulty
    print(f"  Computing difficulty across {n_subsets} subsets (size={subset_size})...")
    sorted_indices, per_sample_scores = compute_per_subset_difficulty(
        base_model=base_model,
        unlearned_model=ref_unlearned,
        forget_dataset=forget_dataset,
        retain_loader=retain_loader,
        layer=layer,
        device=device,
        n_subsets=n_subsets,
        subset_size=subset_size,
        rine_epochs=300,
        seed=seed,
    )
    
    print(f"  Difficulty range: [{per_sample_scores.min():.4f}, "
          f"{per_sample_scores.max():.4f}] bits")
    print(f"  Mean difficulty: {per_sample_scores.mean():.4f} bits")
    
    return forget_dataset, per_sample_scores


def phase_b_stress_test(
    base_model,
    dataset_obj,
    forget_dataset,
    per_sample_scores: np.ndarray,
    methods: list,
    device: str,
    seed: int,
    unlearn_epochs: int,
    unlearn_lr: float,
    split_size: int,
    layer: str = 'avgpool',
    rine_epochs: int = 500,
    rine_beta: float = 5.0,
):
    """
    Phase B: Run stress test across easy/hard/random splits.
    """
    print("\n" + "=" * 60)
    print("PHASE B: STRESS TESTING")
    print("=" * 60)
    
    # Create difficulty-based splits
    splits = create_difficulty_splits(
        forget_dataset, per_sample_scores, split_size, seed=seed
    )
    
    print(f"  Split size: {split_size} samples each")
    print(f"  Easy I∩ range: [{per_sample_scores[splits['easy']].min():.4f}, "
          f"{per_sample_scores[splits['easy']].max():.4f}]")
    print(f"  Hard I∩ range: [{per_sample_scores[splits['hard']].min():.4f}, "
          f"{per_sample_scores[splits['hard']].max():.4f}]")
    
    test_loader = dataset_obj.get_test_loader(128)
    retain_loader = dataset_obj.get_retain_loader(128)
    
    all_results = []
    
    # Compute retrained model for parameter distance reference
    print("\n  Computing retrain gold standard...")
    full_forget_loader = dataset_obj.get_forget_loader(128)
    retrain_model = copy.deepcopy(base_model)
    retrain_unlearner = RetrainUnlearner(retrain_model, device=device, epochs=100, lr=0.1)
    retrained_model = retrain_unlearner.unlearn(
        full_forget_loader, retain_loader, test_loader=test_loader
    )
    
    # Extract retrained representations on full forget set
    retrain_extractor = ResNetActivationExtractor(retrained_model, device)
    Z_retrained_full, _ = retrain_extractor.extract(full_forget_loader, layer)
    
    for split_name, split_indices in splits.items():
        print(f"\n  --- Split: {split_name.upper()} ({len(split_indices)} samples) ---")
        
        # Create dataloader for this split of the forget set
        split_forget_dataset = Subset(forget_dataset, split_indices)
        split_forget_loader = DataLoader(
            split_forget_dataset, batch_size=128, shuffle=False
        )
        
        for method in methods:
            if method == 'retrain':
                continue  # Retrain is reference, not tested
            
            set_seed(seed)
            print(f"    Testing: {method} on {split_name}...", end=" ")
            
            model_copy = copy.deepcopy(base_model)
            unlearner = get_unlearner(
                method, model_copy, device, unlearn_epochs, unlearn_lr
            )
            
            start = time.time()
            if method == 'iweup_v2':
                unlearned_model = unlearner.unlearn(
                    split_forget_loader, retain_loader, test_loader=test_loader
                )
            else:
                unlearned_model = unlearner.unlearn(
                    split_forget_loader, retain_loader
                )
            elapsed = time.time() - start
            
            # Evaluate metrics
            forget_acc = evaluate_model(unlearned_model, split_forget_loader, device)
            test_acc = evaluate_model(unlearned_model, test_loader, device)
            retain_acc = evaluate_model(unlearned_model, retain_loader, device)
            
            # Parameter distance to retrain
            param_dist = compute_parameter_distance(unlearned_model, retrained_model)
            
            # Representation shift
            unl_extractor = ResNetActivationExtractor(unlearned_model, device)
            Z_unl_split, _ = unl_extractor.extract(split_forget_loader, layer)
            Z_retrained_split = Z_retrained_full[split_indices]
            repr_shift = compute_representation_shift(Z_unl_split, Z_retrained_split)
            
            # RINE audit on this split
            Z_base_split, Z_unl_audit, Y_audit = extract_for_audit(
                base_model, unlearned_model,
                split_forget_loader, retain_loader,
                layer=layer, device=device,
            )
            dim = Z_base_split.shape[1]
            rine = RINE(dim, dim, device=device)
            rine_metrics = rine.train_estimator(
                Z_base_split, Z_unl_audit, Y_audit,
                beta=rine_beta, epochs=rine_epochs, verbose=False,
            )
            
            # MIA
            try:
                mia_result = shadow_model_mia(
                    unlearned_model, split_forget_loader, retain_loader,
                    test_loader, device
                )
                mia_score = mia_result.get('forget_member_ratio', -1)
            except Exception:
                mia_score = -1.0
            
            result = {
                'method': method,
                'split': split_name,
                'seed': seed,
                'forget_acc': forget_acc,
                'test_acc': test_acc,
                'retain_acc': retain_acc,
                'I_cap': rine_metrics['Residual Knowledge'],
                'param_distance': param_dist,
                'repr_shift': repr_shift,
                'mia': mia_score,
                'time': elapsed,
            }
            all_results.append(result)
            
            print(f"I∩={result['I_cap']:.4f} | TestAcc={test_acc:.1f}% | "
                  f"MIA={mia_score:.3f} | ParamDist={param_dist:.2f}")
    
    return all_results


def print_stress_table(all_results):
    """Print stress test results grouped by method."""
    print("\n" + "=" * 120)
    print("STRESS TEST RESULTS — ROBUSTNESS ACROSS DIFFICULTY SPLITS")
    print("=" * 120)
    
    methods = sorted(set(r['method'] for r in all_results))
    splits = ['easy', 'random', 'hard']
    
    print(f"{'Method':<16} {'Split':<8} {'I∩':>8} {'ForgetAcc':>10} "
          f"{'TestAcc':>10} {'RetainAcc':>10} {'MIA':>8} "
          f"{'ParamDist':>10} {'ReprShift':>10}")
    print("-" * 120)
    
    for method in methods:
        for split in splits:
            results = [r for r in all_results 
                      if r['method'] == method and r['split'] == split]
            if not results:
                continue
            r = results[0]
            print(f"{method:<16} {split:<8} {r['I_cap']:>8.4f} "
                  f"{r['forget_acc']:>9.1f}% {r['test_acc']:>9.1f}% "
                  f"{r['retain_acc']:>9.1f}% {r['mia']:>8.3f} "
                  f"{r['param_distance']:>10.2f} {r['repr_shift']:>10.4f}")
        print("-" * 120)
    
    # Robustness analysis
    print("\nROBUSTNESS ANALYSIS (Hard-Easy Delta):")
    print(f"{'Method':<16} {'ΔI∩':>8} {'ΔTestAcc':>10} {'ΔMIA':>8} "
          f"{'Verdict':>12}")
    print("-" * 60)
    
    for method in methods:
        easy = [r for r in all_results 
                if r['method'] == method and r['split'] == 'easy']
        hard = [r for r in all_results 
                if r['method'] == method and r['split'] == 'hard']
        
        if not easy or not hard:
            continue
        
        delta_icap = hard[0]['I_cap'] - easy[0]['I_cap']
        delta_test = hard[0]['test_acc'] - easy[0]['test_acc']
        delta_mia = hard[0]['mia'] - easy[0]['mia']
        
        if abs(delta_icap) < 0.05 and abs(delta_test) < 2.0:
            verdict = "✓ ROBUST"
        elif abs(delta_icap) < 0.1:
            verdict = "~ MODERATE"
        else:
            verdict = "✗ FRAGILE"
        
        print(f"{method:<16} {delta_icap:>+8.4f} {delta_test:>+9.1f}% "
              f"{delta_mia:>+8.3f} {verdict:>12}")


def main():
    parser = argparse.ArgumentParser(
        description='Stress-Test Protocol for Unlearning Methods'
    )
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['cifar10', 'cifar100'])
    parser.add_argument('--forget_class', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--methods', type=str, nargs='+',
                       default=['finetune', 'gradient_ascent', 'scrub',
                                'bad_teacher', 'amnesiac', 'salun', 'iweup_v2'])
    parser.add_argument('--checkpoint', type=str,
                       default='checkpoints/cifar10_resnet18_original.pt')
    parser.add_argument('--unlearn_epochs', type=int, default=15)
    parser.add_argument('--unlearn_lr', type=float, default=0.001)
    parser.add_argument('--split_size', type=int, default=500,
                       help='Number of samples in each difficulty split')
    parser.add_argument('--layer', type=str, default='avgpool')
    parser.add_argument('--output_dir', type=str, default='results/stress_test')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("UNLEARNING STRESS-TEST PROTOCOL")
    print("=" * 70)
    print(f"  Dataset: {args.dataset}")
    print(f"  Forget class: {args.forget_class}")
    print(f"  Methods: {args.methods}")
    print(f"  Split size: {args.split_size}")
    
    set_seed(args.seed)
    num_classes = get_num_classes(args.dataset)
    
    # Load dataset using project's API
    data_config = DataConfig(
        name=args.dataset,
        forget_classes=[args.forget_class],
        forget_type='class',
    )
    dataset_obj = load_dataset(data_config, seed=args.seed)
    
    # Load base model
    model_config = ModelConfig(
        architecture='resnet18', num_classes=num_classes
    )
    base_model = create_model(model_config)
    base_model = base_model.to(args.device)
    
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    base_model.load_state_dict(checkpoint['model_state_dict'])
    base_model.eval()
    
    # Phase A: Difficulty estimation
    forget_dataset, per_sample_scores = phase_a_difficulty_estimation(
        base_model=base_model,
        dataset_obj=dataset_obj,
        device=args.device,
        seed=args.seed,
        layer=args.layer,
    )
    
    # Save difficulty scores
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    np.save(
        f"{args.output_dir}/difficulty_scores_class{args.forget_class}.npy",
        per_sample_scores
    )
    
    # Phase B: Stress test
    all_results = phase_b_stress_test(
        base_model=base_model,
        dataset_obj=dataset_obj,
        forget_dataset=forget_dataset,
        per_sample_scores=per_sample_scores,
        methods=args.methods,
        device=args.device,
        seed=args.seed,
        unlearn_epochs=args.unlearn_epochs,
        unlearn_lr=args.unlearn_lr,
        split_size=args.split_size,
        layer=args.layer,
    )
    
    # Print results
    print_stress_table(all_results)
    
    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = f"{args.output_dir}/stress_test_{timestamp}.csv"
    keys = all_results[0].keys()
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(all_results)
    
    json_path = f"{args.output_dir}/stress_test_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {csv_path}")


if __name__ == '__main__':
    main()
