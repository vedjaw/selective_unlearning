"""
Information Decomposition Audit for Machine Unlearning.

Runs the RINE-based audit across all unlearning methods to compute:
- I∩ (Residual Knowledge) for each method
- I^B_uniq (Unlearned Knowledge) 
- Risk scores for inference-time abstention
- Comparison against retrain-from-scratch gold standard

Usage:
    python experiments/run_audit.py --dataset cifar10 \
        --checkpoint checkpoints/cifar10_resnet18_original.pt
"""

import argparse
import torch
import numpy as np
import random
import json
import time
import copy
import csv
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore', category=UserWarning)

import sys
sys.path.append(str(Path(__file__).parent.parent))

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
from src.auditing.activation_extractor import extract_for_audit


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


def run_audit_for_method(
    method: str,
    base_model,
    forget_loader,
    retain_loader,
    test_loader,
    device: str,
    seed: int,
    unlearn_epochs: int,
    unlearn_lr: float,
    rine_epochs: int = 1000,
    rine_beta: float = 5.0,
    layer: str = 'avgpool',
    cache_dir: str = None,
    verbose: bool = True,
):
    """
    Run a complete audit for one unlearning method.
    """
    set_seed(seed)
    
    # 1. Run unlearning
    model_copy = copy.deepcopy(base_model)
    unlearner = get_unlearner(method, model_copy, device, unlearn_epochs, unlearn_lr)
    
    start_time = time.time()
    if method in ('retrain', 'iweup_v2'):
        unlearned_model = unlearner.unlearn(
            forget_loader, retain_loader, test_loader=test_loader
        )
    else:
        unlearned_model = unlearner.unlearn(forget_loader, retain_loader)
    elapsed = time.time() - start_time
    
    if verbose:
        print(f"  [{method}] Unlearning done in {elapsed:.1f}s")
    
    # 2. Extract representations for RINE audit
    exp_name = f"{method}_s{seed}"
    Z_base, Z_unlearned, Y = extract_for_audit(
        base_model, unlearned_model,
        forget_loader, retain_loader,
        layer=layer, device=device,
        cache_dir=cache_dir, experiment_name=exp_name,
    )
    
    if verbose:
        print(f"  [{method}] Extracted {Z_base.shape[0]} representations "
              f"(dim={Z_base.shape[1]})")
    
    # 3. Run RINE audit
    dim = Z_base.shape[1]
    rine = RINE(dim, dim, device=device)
    
    metrics = rine.train_estimator(
        Z_base, Z_unlearned, Y,
        beta=rine_beta, epochs=rine_epochs,
        verbose=verbose,
    )
    
    # 4. Compute risk scores
    risk = rine.compute_risk_scores(metrics, Y)
    
    # 5. Standard accuracy evaluation
    unlearned_model.eval()
    
    # Forget accuracy
    correct_f = total_f = 0
    with torch.no_grad():
        for inputs, targets in forget_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = unlearned_model(inputs)
            _, predicted = outputs.max(1)
            total_f += targets.size(0)
            correct_f += predicted.eq(targets).sum().item()
    forget_acc = 100.0 * correct_f / total_f
    
    # Test accuracy
    correct_t = total_t = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = unlearned_model(inputs)
            _, predicted = outputs.max(1)
            total_t += targets.size(0)
            correct_t += predicted.eq(targets).sum().item()
    test_acc = 100.0 * correct_t / total_t
    
    results = {
        'method': method,
        'seed': seed,
        'I_cap': metrics['Residual Knowledge'],
        'I_uniq_B': metrics['Unlearned Knowledge'],
        'Total_Base_Info': metrics['Total Base Info'],
        'Total_Unlearned_Info': metrics['Total Unlearned Info'],
        'avg_risk_forget': risk['avg_risk_forget'],
        'avg_risk_retain': risk['avg_risk_retain'],
        'forget_acc': forget_acc,
        'test_acc': test_acc,
        'time': elapsed,
    }
    
    if verbose:
        print(f"  [{method}] I∩={results['I_cap']:.4f} | "
              f"I^B_uniq={results['I_uniq_B']:.4f} | "
              f"ForgetAcc={forget_acc:.1f}% | "
              f"TestAcc={test_acc:.1f}% | "
              f"Risk={risk['avg_risk_forget']:.4f}")
    
    return results


def print_audit_table(all_results):
    """Print formatted audit results."""
    print("\n" + "=" * 90)
    print("INFORMATION DECOMPOSITION AUDIT RESULTS")
    print("=" * 90)
    print(f"{'Method':<18} {'I∩ (Residual)':>14} {'I^B (Unlearned)':>16} "
          f"{'Risk(Forget)':>14} {'ForgetAcc':>10} {'TestAcc':>10} {'Time':>8}")
    print("-" * 90)
    
    for r in sorted(all_results, key=lambda x: x['I_cap']):
        print(f"{r['method']:<18} {r['I_cap']:>14.4f} {r['I_uniq_B']:>16.4f} "
              f"{r['avg_risk_forget']:>14.4f} {r['forget_acc']:>9.1f}% "
              f"{r['test_acc']:>9.1f}% {r['time']:>7.1f}s")
    
    print("=" * 90)
    print("Lower I∩ = better unlearning. Retrain should have I∩ ≈ 0.")


def save_results(all_results, output_dir):
    """Save audit results to CSV and JSON."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    csv_path = f"{output_dir}/audit_results_{timestamp}.csv"
    keys = all_results[0].keys()
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(all_results)
    
    json_path = f"{output_dir}/audit_results_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='Information Decomposition Audit')
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['cifar10', 'cifar100'])
    parser.add_argument('--forget_class', type=int, default=0,
                       help='Class to forget for class-level audit')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42])
    parser.add_argument('--methods', type=str, nargs='+',
                       default=['finetune', 'gradient_ascent', 'scrub',
                                'bad_teacher', 'amnesiac', 'salun',
                                'iweup_v2', 'retrain'])
    parser.add_argument('--checkpoint', type=str,
                       default='checkpoints/cifar10_resnet18_original.pt')
    parser.add_argument('--unlearn_epochs', type=int, default=15)
    parser.add_argument('--unlearn_lr', type=float, default=0.001)
    parser.add_argument('--rine_epochs', type=int, default=1000)
    parser.add_argument('--rine_beta', type=float, default=5.0)
    parser.add_argument('--layer', type=str, default='avgpool',
                       choices=['layer1', 'layer2', 'layer3', 'layer4', 'avgpool'])
    parser.add_argument('--output_dir', type=str, default='results/audit')
    parser.add_argument('--cache_dir', type=str, default='activations')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("INFORMATION DECOMPOSITION AUDIT")
    print("=" * 70)
    print(f"  Dataset: {args.dataset}")
    print(f"  Forget class: {args.forget_class}")
    print(f"  Methods: {args.methods}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Layer: {args.layer}")
    print(f"  RINE: {args.rine_epochs} epochs, β={args.rine_beta}")
    
    num_classes = get_num_classes(args.dataset)
    all_results = []
    
    for seed in args.seeds:
        set_seed(seed)
        
        print(f"\n--- Seed {seed} ---")
        
        # Load dataset using project's API
        data_config = DataConfig(
            name=args.dataset,
            forget_classes=[args.forget_class],
            forget_type='class',
        )
        dataset_obj = load_dataset(data_config, seed=seed)
        
        forget_loader = dataset_obj.get_forget_loader(128)
        retain_loader = dataset_obj.get_retain_loader(128)
        test_loader = dataset_obj.get_test_loader(128)
        
        # Load base model using project's API
        model_config = ModelConfig(
            architecture='resnet18', num_classes=num_classes
        )
        base_model = create_model(model_config)
        base_model = base_model.to(args.device)
        
        checkpoint = torch.load(args.checkpoint, map_location=args.device)
        base_model.load_state_dict(checkpoint['model_state_dict'])
        base_model.eval()
        
        # Run audit for each method
        for method in args.methods:
            print(f"\n  Auditing: {method}")
            
            result = run_audit_for_method(
                method=method,
                base_model=base_model,
                forget_loader=forget_loader,
                retain_loader=retain_loader,
                test_loader=test_loader,
                device=args.device,
                seed=seed,
                unlearn_epochs=args.unlearn_epochs,
                unlearn_lr=args.unlearn_lr,
                rine_epochs=args.rine_epochs,
                rine_beta=args.rine_beta,
                layer=args.layer,
                cache_dir=args.cache_dir,
            )
            
            all_results.append(result)
    
    print_audit_table(all_results)
    save_results(all_results, args.output_dir)


if __name__ == '__main__':
    main()
