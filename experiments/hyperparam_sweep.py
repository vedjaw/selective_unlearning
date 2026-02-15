"""
Phase 1 Hyperparameter Sweep: Find optimal MIA/accuracy trade-off.

Tests multiple weight configurations for IWEUP v2 to find
the Pareto-optimal point where MIA < 0.15 AND Test Acc > 87%.

Usage:
    python experiments/hyperparam_sweep.py \
        --checkpoint checkpoints/cifar10_resnet18_original.pt \
        --forget_class 0 --seed 42
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
import random
import json
import csv
import time
from pathlib import Path
from datetime import datetime
from itertools import product

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.config import DataConfig, ModelConfig
from src.data.dataset import load_dataset
from src.models.classifier import create_model
from src.unlearning.iweup_v2 import IWEUPv2Unlearner
from src.evaluation.metrics import compute_unlearning_metrics
from src.evaluation.mia import simple_mia_evaluation


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# Define search space
# Key hypothesis: retain_weight is too low relative to forget weights
SWEEP_CONFIGS = [
    # Name,  uniformity, entropy, mia, retain, temp, epochs, lr
    # --- Baseline (current v2.1) ---
    ("baseline",        5.0, 1.0, 1.0,  5.0,  1.5, 15, 0.001),
    
    # --- Increase retain weight (protect test accuracy) ---
    ("retain_10",       5.0, 1.0, 1.0, 10.0,  1.5, 15, 0.001),
    ("retain_15",       5.0, 1.0, 1.0, 15.0,  1.5, 15, 0.001),
    ("retain_20",       5.0, 1.0, 1.0, 20.0,  1.5, 15, 0.001),
    
    # --- Reduce forget aggressiveness ---
    ("mild_forget",     3.0, 0.5, 0.5,  5.0,  1.5, 15, 0.001),
    ("mild_retain15",   3.0, 0.5, 0.5, 15.0,  1.5, 15, 0.001),
    
    # --- Lower learning rate (gentler updates) ---
    ("low_lr",          5.0, 1.0, 1.0,  5.0,  1.5, 20, 0.0005),
    ("low_lr_retain10", 5.0, 1.0, 1.0, 10.0,  1.5, 20, 0.0005),
    
    # --- Temperature exploration ---
    ("temp_1.0",        5.0, 1.0, 1.0,  5.0,  1.0, 15, 0.001),
    ("temp_2.0",        5.0, 1.0, 1.0,  5.0,  2.0, 15, 0.001),
    
    # --- Balanced configs ---
    ("balanced_a",      3.0, 0.5, 1.0, 10.0,  1.5, 15, 0.001),
    ("balanced_b",      2.0, 0.5, 1.5, 15.0,  1.5, 15, 0.001),
    ("balanced_c",      4.0, 1.0, 0.5, 12.0,  1.5, 15, 0.001),
    
    # --- Fewer epochs (less damage) ---
    ("short_5ep",       5.0, 1.0, 1.0,  5.0,  1.5,  5, 0.001),
    ("short_8ep",       5.0, 1.0, 1.0,  5.0,  1.5,  8, 0.001),
    ("short_10ep",      5.0, 1.0, 1.0, 10.0,  1.5, 10, 0.001),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--forget_class', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default='results/sweep')
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = args.device
    num_classes = 10 if args.dataset == 'cifar10' else 100
    
    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    data_config = DataConfig(
        name=args.dataset,
        forget_classes=[args.forget_class],
        forget_type='class'
    )
    dataset_obj = load_dataset(data_config, seed=args.seed)
    
    batch_size = 128
    forget_loader = dataset_obj.get_forget_loader(batch_size)
    retain_loader = dataset_obj.get_retain_loader(batch_size)
    test_loader = dataset_obj.get_test_loader(batch_size)
    
    # Load checkpoint once
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Results storage
    results = []
    
    print("=" * 100)
    print(f"HYPERPARAMETER SWEEP - {len(SWEEP_CONFIGS)} configurations")
    print(f"Dataset: {args.dataset}, Forget class: {args.forget_class}, Seed: {args.seed}")
    print("=" * 100)
    print(f"\n{'Config':<20} {'Forget↓':>8} {'Retain↑':>8} {'Test↑':>8} {'MIA↓':>8} {'Time':>8}")
    print("-" * 70)
    
    for config in SWEEP_CONFIGS:
        name, uni_w, ent_w, mia_w, ret_w, temp, epochs, lr = config
        
        set_seed(args.seed)
        
        # Fresh model from checkpoint
        model = create_model(ModelConfig(architecture='resnet18', num_classes=num_classes))
        model = model.to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Create unlearner with these params
        unlearner = IWEUPv2Unlearner(
            model, device=device,
            epochs=epochs, lr=lr,
            uniformity_weight=uni_w,
            entropy_weight=ent_w,
            mia_weight=mia_w,
            retain_weight=ret_w,
            temperature=temp,
            verbose=False,
        )
        
        start = time.time()
        try:
            model = unlearner.unlearn(forget_loader, retain_loader, test_loader=test_loader)
        except Exception as e:
            print(f"{name:<20} {'FAILED':>8} - {str(e)[:40]}")
            continue
        elapsed = time.time() - start
        
        # Evaluate
        forget_metrics = unlearner.evaluate(forget_loader)
        retain_metrics = unlearner.evaluate(retain_loader)
        test_metrics = unlearner.evaluate(test_loader)
        mia_results = simple_mia_evaluation(model, forget_loader, retain_loader, device=device)
        mia_score = mia_results.get('mia_score', mia_results.get('forget_member_ratio', -1))
        
        result = {
            'name': name,
            'uniformity_weight': uni_w,
            'entropy_weight': ent_w,
            'mia_weight': mia_w,
            'retain_weight': ret_w,
            'temperature': temp,
            'epochs': epochs,
            'lr': lr,
            'forget_acc': forget_metrics['accuracy'],
            'retain_acc': retain_metrics['accuracy'],
            'test_acc': test_metrics['accuracy'],
            'mia_score': mia_score,
            'time': elapsed,
        }
        results.append(result)
        
        # Color code: green if target met
        target_met = (
            5.0 <= result['forget_acc'] <= 15.0 and
            result['test_acc'] >= 87.0 and
            result['mia_score'] <= 0.15
        )
        marker = " ✓ GOOD" if target_met else ""
        
        print(f"{name:<20} {result['forget_acc']:>7.2f}% {result['retain_acc']:>7.2f}% "
              f"{result['test_acc']:>7.2f}% {result['mia_score']:>7.4f} {elapsed:>7.1f}s{marker}")
    
    # Save results
    csv_path = output_dir / f"sweep_{args.dataset}_class{args.forget_class}_s{args.seed}.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    
    json_path = output_dir / f"sweep_{args.dataset}_class{args.forget_class}_s{args.seed}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Summary
    print(f"\n{'=' * 100}")
    print("PARETO ANALYSIS")
    print(f"{'=' * 100}")
    print(f"\nTarget: Forget ~10%, Test ≥ 87%, MIA ≤ 0.15")
    print()
    
    # Find configs meeting criteria
    good_configs = [r for r in results 
                    if 5.0 <= r['forget_acc'] <= 15.0 
                    and r['test_acc'] >= 85.0  # slightly relaxed to find candidates
                    and r['mia_score'] <= 0.20]
    
    if good_configs:
        # Sort by test accuracy (higher is better)
        good_configs.sort(key=lambda x: x['test_acc'], reverse=True)
        print("CANDIDATE CONFIGURATIONS (sorted by Test Acc):")
        print(f"{'Config':<20} {'Forget':>8} {'Test':>8} {'MIA':>8}")
        print("-" * 50)
        for r in good_configs:
            print(f"{r['name']:<20} {r['forget_acc']:>7.2f}% {r['test_acc']:>7.2f}% {r['mia_score']:>7.4f}")
    else:
        print("No configurations met all criteria.")
        print("\nBest by test accuracy (with reasonable forget/MIA):")
        # Show top 5 by test acc
        sorted_by_test = sorted(results, key=lambda x: x['test_acc'], reverse=True)
        for r in sorted_by_test[:5]:
            print(f"  {r['name']:<20} Forget={r['forget_acc']:.2f}% Test={r['test_acc']:.2f}% MIA={r['mia_score']:.4f}")
    
    print(f"\nResults saved to: {csv_path}")
    print(f"Results saved to: {json_path}")


if __name__ == '__main__':
    main()
