"""
Phase 1 Diagnostic: Per-class accuracy analysis for IWEUP v2.

Compares original model vs unlearned model per-class to find
where the 7% test accuracy drop is coming from.

Usage:
    python experiments/diagnose_accuracy.py \
        --checkpoint checkpoints/cifar10_resnet18_original.pt \
        --forget_class 0 --seed 42
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
import random
from pathlib import Path
from collections import defaultdict

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.config import DataConfig, ModelConfig
from src.data.dataset import load_dataset
from src.models.classifier import create_model
from src.unlearning import IWEUPv2Unlearner


CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def per_class_accuracy(model, loader, device, num_classes=10):
    """Compute accuracy per class."""
    correct = defaultdict(int)
    total = defaultdict(int)
    confidences = defaultdict(list)
    
    model.eval()
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            max_conf, preds = probs.max(dim=1)
            
            for i in range(targets.size(0)):
                cls = targets[i].item()
                total[cls] += 1
                if preds[i].item() == cls:
                    correct[cls] += 1
                confidences[cls].append(max_conf[i].item())
    
    results = {}
    for cls in range(num_classes):
        acc = 100 * correct[cls] / total[cls] if total[cls] > 0 else 0
        avg_conf = np.mean(confidences[cls]) if confidences[cls] else 0
        results[cls] = {
            'accuracy': acc,
            'total': total[cls],
            'correct': correct[cls],
            'avg_confidence': avg_conf,
        }
    return results


def confidence_distribution(model, loader, device):
    """Get confidence stats for a loader."""
    all_confs = []
    all_losses = []
    model.eval()
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            max_conf = probs.max(dim=1)[0]
            losses = criterion(outputs, targets)
            
            all_confs.extend(max_conf.cpu().tolist())
            all_losses.extend(losses.cpu().tolist())
    
    return {
        'mean_conf': np.mean(all_confs),
        'std_conf': np.std(all_confs),
        'mean_loss': np.mean(all_losses),
        'std_loss': np.std(all_losses),
        'median_conf': np.median(all_confs),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--forget_class', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = args.device
    num_classes = 10 if args.dataset == 'cifar10' else 100
    
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
    
    # === ORIGINAL MODEL ===
    print("=" * 70)
    print("ORIGINAL MODEL (before unlearning)")
    print("=" * 70)
    
    orig_model = create_model(ModelConfig(architecture='resnet18', num_classes=num_classes))
    orig_model = orig_model.to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    orig_model.load_state_dict(checkpoint['model_state_dict'])
    
    orig_test_results = per_class_accuracy(orig_model, test_loader, device, num_classes)
    
    print(f"\n{'Class':<15} {'Acc':>8} {'Conf':>8} {'Count':>8}")
    print("-" * 45)
    orig_total_correct = 0
    orig_total = 0
    for cls in range(num_classes):
        r = orig_test_results[cls]
        label = CIFAR10_CLASSES[cls] if args.dataset == 'cifar10' and cls < 10 else f"class_{cls}"
        marker = " ← FORGET" if cls == args.forget_class else ""
        print(f"{label:<15} {r['accuracy']:>7.2f}% {r['avg_confidence']:>7.4f} {r['total']:>8}{marker}")
        orig_total_correct += r['correct']
        orig_total += r['total']
    print(f"\n{'OVERALL':<15} {100*orig_total_correct/orig_total:>7.2f}%")
    
    # === UNLEARN ===
    print(f"\n{'=' * 70}")
    print(f"UNLEARNING class {args.forget_class} with IWEUP v2")
    print(f"{'=' * 70}")
    
    # Fresh model
    model = create_model(ModelConfig(architecture='resnet18', num_classes=num_classes))
    model = model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    unlearner = IWEUPv2Unlearner(model, device=device, epochs=15, lr=0.001)
    model = unlearner.unlearn(forget_loader, retain_loader, test_loader=test_loader)
    
    # === UNLEARNED MODEL ===
    print(f"\n{'=' * 70}")
    print("UNLEARNED MODEL (after IWEUP v2)")
    print("=" * 70)
    
    unl_test_results = per_class_accuracy(model, test_loader, device, num_classes)
    
    print(f"\n{'Class':<15} {'Orig':>8} {'Unlearn':>8} {'Delta':>8} {'Conf':>8}")
    print("-" * 55)
    unl_total_correct = 0
    unl_total = 0
    for cls in range(num_classes):
        orig_r = orig_test_results[cls]
        unl_r = unl_test_results[cls]
        delta = unl_r['accuracy'] - orig_r['accuracy']
        marker = " ← FORGET" if cls == args.forget_class else ""
        label = CIFAR10_CLASSES[cls] if args.dataset == 'cifar10' and cls < 10 else f"class_{cls}"
        print(f"{label:<15} {orig_r['accuracy']:>7.2f}% {unl_r['accuracy']:>7.2f}% {delta:>+7.2f}% {unl_r['avg_confidence']:>7.4f}{marker}")
        unl_total_correct += unl_r['correct']
        unl_total += unl_r['total']
    
    print(f"\n{'OVERALL':<15} {100*orig_total_correct/orig_total:>7.2f}% {100*unl_total_correct/unl_total:>7.2f}% {100*(unl_total_correct-orig_total_correct)/orig_total:>+7.2f}%")
    
    # === CONFIDENCE ANALYSIS ===
    print(f"\n{'=' * 70}")
    print("CONFIDENCE DISTRIBUTION ANALYSIS")
    print("=" * 70)
    
    forget_conf = confidence_distribution(model, forget_loader, device)
    retain_conf = confidence_distribution(model, retain_loader, device)
    test_conf = confidence_distribution(model, test_loader, device)
    
    print(f"\n{'Set':<15} {'Mean Conf':>10} {'Std':>8} {'Mean Loss':>10} {'Median':>8}")
    print("-" * 55)
    print(f"{'Forget':<15} {forget_conf['mean_conf']:>10.4f} {forget_conf['std_conf']:>8.4f} {forget_conf['mean_loss']:>10.4f} {forget_conf['median_conf']:>8.4f}")
    print(f"{'Retain':<15} {retain_conf['mean_conf']:>10.4f} {retain_conf['std_conf']:>8.4f} {retain_conf['mean_loss']:>10.4f} {retain_conf['median_conf']:>8.4f}")
    print(f"{'Test':<15} {test_conf['mean_conf']:>10.4f} {test_conf['std_conf']:>8.4f} {test_conf['mean_loss']:>10.4f} {test_conf['median_conf']:>8.4f}")
    
    # === DIAGNOSIS ===
    print(f"\n{'=' * 70}")
    print("DIAGNOSIS SUMMARY")
    print("=" * 70)
    
    # Find most affected non-forget classes
    deltas = []
    for cls in range(num_classes):
        if cls == args.forget_class:
            continue
        delta = unl_test_results[cls]['accuracy'] - orig_test_results[cls]['accuracy']
        label = CIFAR10_CLASSES[cls] if args.dataset == 'cifar10' and cls < 10 else f"class_{cls}"
        deltas.append((label, cls, delta))
    
    deltas.sort(key=lambda x: x[2])
    
    print("\nMost affected NON-FORGET classes (sorted by accuracy drop):")
    for label, cls, delta in deltas[:5]:
        print(f"  {label}: {delta:+.2f}%")
    
    # Check if drop is concentrated or spread
    big_drops = [d for d in deltas if d[2] < -3.0]
    if len(big_drops) >= 3:
        print(f"\n⚠️  Drop is SPREAD across {len(big_drops)} classes → feature representation damage")
        print("   FIX: Selective layer freezing or stronger retain regularization")
    elif len(big_drops) >= 1:
        print(f"\n⚠️  Drop CONCENTRATED in {len(big_drops)} class(es) → likely similar to forget class")
        print("   FIX: Reduce uniformity weight or add class-specific protection")
    else:
        print("\n✅ No significant drops in non-forget classes")


if __name__ == '__main__':
    main()
