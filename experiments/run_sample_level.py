"""
Sample-level unlearning experiments.

Tests unlearning of random samples (simulating "one user's data removal")
instead of entire classes. This is a different and arguably harder setting
because the forget set spans all classes.

Key differences from class-level:
- forget_type='random' with forget_ratio (e.g., 1%, 5%, 10%)
- No NF-Test needed (forget samples span all classes)
- MIA is the primary metric (can the attacker tell which samples were forgotten?)
- Forget accuracy is less meaningful (samples span all classes, so accuracy 
  should stay high on those classes but the model should not be "confident" 
  on those specific samples)
"""

import argparse
import torch
import numpy as np
import random
import json
import time
import warnings
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import csv

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

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
)
from src.evaluation.mia import shadow_model_mia


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    else:
        raise ValueError(f"Unknown method: {method}")


@torch.no_grad()
def evaluate_sample_level(model, loader, device):
    """
    Evaluate model on a data loader.
    Returns accuracy and average confidence on true class.
    """
    model.eval()
    correct = 0
    total = 0
    total_confidence = 0.0
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)
        
        # Confidence on true class
        probs = torch.softmax(outputs, dim=1)
        batch_indices = torch.arange(outputs.size(0), device=device)
        true_class_conf = probs[batch_indices, targets]
        total_confidence += true_class_conf.sum().item()
        
        # Loss
        total_loss += criterion(outputs, targets).item()
    
    return {
        'accuracy': 100.0 * correct / total,
        'avg_confidence': total_confidence / total,
        'avg_loss': total_loss / total,
    }


def run_sample_level_experiment(
    dataset: str,
    forget_ratio: float,
    seed: int,
    methods: list,
    checkpoint_path: str,
    device: str,
    unlearn_epochs: int,
    unlearn_lr: float,
    verbose: bool = True,
):
    """
    Run a single sample-level unlearning experiment.
    
    Args:
        dataset: Dataset name
        forget_ratio: Fraction of training data to forget (e.g., 0.01 = 1%)
        seed: Random seed
        methods: List of methods to run
        checkpoint_path: Path to base model checkpoint
        device: Device to use
        unlearn_epochs: Number of unlearning epochs
        unlearn_lr: Unlearning learning rate
        verbose: Whether to print progress
    """
    set_seed(seed)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Dataset: {dataset}, Forget Ratio: {forget_ratio*100:.1f}%, Seed: {seed}")
        print(f"{'='*70}")
    
    # Create data config with random sample forgetting
    data_config = DataConfig(
        name=dataset,
        forget_type='random',
        forget_ratio=forget_ratio,
        forget_classes=[],  # Not used for random type
    )
    
    dataset_obj = load_dataset(data_config, seed=seed)
    stats = dataset_obj.get_stats()
    
    if verbose:
        print(f"  Total train: {stats['total_train']}")
        print(f"  Forget set: {stats['forget_size']} samples ({stats['forget_ratio']*100:.1f}%)")
        print(f"  Retain set: {stats['retain_size']} samples")
        print(f"  Test set: {stats['test_size']} samples")
    
    # Determine num_classes
    if dataset == 'cifar10':
        num_classes = 10
    elif dataset == 'cifar100':
        num_classes = 100
    else:
        num_classes = 10
    
    batch_size = 128
    forget_loader = dataset_obj.get_forget_loader(batch_size)
    retain_loader = dataset_obj.get_retain_loader(batch_size)
    test_loader = dataset_obj.get_test_loader(batch_size)
    
    results = {}
    
    for method in methods:
        if verbose:
            print(f"\n  --- {method.upper()} ---")
        
        # Load fresh model
        model = create_model(ModelConfig(architecture='resnet18', num_classes=num_classes))
        model = model.to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        start_time = time.time()
        
        try:
            unlearner = get_unlearner(method, model, device, unlearn_epochs, unlearn_lr)
            if method == 'iweup_v2':
                model = unlearner.unlearn(forget_loader, retain_loader, test_loader=test_loader)
            else:
                model = unlearner.unlearn(forget_loader, retain_loader)
            success = True
        except Exception as e:
            print(f"    Error: {e}")
            import traceback
            traceback.print_exc()
            success = False
        
        elapsed = time.time() - start_time
        
        if success:
            # Evaluate
            forget_eval = evaluate_sample_level(model, forget_loader, device)
            retain_eval = evaluate_sample_level(model, retain_loader, device)
            test_eval = evaluate_sample_level(model, test_loader, device)
            
            # MIA: this is the KEY metric for sample-level unlearning
            # Can an attacker tell which samples were removed?
            try:
                mia_result = shadow_model_mia(
                    model, forget_loader, retain_loader, test_loader, device
                )
                mia_score = mia_result['forget_member_ratio']
            except Exception as e:
                if verbose:
                    print(f"    MIA failed: {e}")
                mia_score = 0.5
            
            results[method] = {
                'forget_acc': forget_eval['accuracy'],
                'forget_conf': forget_eval['avg_confidence'],
                'forget_loss': forget_eval['avg_loss'],
                'retain_acc': retain_eval['accuracy'],
                'test_acc': test_eval['accuracy'],
                'mia': mia_score,
                'time': elapsed,
                'success': True,
            }
            
            if verbose:
                print(f"    Forget Acc: {forget_eval['accuracy']:.2f}%, "
                      f"Conf: {forget_eval['avg_confidence']:.4f}")
                print(f"    Retain Acc: {retain_eval['accuracy']:.2f}%, "
                      f"Test Acc: {test_eval['accuracy']:.2f}%")
                print(f"    MIA: {mia_score:.4f}, Time: {elapsed:.1f}s")
        else:
            results[method] = {
                'forget_acc': np.nan, 'forget_conf': np.nan,
                'forget_loss': np.nan, 'retain_acc': np.nan,
                'test_acc': np.nan, 'mia': np.nan,
                'time': elapsed, 'success': False,
            }
    
    return results


def aggregate_results(all_results):
    """Aggregate results from multiple runs."""
    method_results = defaultdict(lambda: defaultdict(list))
    
    for result in all_results:
        for method, metrics in result.items():
            if metrics['success']:
                for key in ['forget_acc', 'forget_conf', 'forget_loss',
                           'retain_acc', 'test_acc', 'mia', 'time']:
                    method_results[method][key].append(metrics[key])
    
    summary = {}
    for method, metrics in sorted(method_results.items()):
        summary[method] = {}
        for key, values in metrics.items():
            summary[method][f'{key}_mean'] = np.mean(values)
            summary[method][f'{key}_std'] = np.std(values)
            summary[method][f'{key}_count'] = len(values)
    
    return summary


def print_summary_table(summary, forget_ratio):
    """Print formatted results table for sample-level experiments."""
    print(f"\n{'='*130}")
    print(f"SAMPLE-LEVEL UNLEARNING RESULTS (Forget Ratio: {forget_ratio*100:.1f}%)")
    print(f"{'='*130}")
    
    header = (f"{'Method':<18} {'ForgetAcc':>12} {'ForgetConf':>12} "
              f"{'RetainAcc↑':>12} {'TestAcc↑':>12} {'MIA↓':>12} {'Time':>12}")
    print(header)
    print("-" * 130)
    
    for method in sorted(summary.keys()):
        metrics = summary[method]
        row = (f"{method:<18} "
               f"{metrics['forget_acc_mean']:>7.2f}±{metrics['forget_acc_std']:.2f} "
               f"{metrics['forget_conf_mean']:>8.4f}±{metrics['forget_conf_std']:.4f} "
               f"{metrics['retain_acc_mean']:>7.2f}±{metrics['retain_acc_std']:.2f} "
               f"{metrics['test_acc_mean']:>7.2f}±{metrics['test_acc_std']:.2f} "
               f"{metrics['mia_mean']:>7.4f}±{metrics['mia_std']:.4f} "
               f"{metrics['time_mean']:>10.1f}s")
        print(row)
    
    print("=" * 130)
    print("\nInterpretation:")
    print("  ForgetAcc: Accuracy on forgotten samples (should stay high — these are random samples across classes)")
    print("  ForgetConf: Avg confidence on true class for forget samples (lower = more forgetting)")
    print("  MIA↓: Fraction classified as 'member' by attacker (0.0 = best, matches non-member)")
    print("  Key: A good method has LOW MIA and LOW ForgetConf while keeping TestAcc HIGH")


def save_results(summary, all_results, output_dir, forget_ratio):
    """Save results to CSV and JSON."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    csv_path = output_path / f"sample_level_ratio{forget_ratio}_{timestamp}.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['method', 'forget_acc_mean', 'forget_acc_std',
                         'forget_conf_mean', 'forget_conf_std',
                         'retain_acc_mean', 'retain_acc_std',
                         'test_acc_mean', 'test_acc_std',
                         'mia_mean', 'mia_std', 'time_mean'])
        for method, metrics in sorted(summary.items()):
            writer.writerow([
                method,
                metrics.get('forget_acc_mean', ''),
                metrics.get('forget_acc_std', ''),
                metrics.get('forget_conf_mean', ''),
                metrics.get('forget_conf_std', ''),
                metrics.get('retain_acc_mean', ''),
                metrics.get('retain_acc_std', ''),
                metrics.get('test_acc_mean', ''),
                metrics.get('test_acc_std', ''),
                metrics.get('mia_mean', ''),
                metrics.get('mia_std', ''),
                metrics.get('time_mean', ''),
            ])
    
    json_path = output_path / f"sample_level_ratio{forget_ratio}_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump({
            'forget_ratio': forget_ratio,
            'summary': summary,
            'all_results': all_results,
        }, f, indent=2, default=float)
    
    print(f"\nResults saved to:")
    print(f"  CSV: {csv_path}")
    print(f"  JSON: {json_path}")


def main():
    parser = argparse.ArgumentParser(description='Sample-level unlearning experiments')
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['cifar10', 'cifar100'])
    parser.add_argument('--forget_ratios', type=float, nargs='+', default=[0.01, 0.05, 0.10],
                       help='Fraction of training data to forget (e.g., 0.01 0.05 0.10)')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    parser.add_argument('--methods', type=str, nargs='+',
                       default=['finetune', 'gradient_ascent', 'scrub',
                                'bad_teacher', 'amnesiac', 'iweup_v2', 'retrain'])
    parser.add_argument('--checkpoint', type=str,
                       default='checkpoints/cifar10_resnet18_original.pt')
    parser.add_argument('--unlearn_epochs', type=int, default=15)
    parser.add_argument('--unlearn_lr', type=float, default=0.001)
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--quiet', action='store_true')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("SAMPLE-LEVEL UNLEARNING EXPERIMENTS")
    print("=" * 70)
    print(f"  Dataset: {args.dataset}")
    print(f"  Forget ratios: {args.forget_ratios}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Methods: {args.methods}")
    
    for forget_ratio in args.forget_ratios:
        print(f"\n\n{'#'*70}")
        print(f"# FORGET RATIO: {forget_ratio*100:.1f}%")
        print(f"{'#'*70}")
        
        all_results = []
        
        for seed in args.seeds:
            result = run_sample_level_experiment(
                dataset=args.dataset,
                forget_ratio=forget_ratio,
                seed=seed,
                methods=args.methods,
                checkpoint_path=args.checkpoint,
                device=args.device,
                unlearn_epochs=args.unlearn_epochs,
                unlearn_lr=args.unlearn_lr,
                verbose=not args.quiet,
            )
            
            for method in result:
                result[method]['forget_ratio'] = forget_ratio
                result[method]['seed'] = seed
            
            all_results.append(result)
        
        summary = aggregate_results(all_results)
        print_summary_table(summary, forget_ratio)
        save_results(summary, all_results, args.output_dir, forget_ratio)


if __name__ == '__main__':
    main()
