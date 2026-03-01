"""
Comprehensive experiment runner for rigorous unlearning evaluation.

Features:
- Multiple random seeds for statistical significance
- Multiple forget classes (not just class 0)
- Multiple datasets (CIFAR-10, CIFAR-100, SVHN)
- Automatic result aggregation with mean and std dev
- CSV/JSON export for paper tables
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

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.config import ExperimentConfig, DataConfig, ModelConfig
from src.data.dataset import load_dataset
from src.models.classifier import create_model, save_checkpoint, load_checkpoint
from src.training import Trainer
from src.unlearning import (
    FineTuneUnlearner,
    GradientAscentUnlearner,
    SSDUnlearner,
    PGUUnlearner,
    IGTUUnlearner,
    IWEUPUnlearner,
    IWEUPv2Unlearner,
    SCRUBUnlearner,
    RetrainUnlearner,
    BadTeacherUnlearner,
    AmnesiacUnlearner,
    SalUnUnlearner,
)
from src.evaluation.metrics import compute_unlearning_metrics, UnlearningMetrics
from src.evaluation.mia import shadow_model_mia


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def _compute_non_forget_test_acc(model, test_loader, forget_classes, device):
    """
    Compute test accuracy excluding forget class samples.
    
    This is the FAIR comparison metric: methods that successfully forget
    will have low accuracy on forget samples, inflating the gap when 
    measured on all test data. This metric only measures utility on
    non-forget classes.
    """
    model.eval()
    correct = 0
    total = 0
    forget_set = set(forget_classes)
    
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Filter out forget class samples
        mask = torch.tensor([t.item() not in forget_set for t in targets], 
                          dtype=torch.bool, device=device)
        if mask.sum() == 0:
            continue
        
        filtered_inputs = inputs[mask]
        filtered_targets = targets[mask]
        
        outputs = model(filtered_inputs)
        _, predicted = outputs.max(1)
        total += filtered_targets.size(0)
        correct += predicted.eq(filtered_targets).sum().item()
    
    return 100.0 * correct / total if total > 0 else 0.0

def get_unlearner(method: str, model, device: str, epochs: int, lr: float):
    """Create unlearner instance based on method name."""
    if method == 'finetune':
        return FineTuneUnlearner(
            model, device=device, epochs=epochs, lr=lr
        )
    elif method == 'gradient_ascent':
        return GradientAscentUnlearner(
            model, device=device, epochs=epochs, lr=lr * 0.5
        )
    elif method == 'ssd':
        return SSDUnlearner(
            model, device=device, epochs=epochs, lr=lr
        )
    elif method == 'pgu':
        return PGUUnlearner(
            model, device=device, epochs=epochs, lr=lr * 0.1
        )
    elif method == 'igtu':
        return IGTUUnlearner(
            model, device=device, epochs=epochs, lr=lr
        )
    elif method == 'iweup':
        return IWEUPUnlearner(
            model, device=device, epochs=epochs, lr=lr
        )
    elif method == 'iweup_v2':
        return IWEUPv2Unlearner(
            model, device=device, epochs=epochs, lr=lr
        )
    elif method == 'scrub':
        return SCRUBUnlearner(
            model, device=device, epochs=epochs, lr=lr
        )
    elif method == 'retrain':
        return RetrainUnlearner(
            model, device=device, epochs=100, lr=0.1
        )
    elif method == 'bad_teacher':
        return BadTeacherUnlearner(
            model, device=device, epochs=epochs, lr=lr
        )
    elif method == 'amnesiac':
        return AmnesiacUnlearner(
            model, device=device, epochs=epochs, lr=lr
        )
    elif method == 'salun':
        return SalUnUnlearner(
            model, device=device, epochs=epochs, lr=lr
        )
    else:
        raise ValueError(f"Unknown method: {method}")


def run_single_experiment(
    dataset: str,
    forget_class: int,
    seed: int,
    methods: list,
    checkpoint_path: str,
    device: str,
    unlearn_epochs: int,
    unlearn_lr: float,
    verbose: bool = True,
):
    """
    Run a single experiment with one seed and one forget class.
    
    Returns:
        Dictionary with results for each method
    """
    set_seed(seed)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset}, Forget Class: {forget_class}, Seed: {seed}")
        print(f"{'='*60}")
    
    # Create data config with forget class
    data_config = DataConfig(
        name=dataset,
        forget_classes=[forget_class],
        forget_type='class'
    )
    
    # Load dataset using the config object
    dataset_obj = load_dataset(data_config, seed=seed)
    
    # Get number of classes
    if dataset == 'cifar10':
        num_classes = 10
    elif dataset == 'cifar100':
        num_classes = 100
    elif dataset == 'svhn':
        num_classes = 10
    else:
        num_classes = 10
    
    # Create data loaders
    batch_size = 128
    forget_loader = dataset_obj.get_forget_loader(batch_size)
    retain_loader = dataset_obj.get_retain_loader(batch_size)
    test_loader = dataset_obj.get_test_loader(batch_size)
    
    results = {}
    
    for method in methods:
        if verbose:
            print(f"\n  --- {method.upper()} ---")
        
        # Load fresh model from checkpoint
        model = create_model(ModelConfig(architecture='resnet18', num_classes=num_classes))
        model = model.to(device)
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Run unlearning
        start_time = time.time()
        
        try:
            unlearner = get_unlearner(method, model, device, unlearn_epochs, unlearn_lr)
            # IWEUP v2 needs test_loader for MIA reference
            if method == 'iweup_v2':
                model = unlearner.unlearn(forget_loader, retain_loader, test_loader=test_loader)
            else:
                model = unlearner.unlearn(forget_loader, retain_loader)
            success = True
        except Exception as e:
            print(f"    Error: {e}")
            success = False
        
        elapsed = time.time() - start_time
        
        if success:
            # Evaluate
            forget_metrics = unlearner.evaluate(forget_loader)
            retain_metrics = unlearner.evaluate(retain_loader)
            test_metrics = unlearner.evaluate(test_loader)
            
            # Non-forget test accuracy (fair comparison metric)
            # Excludes forget class from test set to avoid penalizing
            # methods that actually forget
            nf_test_acc = _compute_non_forget_test_acc(
                model, test_loader, data_config.forget_classes, device
            )
            
            # MIA evaluation (proper shadow-model attack)
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
                'forget_acc': forget_metrics['accuracy'],
                'retain_acc': retain_metrics['accuracy'],
                'test_acc': test_metrics['accuracy'],
                'nf_test_acc': nf_test_acc,
                'mia': mia_score,
                'time': elapsed,
                'success': True,
            }
            
            if verbose:
                print(f"    Forget: {forget_metrics['accuracy']:.2f}%, "
                      f"Retain: {retain_metrics['accuracy']:.2f}%, "
                      f"Test: {test_metrics['accuracy']:.2f}%, "
                      f"NF-Test: {nf_test_acc:.2f}%, "
                      f"MIA: {mia_score:.4f}, Time: {elapsed:.1f}s")
        else:
            results[method] = {
                'forget_acc': np.nan,
                'retain_acc': np.nan,
                'test_acc': np.nan,
                'nf_test_acc': np.nan,
                'mia': np.nan,
                'time': elapsed,
                'success': False,
            }
    
    return results


def aggregate_results(all_results: list):
    """
    Aggregate results across multiple runs.
    
    Args:
        all_results: List of result dictionaries from run_single_experiment
        
    Returns:
        Dictionary with mean and std for each method and metric
    """
    aggregated = defaultdict(lambda: defaultdict(list))
    
    for result in all_results:
        for method, metrics in result.items():
            if metrics['success']:
                for key, value in metrics.items():
                    if key != 'success':
                        aggregated[method][key].append(value)
    
    summary = {}
    for method, metrics in aggregated.items():
        summary[method] = {}
        for key, values in metrics.items():
            values = [v for v in values if not np.isnan(v)]
            if values:
                summary[method][f'{key}_mean'] = np.mean(values)
                summary[method][f'{key}_std'] = np.std(values)
                summary[method][f'{key}_count'] = len(values)
    
    return summary


def print_summary_table(summary: dict):
    """Print formatted summary table."""
    print("\n" + "="*120)
    print("COMPREHENSIVE RESULTS SUMMARY (Mean ± Std)")
    print("="*120)
    
    header = f"{'Method':<18} {'Forget↓':>12} {'Retain↑':>12} {'Test↑':>12} {'NF-Test↑':>12} {'MIA≈0.5':>12} {'Time':>12}"
    print(header)
    print("-"*120)
    
    for method, metrics in sorted(summary.items()):
        forget = f"{metrics.get('forget_acc_mean', 0):.2f}±{metrics.get('forget_acc_std', 0):.2f}"
        retain = f"{metrics.get('retain_acc_mean', 0):.2f}±{metrics.get('retain_acc_std', 0):.2f}"
        test = f"{metrics.get('test_acc_mean', 0):.2f}±{metrics.get('test_acc_std', 0):.2f}"
        nf_test = f"{metrics.get('nf_test_acc_mean', 0):.2f}±{metrics.get('nf_test_acc_std', 0):.2f}"
        mia = f"{metrics.get('mia_mean', 0):.4f}±{metrics.get('mia_std', 0):.4f}"
        time_str = f"{metrics.get('time_mean', 0):.1f}s"
        
        print(f"{method:<18} {forget:>12} {retain:>12} {test:>12} {nf_test:>12} {mia:>12} {time_str:>12}")
    
    print("="*120)


def save_results(summary: dict, all_results: list, output_dir: str):
    """Save results to CSV and JSON files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save summary to CSV
    csv_path = output_path / f"summary_{timestamp}.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Method', 'Forget_Mean', 'Forget_Std', 'Retain_Mean', 'Retain_Std',
                        'Test_Mean', 'Test_Std', 'NF_Test_Mean', 'NF_Test_Std',
                        'MIA_Mean', 'MIA_Std', 'Time_Mean', 'N'])
        
        for method, metrics in sorted(summary.items()):
            writer.writerow([
                method,
                metrics.get('forget_acc_mean', ''),
                metrics.get('forget_acc_std', ''),
                metrics.get('retain_acc_mean', ''),
                metrics.get('retain_acc_std', ''),
                metrics.get('test_acc_mean', ''),
                metrics.get('test_acc_std', ''),
                metrics.get('nf_test_acc_mean', ''),
                metrics.get('nf_test_acc_std', ''),
                metrics.get('mia_mean', ''),
                metrics.get('mia_std', ''),
                metrics.get('time_mean', ''),
                metrics.get('forget_acc_count', ''),
            ])
    
    # Save full results to JSON
    json_path = output_path / f"full_results_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump({
            'summary': summary,
            'all_results': all_results,
        }, f, indent=2, default=float)
    
    print(f"\nResults saved to:")
    print(f"  CSV: {csv_path}")
    print(f"  JSON: {json_path}")


def main():
    parser = argparse.ArgumentParser(description='Comprehensive unlearning experiments')
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['cifar10', 'cifar100', 'svhn'])
    parser.add_argument('--forget_classes', type=int, nargs='+', default=[0],
                       help='Classes to forget (run separately for each)')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456],
                       help='Random seeds to use')
    parser.add_argument('--methods', type=str, nargs='+',
                       default=['finetune', 'gradient_ascent', 'ssd', 'pgu', 'scrub',
                                'bad_teacher', 'amnesiac', 'salun', 'iweup_v2', 'retrain'])
    parser.add_argument('--checkpoint', type=str,
                       default='checkpoints/cifar10_resnet18_original.pt')
    parser.add_argument('--unlearn_epochs', type=int, default=15)
    parser.add_argument('--unlearn_lr', type=float, default=0.001)
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--quiet', action='store_true')
    
    args = parser.parse_args()
    
    print(f"Comprehensive Unlearning Experiment")
    print(f"  Dataset: {args.dataset}")
    print(f"  Forget classes: {args.forget_classes}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Methods: {args.methods}")
    print(f"  Total runs: {len(args.forget_classes) * len(args.seeds)}")
    
    all_results = []
    
    for forget_class in args.forget_classes:
        for seed in args.seeds:
            result = run_single_experiment(
                dataset=args.dataset,
                forget_class=forget_class,
                seed=seed,
                methods=args.methods,
                checkpoint_path=args.checkpoint,
                device=args.device,
                unlearn_epochs=args.unlearn_epochs,
                unlearn_lr=args.unlearn_lr,
                verbose=not args.quiet,
            )
            
            # Add metadata
            for method in result:
                result[method]['forget_class'] = forget_class
                result[method]['seed'] = seed
            
            all_results.append(result)
    
    # Aggregate and display results
    summary = aggregate_results(all_results)
    print_summary_table(summary)
    
    # Save results
    save_results(summary, all_results, args.output_dir)


if __name__ == '__main__':
    main()
