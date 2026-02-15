"""
Main experiment script for class-wise unlearning.

This script runs a complete unlearning experiment:
1. Train a model on full dataset
2. Apply various unlearning methods to forget specific classes
3. Evaluate and compare results
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

# Suppress sklearn warnings about undefined ROC AUC
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
)
from src.evaluation.metrics import compute_unlearning_metrics, UnlearningMetrics
from src.evaluation.mia import simple_mia_evaluation


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def run_experiment(
    config: ExperimentConfig,
    unlearning_methods: list = None,
):
    """
    Run a complete class unlearning experiment.
    
    Args:
        config: Experiment configuration
        unlearning_methods: List of methods to evaluate
    """
    if unlearning_methods is None:
        unlearning_methods = ['finetune', 'gradient_ascent', 'ssd', 'pgu', 'igtu', 'iweup']
    
    print("=" * 60)
    print(f"Running experiment: {config.name}")
    print(f"Dataset: {config.data.name}")
    print(f"Forget classes: {config.data.forget_classes}")
    print(f"Device: {config.device}")
    print("=" * 60)
    
    set_seed(config.seed)
    device = config.device
    
    # Create output directory
    output_dir = Path(config.output_dir) / config.name / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump({
            'name': config.name,
            'seed': config.seed,
            'dataset': config.data.name,
            'forget_classes': config.data.forget_classes,
            'model': config.model.architecture,
        }, f, indent=2)
    
    # ==================== Data Loading ====================
    print("\n[1/5] Loading dataset...")
    dataset = load_dataset(config.data, seed=config.seed)
    stats = dataset.get_stats()
    print(f"  Total train: {stats['total_train']}")
    print(f"  Forget size: {stats['forget_size']} ({stats['forget_ratio']*100:.1f}%)")
    print(f"  Retain size: {stats['retain_size']}")
    print(f"  Test size: {stats['test_size']}")
    
    # Create data loaders
    forget_loader = dataset.get_forget_loader(config.data.batch_size, num_workers=config.data.num_workers)
    retain_loader = dataset.get_retain_loader(config.data.batch_size, num_workers=config.data.num_workers)
    test_loader = dataset.get_test_loader(config.data.batch_size, num_workers=config.data.num_workers)
    full_train_loader = dataset.get_full_train_loader(config.data.batch_size, num_workers=config.data.num_workers)
    
    # ==================== Model Training ====================
    print("\n[2/5] Training original model...")
    
    checkpoint_path = Path(config.model.checkpoint_dir) / f"{config.data.name}_{config.model.architecture}_original.pt"
    
    model = create_model(config.model, device)
    
    if checkpoint_path.exists():
        print(f"  Loading existing checkpoint from {checkpoint_path}")
        load_checkpoint(model, str(checkpoint_path), device=device)
    else:
        trainer = Trainer(model, device=device)
        trainer.train(
            full_train_loader,
            val_loader=test_loader,
            epochs=config.training.epochs,
            lr=config.training.learning_rate,
            checkpoint_dir=config.model.checkpoint_dir,
        )
        save_checkpoint(model, None, config.training.epochs, 0.0, str(checkpoint_path))
    
    # Evaluate original model
    original_model = create_model(config.model, device)
    original_model.load_state_dict(model.state_dict())
    original_model.eval()
    
    print("\n  Original model performance:")
    from src.evaluation.metrics import compute_accuracy
    forget_acc, _ = compute_accuracy(original_model, forget_loader, device)
    retain_acc, _ = compute_accuracy(original_model, retain_loader, device)
    test_acc, _ = compute_accuracy(original_model, test_loader, device)
    print(f"    Forget accuracy: {forget_acc:.2f}%")
    print(f"    Retain accuracy: {retain_acc:.2f}%")
    print(f"    Test accuracy: {test_acc:.2f}%")
    
    # ==================== Unlearning ====================
    print("\n[3/5] Running unlearning methods...")
    
    results = {}
    
    for method_name in unlearning_methods:
        print(f"\n  --- {method_name.upper()} ---")
        
        # Create fresh copy of model
        model_copy = create_model(config.model, device)
        model_copy.load_state_dict(original_model.state_dict())
        
        # Create unlearner
        start_time = time.time()
        
        if method_name == 'finetune':
            unlearner = FineTuneUnlearner(
                model_copy, device=device,
                epochs=config.unlearning.unlearn_epochs,
                lr=config.unlearning.unlearn_lr,
            )
        elif method_name == 'gradient_ascent':
            unlearner = GradientAscentUnlearner(
                model_copy, device=device,
                epochs=config.unlearning.unlearn_epochs,
                lr=config.unlearning.unlearn_lr,
            )
        elif method_name == 'ssd':
            unlearner = SSDUnlearner(
                model_copy, device=device,
                dampening_lambda=config.unlearning.ssd_lambda,
                selection_alpha=config.unlearning.ssd_alpha,
                epochs=config.unlearning.unlearn_epochs,
            )
        elif method_name == 'pgu':
            unlearner = PGUUnlearner(
                model_copy, device=device,
                epochs=config.unlearning.unlearn_epochs,
                lr=config.unlearning.unlearn_lr,
                project_dim=config.unlearning.pgu_project_dim,
            )
        elif method_name == 'igtu':
            unlearner = IGTUUnlearner(
                model_copy, device=device,
                epochs=config.unlearning.unlearn_epochs,
                lr=config.unlearning.unlearn_lr,
                influence_samples=config.unlearning.igtu_influence_samples,
                lissa_depth=config.unlearning.igtu_lissa_depth,
                lissa_scale=config.unlearning.igtu_lissa_scale,
                retention_weight=config.unlearning.igtu_retention_weight,
            )
        elif method_name == 'iweup':
            unlearner = IWEUPUnlearner(
                model_copy, device=device,
                epochs=config.unlearning.unlearn_epochs,
                lr=config.unlearning.unlearn_lr,
            )
        else:
            print(f"  Unknown method: {method_name}")
            continue
        
        # Run unlearning
        unlearned_model = unlearner.unlearn(forget_loader, retain_loader)
        unlearn_time = time.time() - start_time
        
        # ==================== Evaluation ====================
        print(f"\n  Evaluating {method_name}...")
        
        # Get certificate if IGTU
        certificate = None
        if method_name == 'igtu' and hasattr(unlearner, 'get_certificate'):
            certificate = unlearner.get_certificate()
        
        # Compute metrics
        metrics = compute_unlearning_metrics(
            unlearned_model,
            original_model,
            forget_loader,
            retain_loader,
            test_loader,
            device=device,
            certificate=certificate,
        )
        metrics.unlearning_time = unlearn_time
        
        # MIA evaluation
        mia_results = simple_mia_evaluation(
            unlearned_model, forget_loader, retain_loader, device
        )
        metrics.mia_auc = (mia_results['forget_mia_auc'] + mia_results['retain_mia_auc']) / 2
        
        results[method_name] = metrics.to_dict()
        results[method_name]['mia_details'] = mia_results
        
        print(f"    Forget accuracy: {metrics.forget_accuracy:.2f}%")
        print(f"    Retain accuracy: {metrics.retain_accuracy:.2f}%")
        print(f"    Test accuracy: {metrics.test_accuracy:.2f}%")
        print(f"    Unlearning accuracy (drop): {metrics.unlearning_accuracy:.2f}%")
        print(f"    Remaining accuracy: {metrics.remaining_accuracy:.2f}%")
        print(f"    MIA forget member ratio: {mia_results['forget_member_ratio']:.4f}")
        print(f"    Time: {unlearn_time:.2f}s")
        
        # Save unlearned model
        save_checkpoint(
            unlearned_model, None, 0, metrics.test_accuracy,
            str(output_dir / f'{method_name}_model.pt')
        )
    
    # ==================== Results Summary ====================
    print("\n[4/5] Results Summary")
    print("=" * 80)
    print(f"{'Method':<15} {'Forget↓':>10} {'Retain↑':>10} {'Test↑':>10} {'UA↑':>10} {'MIA↓':>10} {'Time':>10}")
    print("-" * 80)
    
    for method_name, result in results.items():
        print(f"{method_name:<15} "
              f"{result['forget_accuracy']:>9.2f}% "
              f"{result['retain_accuracy']:>9.2f}% "
              f"{result['test_accuracy']:>9.2f}% "
              f"{result['unlearning_accuracy']:>9.2f}% "
              f"{result['mia_details']['forget_member_ratio']:>9.4f} "
              f"{result['unlearning_time']:>9.2f}s")
    print("=" * 80)
    
    # ==================== Save Results ====================
    print("\n[5/5] Saving results...")
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_dir}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Class Unlearning Experiment')
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['cifar10', 'cifar100', 'svhn'])
    parser.add_argument('--forget_class', type=int, default=0,
                       help='Class to forget (for single class unlearning)')
    parser.add_argument('--model', type=str, default='resnet18',
                       choices=['resnet18', 'resnet34', 'vgg16'])
    parser.add_argument('--epochs', type=int, default=100,
                       help='Training epochs for original model')
    parser.add_argument('--unlearn_epochs', type=int, default=15,
                       help='Unlearning epochs')
    parser.add_argument('--unlearn_lr', type=float, default=0.001,
                       help='Learning rate for unlearning (lower is more stable)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--methods', type=str, nargs='+',
                       default=['finetune', 'gradient_ascent', 'ssd', 'pgu', 'igtu'],
                       help='Unlearning methods to evaluate')
    
    args = parser.parse_args()
    
    # Create config
    num_classes = {'cifar10': 10, 'cifar100': 100, 'svhn': 10}[args.dataset]
    
    config = ExperimentConfig(
        name=f"{args.dataset}_forget_class_{args.forget_class}",
        seed=args.seed,
        device=args.device,
        data=DataConfig(
            name=args.dataset,
            forget_type='class',
            forget_classes=[args.forget_class],
        ),
        model=ModelConfig(
            architecture=args.model,
            num_classes=num_classes,
        ),
        output_dir=args.output_dir,
    )
    config.training.epochs = args.epochs
    config.unlearning.unlearn_epochs = args.unlearn_epochs
    config.unlearning.unlearn_lr = args.unlearn_lr
    
    # Run experiment
    run_experiment(config, args.methods)


if __name__ == '__main__':
    main()
