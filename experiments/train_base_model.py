"""
Train a ResNet-18 model on a specified dataset and save the checkpoint.

Usage:
    # CIFAR-10 (default)
    python experiments/train_base_model.py --dataset cifar10
    
    # CIFAR-100
    python experiments/train_base_model.py --dataset cifar100
    
    # Custom settings
    python experiments/train_base_model.py --dataset cifar100 --epochs 200 --lr 0.1
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import torch.nn as nn
from pathlib import Path

from src.models.classifier import ResNet18CIFAR
from src.data.dataset import load_dataset
from src.config import DataConfig
from src.training import Trainer


def main():
    parser = argparse.ArgumentParser(description='Train base model for unlearning experiments')
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['cifar10', 'cifar100', 'svhn'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default='checkpoints')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    # Determine num classes
    num_classes = {'cifar10': 10, 'cifar100': 100, 'svhn': 10}[args.dataset]
    
    print(f"Training ResNet-18 on {args.dataset.upper()}")
    print(f"  Classes: {num_classes}")
    print(f"  Epochs: {args.epochs}")
    print(f"  LR: {args.lr}")
    print(f"  Device: {args.device}")
    
    # Create model
    model = ResNet18CIFAR(num_classes=num_classes).to(args.device)
    
    # Load FULL dataset (no forget class — this is the original model)
    data_config = DataConfig(
        name=args.dataset,
        forget_classes=[],  # No forget class for base training
        forget_type='class'
    )
    dataset_obj = load_dataset(data_config, seed=args.seed)
    
    # Use full train set and test set
    train_loader = dataset_obj.get_full_train_loader(args.batch_size)
    test_loader = dataset_obj.get_test_loader(args.batch_size)
    
    # Train
    trainer = Trainer(model, device=args.device, verbose=True)
    model = trainer.train(
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=args.epochs,
        lr=args.lr,
        momentum=0.9,
        weight_decay=5e-4,
        lr_scheduler='step',
        lr_decay_epochs=(50, 75) if args.epochs <= 100 else (100, 150),
        lr_decay_factor=0.1,
        checkpoint_dir=args.output_dir,
        checkpoint_frequency=25,
    )
    
    # Save final model
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    final_path = output_path / f"{args.dataset}_resnet18_original.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'dataset': args.dataset,
        'num_classes': num_classes,
        'epochs': args.epochs,
        'seed': args.seed,
    }, final_path)
    
    # Final evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    test_acc = 100.0 * correct / total
    print(f"\nFinal test accuracy: {test_acc:.2f}%")
    print(f"Checkpoint saved to: {final_path}")


if __name__ == '__main__':
    main()
