"""
Activation Extractor for ResNet models.

Extracts intermediate representations from any layer of a ResNet model
using PyTorch hooks. These representations are used by RINE for
information decomposition auditing.

Usage:
    extractor = ResNetActivationExtractor(model, device='cuda')
    Z = extractor.extract(dataloader, layer='avgpool')
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Optional, List, Dict
from pathlib import Path


class ResNetActivationExtractor:
    """
    Hook-based activation extractor for ResNet models.
    
    Extracts representations from specified layers and returns
    them as numpy arrays suitable for RINE auditing.
    """
    
    # Map of friendly names to ResNet-18 layer attributes
    LAYER_MAP = {
        'layer1': 'layer1',
        'layer2': 'layer2',
        'layer3': 'layer3',
        'layer4': 'layer4',
        'avgpool': 'avgpool',
    }
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
    ):
        self.model = model
        self.device = device
        self._activations = {}
        self._hooks = []
    
    def _register_hook(self, layer_name: str):
        """Register a forward hook on the specified layer."""
        # Navigate to the target layer
        target = self.model
        for attr in self.LAYER_MAP[layer_name].split('.'):
            target = getattr(target, attr)
        
        def hook_fn(module, input, output):
            self._activations[layer_name] = output.detach()
        
        handle = target.register_forward_hook(hook_fn)
        self._hooks.append(handle)
    
    def _remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
    
    @torch.no_grad()
    def extract(
        self,
        dataloader: DataLoader,
        layer: str = 'avgpool',
        cache_path: Optional[str] = None,
    ) -> tuple:
        """
        Extract representations from the specified layer.
        
        Args:
            dataloader: DataLoader providing (inputs, targets)
            layer: Layer name to extract from (see LAYER_MAP)
            cache_path: Optional path to cache/load results (.npz)
            
        Returns:
            Tuple of (activations: np.ndarray, labels: np.ndarray)
            activations shape: (N, D) where D is the flattened 
            representation dimension
        """
        # Check cache
        if cache_path and Path(cache_path).exists():
            data = np.load(cache_path)
            return data['activations'], data['labels']
        
        if layer not in self.LAYER_MAP:
            raise ValueError(
                f"Unknown layer '{layer}'. "
                f"Available: {list(self.LAYER_MAP.keys())}"
            )
        
        self.model.eval()
        self.model.to(self.device)
        self._register_hook(layer)
        
        all_activations = []
        all_labels = []
        
        for inputs, targets in dataloader:
            inputs = inputs.to(self.device)
            
            # Forward pass triggers the hook
            _ = self.model(inputs)
            
            # Get the captured activation
            act = self._activations[layer]
            
            # Flatten: (batch, C, H, W) → (batch, C*H*W)
            # or (batch, C, 1, 1) → (batch, C) for avgpool
            act_flat = act.view(act.size(0), -1).cpu().numpy()
            
            all_activations.append(act_flat)
            all_labels.append(targets.numpy())
        
        self._remove_hooks()
        
        activations = np.concatenate(all_activations, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        
        # Cache to disk
        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache_path, activations=activations, labels=labels)
        
        return activations, labels
    
    @torch.no_grad()
    def extract_all_layers(
        self,
        dataloader: DataLoader,
        layers: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Extract representations from multiple layers simultaneously.
        
        Args:
            dataloader: DataLoader providing (inputs, targets)
            layers: List of layer names. Defaults to all available.
            
        Returns:
            Dictionary mapping layer_name → (activations, labels)
        """
        if layers is None:
            layers = list(self.LAYER_MAP.keys())
        
        self.model.eval()
        self.model.to(self.device)
        
        # Register hooks for all requested layers
        for layer_name in layers:
            self._register_hook(layer_name)
        
        all_activations = {name: [] for name in layers}
        all_labels = []
        
        for inputs, targets in dataloader:
            inputs = inputs.to(self.device)
            _ = self.model(inputs)
            
            for name in layers:
                act = self._activations[name]
                act_flat = act.view(act.size(0), -1).cpu().numpy()
                all_activations[name].append(act_flat)
            
            all_labels.append(targets.numpy())
        
        self._remove_hooks()
        
        labels = np.concatenate(all_labels, axis=0)
        results = {}
        for name in layers:
            results[name] = (
                np.concatenate(all_activations[name], axis=0),
                labels,
            )
        
        return results


def extract_for_audit(
    base_model: nn.Module,
    unlearned_model: nn.Module,
    forget_loader: DataLoader,
    retain_loader: DataLoader,
    layer: str = 'avgpool',
    device: str = 'cuda',
    cache_dir: Optional[str] = None,
    experiment_name: str = 'default',
) -> tuple:
    """
    Convenience function to extract representations for RINE auditing.
    
    Creates binary labels (forget=1, retain=0), extracts activations
    from both models, and returns everything RINE needs.
    
    Args:
        base_model: Original trained model
        unlearned_model: Model after unlearning
        forget_loader: DataLoader for forget set
        retain_loader: DataLoader for retain set
        layer: ResNet layer to extract from
        device: Computation device
        cache_dir: Optional directory for caching activations
        experiment_name: Name for cache file naming
        
    Returns:
        Tuple of (Z_base, Z_unlearned, Y) as numpy arrays
        Y is binary: 1=forget, 0=retain
    """
    # Extract base model activations
    base_extractor = ResNetActivationExtractor(base_model, device)
    
    base_cache_f = f"{cache_dir}/base_{experiment_name}_forget.npz" if cache_dir else None
    base_cache_r = f"{cache_dir}/base_{experiment_name}_retain.npz" if cache_dir else None
    
    Z_base_forget, _ = base_extractor.extract(forget_loader, layer, base_cache_f)
    Z_base_retain, _ = base_extractor.extract(retain_loader, layer, base_cache_r)
    
    # Extract unlearned model activations
    unl_extractor = ResNetActivationExtractor(unlearned_model, device)
    
    unl_cache_f = f"{cache_dir}/unl_{experiment_name}_forget.npz" if cache_dir else None
    unl_cache_r = f"{cache_dir}/unl_{experiment_name}_retain.npz" if cache_dir else None
    
    Z_unl_forget, _ = unl_extractor.extract(forget_loader, layer, unl_cache_f)
    Z_unl_retain, _ = unl_extractor.extract(retain_loader, layer, unl_cache_r)
    
    # Combine and create binary labels
    Z_base = np.concatenate([Z_base_forget, Z_base_retain], axis=0)
    Z_unlearned = np.concatenate([Z_unl_forget, Z_unl_retain], axis=0)
    Y = np.concatenate([
        np.ones(len(Z_base_forget)),    # forget = 1
        np.zeros(len(Z_base_retain)),   # retain = 0
    ])
    
    # Shuffle together to avoid ordering bias
    perm = np.random.permutation(len(Y))
    Z_base = Z_base[perm]
    Z_unlearned = Z_unlearned[perm]
    Y = Y[perm]
    
    return Z_base, Z_unlearned, Y
