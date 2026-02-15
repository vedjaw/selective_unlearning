"""
Influence Functions for Neural Networks.

Implements efficient influence function computation using:
- LiSSA (Linear time Stochastic Second-order Algorithm)
- Arnoldi iteration for Hessian approximation

References:
- Koh & Liang (2017): "Understanding Black-box Predictions via Influence Functions"
- Agarwal et al. (2017): "Second-order Stochastic Optimization for Machine Learning"
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Tuple, Optional, Dict, Callable
import numpy as np
from tqdm import tqdm


def get_gradient_vector(
    model: nn.Module,
    loss: torch.Tensor,
    flatten: bool = True
) -> torch.Tensor:
    """
    Compute gradient of loss with respect to model parameters.
    
    Args:
        model: Neural network model
        loss: Scalar loss value
        flatten: Whether to flatten gradients into a single vector
        
    Returns:
        Gradient vector or list of gradient tensors
    """
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    
    if flatten:
        return torch.cat([g.flatten() for g in grads])
    return grads


def hvp(
    model: nn.Module,
    loss: torch.Tensor,
    vector: torch.Tensor,
) -> torch.Tensor:
    """
    Compute Hessian-Vector Product: H @ v
    
    Uses the trick: H @ v = d/dt [grad(loss) @ v] evaluated at t=0
    
    Args:
        model: Neural network model
        loss: Scalar loss value (with create_graph=True on gradient)
        vector: Vector to multiply with Hessian
        
    Returns:
        Hessian-vector product
    """
    # First compute gradient
    grads = get_gradient_vector(model, loss, flatten=True)
    
    # Compute product with vector
    grad_vector_product = (grads * vector).sum()
    
    # Second derivative gives Hessian-vector product
    hvp_result = torch.autograd.grad(
        grad_vector_product,
        model.parameters(),
        retain_graph=True
    )
    
    return torch.cat([h.flatten() for h in hvp_result])


def compute_inverse_hvp_lissa(
    model: nn.Module,
    train_loader: DataLoader,
    vector: torch.Tensor,
    device: str = 'cuda',
    depth: int = 5000,
    scale: float = 25.0,
    damping: float = 0.01,
    num_samples: int = 1,
) -> torch.Tensor:
    """
    Compute inverse Hessian-vector product using LiSSA.
    
    LiSSA approximates H^{-1} @ v using the Neumann series:
    H^{-1} = sum_{i=0}^{inf} (I - H/scale)^i / scale
    
    Args:
        model: Neural network model
        train_loader: DataLoader for computing Hessian estimates
        vector: Vector v for which to compute H^{-1} @ v
        device: Computation device
        depth: Number of recursion steps
        scale: Scaling factor for Hessian
        damping: Damping factor for numerical stability
        num_samples: Number of LiSSA estimates to average
        
    Returns:
        Approximate inverse Hessian-vector product
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    ihvp_estimates = []
    
    for _ in range(num_samples):
        current_estimate = vector.clone()
        
        # Stochastic recursion
        data_iter = iter(train_loader)
        
        for i in range(depth):
            # Get a random batch
            try:
                inputs, targets = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                inputs, targets = next(data_iter)
            
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Compute loss and HVP
            model.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Compute Hessian-vector product
            hvp_result = hvp(model, loss, current_estimate)
            
            # Update estimate: v_{i+1} = v + (I - H/scale) @ v_i
            current_estimate = vector + (1 - damping) * current_estimate - hvp_result / scale
        
        ihvp_estimates.append(current_estimate / scale)
    
    # Average estimates
    return torch.stack(ihvp_estimates).mean(dim=0)


def compute_inverse_hvp_arnoldi(
    model: nn.Module,
    train_loader: DataLoader,
    vector: torch.Tensor,
    device: str = 'cuda',
    arnoldi_dim: int = 200,
    damping: float = 0.01,
) -> torch.Tensor:
    """
    Compute inverse HVP using Arnoldi iteration.
    
    Builds a low-rank approximation of the Hessian using Arnoldi iteration,
    then computes the inverse-HVP in the low-dimensional subspace.
    
    Args:
        model: Neural network model
        train_loader: DataLoader for Hessian estimates
        vector: Vector for inverse HVP
        device: Computation device
        arnoldi_dim: Dimension of Krylov subspace
        damping: Damping for numerical stability
        
    Returns:
        Approximate inverse HVP
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    # Number of parameters
    n_params = sum(p.numel() for p in model.parameters())
    
    # Initialize Arnoldi basis
    Q = torch.zeros(n_params, arnoldi_dim + 1, device=device)
    H = torch.zeros(arnoldi_dim + 1, arnoldi_dim, device=device)
    
    # Normalize initial vector
    q0 = vector / (vector.norm() + 1e-10)
    Q[:, 0] = q0
    
    data_iter = iter(train_loader)
    
    for j in range(arnoldi_dim):
        # Get batch for HVP
        try:
            inputs, targets = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            inputs, targets = next(data_iter)
        
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Compute HVP
        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        q_j = Q[:, j]
        v = hvp(model, loss, q_j)
        
        # Arnoldi orthogonalization
        for i in range(j + 1):
            H[i, j] = torch.dot(Q[:, i], v)
            v = v - H[i, j] * Q[:, i]
        
        H[j + 1, j] = v.norm()
        if H[j + 1, j] > 1e-10:
            Q[:, j + 1] = v / H[j + 1, j]
    
    # Add damping to H
    H_square = H[:arnoldi_dim, :arnoldi_dim]
    H_damped = H_square + damping * torch.eye(arnoldi_dim, device=device)
    
    # Project vector onto Krylov subspace and solve
    beta = vector.norm()
    e1 = torch.zeros(arnoldi_dim, device=device)
    e1[0] = beta
    
    # Solve H^{-1} @ e1 in low-dimensional space
    try:
        y = torch.linalg.solve(H_damped, e1)
    except:
        # Fallback to pseudo-inverse
        y = torch.linalg.lstsq(H_damped, e1).solution
    
    # Project back to full space
    return Q[:, :arnoldi_dim] @ y


def compute_sample_influence(
    model: nn.Module,
    train_loader: DataLoader,
    test_sample: Tuple[torch.Tensor, torch.Tensor],
    train_sample: Tuple[torch.Tensor, torch.Tensor],
    device: str = 'cuda',
    method: str = 'lissa',
    **kwargs
) -> float:
    """
    Compute the influence of a training sample on a test prediction.
    
    Influence = -grad_test^T @ H^{-1} @ grad_train
    
    A positive influence means removing the training sample would
    increase the test loss (the training sample was helpful).
    
    Args:
        model: Trained neural network
        train_loader: DataLoader for Hessian estimation
        test_sample: (input, target) tuple for test sample
        train_sample: (input, target) tuple for training sample
        device: Computation device
        method: 'lissa' or 'arnoldi'
        **kwargs: Additional arguments for inverse HVP method
        
    Returns:
        Influence score (scalar)
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    # Compute gradient for test sample
    test_input, test_target = test_sample
    test_input = test_input.unsqueeze(0).to(device) if test_input.dim() == 3 else test_input.to(device)
    test_target = test_target.unsqueeze(0).to(device) if test_target.dim() == 0 else test_target.to(device)
    
    model.zero_grad()
    test_output = model(test_input)
    test_loss = criterion(test_output, test_target)
    test_grad = get_gradient_vector(model, test_loss, flatten=True)
    
    # Compute inverse HVP for test gradient
    if method == 'lissa':
        ihvp = compute_inverse_hvp_lissa(
            model, train_loader, test_grad, device, **kwargs
        )
    elif method == 'arnoldi':
        ihvp = compute_inverse_hvp_arnoldi(
            model, train_loader, test_grad, device, **kwargs
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Compute gradient for training sample
    train_input, train_target = train_sample
    train_input = train_input.unsqueeze(0).to(device) if train_input.dim() == 3 else train_input.to(device)
    train_target = train_target.unsqueeze(0).to(device) if train_target.dim() == 0 else train_target.to(device)
    
    model.zero_grad()
    train_output = model(train_input)
    train_loss = criterion(train_output, train_target)
    train_grad = get_gradient_vector(model, train_loss, flatten=True)
    
    # Influence = -ihvp @ train_grad
    influence = -torch.dot(ihvp, train_grad.detach())
    
    return influence.item()


def compute_batch_influence(
    model: nn.Module,
    train_loader: DataLoader,
    samples: List[Tuple[torch.Tensor, torch.Tensor]],
    device: str = 'cuda',
    method: str = 'lissa',
    verbose: bool = True,
    **kwargs
) -> List[float]:
    """
    Compute influences for a batch of samples.
    
    Optimized to share the inverse HVP computation across samples.
    
    Args:
        model: Trained neural network
        train_loader: DataLoader for Hessian estimation
        samples: List of (input, target) tuples
        device: Computation device
        method: 'lissa' or 'arnoldi'
        verbose: Show progress bar
        **kwargs: Additional arguments
        
    Returns:
        List of influence scores
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    influences = []
    iterator = tqdm(samples, desc="Computing influences") if verbose else samples
    
    for sample_input, sample_target in iterator:
        sample_input = sample_input.unsqueeze(0).to(device) if sample_input.dim() == 3 else sample_input.to(device)
        sample_target = sample_target.unsqueeze(0).to(device) if sample_target.dim() == 0 else sample_target.to(device)
        
        model.zero_grad()
        output = model(sample_input)
        loss = criterion(output, sample_target)
        
        # Gradient for this sample
        grad = get_gradient_vector(model, loss, flatten=True).detach()
        
        # Inverse HVP
        if method == 'lissa':
            ihvp = compute_inverse_hvp_lissa(
                model, train_loader, grad, device, **kwargs
            )
        else:
            ihvp = compute_inverse_hvp_arnoldi(
                model, train_loader, grad, device, **kwargs
            )
        
        # Self-influence: how much does this sample influence itself
        influence = torch.dot(ihvp.detach(), grad).item()
        influences.append(influence)
    
    return influences


def compute_influence_on_parameters(
    model: nn.Module,
    train_loader: DataLoader,
    sample: Tuple[torch.Tensor, torch.Tensor],
    device: str = 'cuda',
    method: str = 'lissa',
    **kwargs
) -> torch.Tensor:
    """
    Compute the influence of a sample on model parameters.
    
    Returns H^{-1} @ grad(sample), which represents how much
    each parameter would change if the sample were upweighted.
    
    Args:
        model: Trained neural network
        train_loader: DataLoader for Hessian estimation
        sample: (input, target) tuple
        device: Computation device
        method: 'lissa' or 'arnoldi'
        
    Returns:
        Parameter influence vector
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    sample_input, sample_target = sample
    sample_input = sample_input.unsqueeze(0).to(device) if sample_input.dim() == 3 else sample_input.to(device)
    sample_target = sample_target.unsqueeze(0).to(device) if sample_target.dim() == 0 else sample_target.to(device)
    
    model.zero_grad()
    output = model(sample_input)
    loss = criterion(output, sample_target)
    grad = get_gradient_vector(model, loss, flatten=True).detach()
    
    if method == 'lissa':
        return compute_inverse_hvp_lissa(model, train_loader, grad, device, **kwargs)
    else:
        return compute_inverse_hvp_arnoldi(model, train_loader, grad, device, **kwargs)
