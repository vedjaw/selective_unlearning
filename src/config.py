"""
Configuration management for unlearning experiments.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Literal
from pathlib import Path


@dataclass
class DataConfig:
    """Dataset configuration."""
    name: Literal["cifar10", "cifar100", "svhn", "celeba"] = "cifar10"
    data_dir: str = "./data"
    batch_size: int = 128
    num_workers: int = 4
    
    # Forget set configuration
    forget_type: Literal["class", "random", "user"] = "class"
    forget_classes: List[int] = field(default_factory=lambda: [0])  # For class-wise
    forget_ratio: float = 0.1  # For random sampling
    forget_indices: Optional[List[int]] = None  # For specific samples
    
    # Data augmentation
    use_augmentation: bool = True


@dataclass
class ModelConfig:
    """Model configuration."""
    architecture: Literal["resnet18", "resnet34", "vgg16", "mobilenetv2"] = "resnet18"
    pretrained: bool = False
    num_classes: int = 10
    
    # Checkpoint paths
    checkpoint_dir: str = "./checkpoints"
    save_frequency: int = 10  # Save every N epochs


@dataclass
class TrainingConfig:
    """Training configuration."""
    epochs: int = 100
    learning_rate: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    
    # Learning rate schedule
    lr_scheduler: Literal["step", "cosine", "none"] = "step"
    lr_decay_epochs: List[int] = field(default_factory=lambda: [50, 75])
    lr_decay_factor: float = 0.1
    
    # Early stopping
    patience: int = 10


@dataclass
class UnlearningConfig:
    """Unlearning algorithm configuration."""
    method: Literal["retrain", "finetune", "gradient_ascent", "ssd", "pgu", "scrub", "igtu"] = "igtu"
    
    # General unlearning params
    unlearn_epochs: int = 10
    unlearn_lr: float = 0.01
    
    # SSD-specific
    ssd_lambda: float = 1.0
    ssd_alpha: float = 0.1
    
    # PGU-specific
    pgu_project_dim: int = 100
    
    # SCRUB-specific
    scrub_alpha: float = 0.5
    scrub_gamma: float = 0.99
    
    # IGTU-specific (our method)
    igtu_influence_samples: int = 1000  # Samples for influence estimation
    igtu_lissa_depth: int = 5000  # LiSSA recursion depth
    igtu_lissa_scale: float = 25.0  # LiSSA damping
    igtu_retention_weight: float = 1.0  # Weight for retention loss


@dataclass
class InfluenceConfig:
    """Influence function configuration."""
    method: Literal["lissa", "arnoldi", "exact"] = "lissa"
    
    # LiSSA params
    lissa_depth: int = 5000
    lissa_samples: int = 1
    lissa_scale: float = 25.0
    
    # Arnoldi params
    arnoldi_dim: int = 200
    
    # General
    recursion_batch_size: int = 128
    use_gpu: bool = True


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    # Membership Inference Attack
    run_mia: bool = True
    mia_shadow_models: int = 3
    mia_attack_type: Literal["confidence", "label_only", "loss"] = "confidence"
    
    # Metrics to compute
    compute_influence_residual: bool = True
    compute_tsne: bool = True
    
    # Logging
    log_frequency: int = 1
    use_wandb: bool = False
    wandb_project: str = "selective-unlearning"


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    name: str = "default"
    seed: int = 42
    device: str = "cuda"
    
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    unlearning: UnlearningConfig = field(default_factory=UnlearningConfig)
    influence: InfluenceConfig = field(default_factory=InfluenceConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    # Output
    output_dir: str = "./outputs"
    
    def __post_init__(self):
        """Create output directories."""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.model.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.data.data_dir).mkdir(parents=True, exist_ok=True)


def get_cifar10_config(forget_class: int = 0) -> ExperimentConfig:
    """Get default config for CIFAR-10 experiments."""
    return ExperimentConfig(
        name=f"cifar10_forget_class_{forget_class}",
        data=DataConfig(
            name="cifar10",
            forget_type="class",
            forget_classes=[forget_class],
        ),
        model=ModelConfig(
            architecture="resnet18",
            num_classes=10,
        ),
    )


def get_cifar100_config(forget_classes: List[int] = None) -> ExperimentConfig:
    """Get default config for CIFAR-100 experiments."""
    if forget_classes is None:
        forget_classes = [0, 1, 2, 3, 4]  # Forget 5 classes
    return ExperimentConfig(
        name=f"cifar100_forget_{len(forget_classes)}_classes",
        data=DataConfig(
            name="cifar100",
            forget_type="class",
            forget_classes=forget_classes,
        ),
        model=ModelConfig(
            architecture="resnet18",
            num_classes=100,
        ),
    )
