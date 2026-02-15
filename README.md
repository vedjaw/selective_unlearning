# Selective Model Unlearning Framework

**Influence-Guided Targeted Unlearning (IGTU) for Neural Networks**

This framework implements state-of-the-art machine unlearning methods for selectively removing the influence of specific training data from neural networks while preserving performance on retained data.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run demo
python demo.py

# Run full experiment
python experiments/exp_class_unlearning.py --dataset cifar10 --forget_class 0
```

## Project Structure

```
selective_unlearning/
├── src/
│   ├── config.py              # Experiment configuration
│   ├── training.py            # Training utilities
│   ├── data/
│   │   └── dataset.py         # Data loading and forget/retain splitting
│   ├── models/
│   │   └── classifier.py      # Model architectures (ResNet, VGG)
│   ├── unlearning/
│   │   ├── base.py            # Base unlearner class
│   │   ├── fine_tune.py       # Fine-tuning baseline
│   │   ├── ssd.py             # Selective Synaptic Dampening
│   │   ├── pgu.py             # Projected Gradient Unlearning
│   │   ├── scrub.py           # SCRUB method
│   │   └── igtu.py            # IGTU (our novel method)
│   ├── influence/
│   │   └── influence_functions.py  # LiSSA, Arnoldi IHVP estimation
│   └── evaluation/
│       ├── metrics.py         # Evaluation metrics
│       └── mia.py             # Membership Inference Attack
├── experiments/
│   └── exp_class_unlearning.py  # Main experiment script
├── demo.py                    # Quick demo
└── requirements.txt           # Dependencies
```

## Implemented Methods

| Method | Description | Reference |
|--------|-------------|-----------|
| **Fine-tune** | Fine-tune on retain set only | Baseline |
| **Gradient Ascent** | Gradient ascent on forget + descent on retain | Baseline |
| **SSD** | Selective Synaptic Dampening | Foster et al. |
| **PGU** | Projected Gradient Unlearning | Hoang et al. 2024 |
| **SCRUB** | Teacher-student distillation | Kurmanji et al. |
| **IGTU** | Influence-Guided Targeted Unlearning | **Ours** |

## Novel Contribution: IGTU

Our proposed **Influence-Guided Targeted Unlearning (IGTU)** combines:

1. **Influence-weighted unlearning loss**: Samples with higher influence get more aggressive unlearning
2. **Gradient projection**: Unlearning gradients are projected orthogonally to retain subspace
3. **Approximate certified bounds**: Residual influence estimation provides unlearning guarantees

```python
from src.unlearning import IGTUUnlearner

unlearner = IGTUUnlearner(
    model,
    device='cuda',
    epochs=10,
    influence_samples=100,
    project_dim=100,
)
unlearned_model = unlearner.unlearn(forget_loader, retain_loader)

# Get unlearning certificate
certificate = unlearner.get_certificate()
print(f"Residual influence bound: {certificate['bound_95']}")
```

## Evaluation Metrics

- **Forget Accuracy (FA)**: Accuracy on forget set (lower is better)
- **Retain Accuracy (RA)**: Accuracy on retain set (higher is better)
- **Unlearning Accuracy (UA)**: Drop in forget accuracy after unlearning
- **MIA AUC**: Membership Inference Attack AUC on forget set (lower is better)
- **Residual Influence**: Upper bound on remaining data influence (IGTU only)

## Running Experiments

```bash
# Class-wise unlearning on CIFAR-10
python experiments/exp_class_unlearning.py \
    --dataset cifar10 \
    --forget_class 0 \
    --model resnet18 \
    --epochs 100 \
    --unlearn_epochs 10 \
    --methods finetune gradient_ascent ssd pgu igtu

# Results saved to ./outputs/
```

## Requirements

- Python >= 3.9
- PyTorch >= 2.0
- CUDA (optional, for GPU acceleration)

## Citation

If you use this framework in your research, please cite:

```bibtex
@article{jawandhia2025igtu,
  title={Influence-Guided Targeted Unlearning for Neural Networks},
  author={Jawandhia, Vedant},
  year={2025}
}
```

## License

MIT License
