# Selective Machine Unlearning with Information-Theoretic Auditing

**A Unified Framework for Privacy-Certified Forgetting**

This framework implements state-of-the-art machine unlearning methods for selectively removing specific training data from neural networks (both class-level and sample-level). It also introduces a novel **Information-Theoretic Auditing Framework** using Partial Information Decomposition (PID) to certify true information removal via the Residual Knowledge ($I_\cap$) metric.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run interactive demo
python demo.py

# Run comprehensive benchmark (all methods)
python experiments/run_comprehensive.py --dataset cifar10
```

## Project Structure

```text
selective_unlearning/
├── src/
│   ├── config.py              # Experiment configurations
│   ├── training.py            # Base model training
│   ├── data/
│   │   └── dataset.py         # Advanced forget/retain splitting logic
│   ├── models/
│   │   └── classifier.py      # ResNet architectures
│   ├── unlearning/            # 10 Unlearning algorithms
│   │   ├── base.py            # Base unlearner interface
│   │   ├── iweup_v2.py        # IWEUP v2 (Our novel unified method)
│   │   ├── scrub.py           # SCRUB baseline
│   │   ├── salun.py           # SalUn baseline
│   │   └── ...                # (Amnesiac, Fine-tune, Bad Teacher, etc.)
│   ├── auditing/              # Novel Information-Theoretic Auditing
│   │   ├── rine.py            # RINE estimator for I∩ (Residual Knowledge)
│   │   ├── activation_extractor.py # Hook-based representation extraction
│   │   └── difficulty.py      # I∩-based per-sample difficulty scoring
│   └── evaluation/
│       ├── metrics.py         # Standard metrics (Accuracy, etc.)
│       └── mia.py             # Membership Inference Attack (MIA)
├── experiments/               # Experimentation scripts
│   ├── run_audit.py           # Core I∩ auditing pipeline
│   ├── run_stress_test.py     # Robustness stress testing (easy/hard splits)
│   └── run_comprehensive.py   # Full evaluation pipeline
├── paper/                     # LaTeX source, figures, and presentations
├── demo.py                    # Interactive demo script
└── requirements.txt
```

## Implemented Methods

| Method | Type | Reference |
|--------|------|-----------|
| **Retrain** | Exact | Gold Standard |
| **Fine-tune** | Approximate | Baseline |
| **Gradient Ascent** | Approximate | Thudi et al. (2022) |
| **SCRUB** | Approximate | Kurmanji et al. (2023) |
| **SalUn** | Approximate | Fan et al. (2024) |
| **Bad Teacher** | Approximate | Chundawat et al. (2023) |
| **Amnesiac** | Approximate | Graves et al. (2021) |
| **SSD** | Approximate | Foster et al. (2023) |
| **IWEUP v2** | Approximate | **Ours (Novel)** |

## Novel Contributions

### 1. IWEUP v2: Unified Unlearning Framework
Our proposed **Importance-Weighted Entropy Unlearning Protocol (v2)** automatically adapts to both class-level and sample-level unlearning. It relies on a novel three-component loss:
1. **Adaptive Confidence Reduction**: Systematically degrades confidence on forgotten samples.
2. **Anti-Memorization**: Pushes representations away from their memorized state.
3. **MIA-Fooling**: Ensures the model mimics the behavior of a model that never saw the data, consistently achieving **MIA Risk = 0.000**.

### 2. PID-based Auditing Framework
Traditional behavioral metrics (like Forget Accuracy) are misleading. We adapted the **RINE estimator** to vision models (ResNet) to compute **Residual Knowledge ($I_\cap$)**:
- **$I_\cap$ (bits)** measures the exact shared information between the base model and unlearned model representations regarding the forget set.
- Allows for **Difficulty Estimation**: Ranking samples from "easy" to "hard" to evaluate method robustness via our **Stress-Test Protocol**.

## Evaluation Metrics

- **Residual Knowledge ($I_\cap$)**: Bits of information retained from the forget set (lower is better, our primary auditing metric).
- **Forget Accuracy (FA)**: Accuracy on forget set.
- **Retain Accuracy (RA)**: Accuracy on retain set.
- **Test Accuracy (TA)**: Accuracy on unseen test data.
- **MIA Risk**: Vulnerability to Membership Inference Attacks (lower is better, 0.0 means perfect privacy).

## Running Experiments

```bash
# Run the Information-Theoretic Audit
python experiments/run_audit.py --dataset cifar10 --forget_ratio 0.1

# Run the Stress-Test Protocol (Evaluates on Easy/Random/Hard splits)
python experiments/run_stress_test.py --dataset cifar10

# Results are saved to ./outputs/ and figures can be generated via:
python paper/generate_figures.py
```

## Requirements

- Python >= 3.9
- PyTorch >= 2.0
- torchvision
- scikit-learn
- matplotlib
- CUDA/MPS (optional, for acceleration)
