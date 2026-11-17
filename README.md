# CNN-Assisted Phase Shift Design for RIS-Aided 6G Networks
## UG-PRefine: Uncertainty-Gated Predict-Refine Framework for Physics-Constrained RIS Phase Optimization

---

## 📌 Project Overview
This repository contains a publication-grade, reproducible PyTorch implementation of **UG-PRefine-RIS**, an **Uncertainty-Gated Differentiable Predict-Refine Neural Network** for real-time discrete (1-bit / 2-bit / 3-bit) phase shift optimization in passive Reconfigurable Intelligent Surface (RIS) aided Multi-User MISO (MU-MISO) 6G wireless networks.

### Primary Research Hypothesis
> **Hypothesis:** Combining 2D CNN spatial feature extraction, epistemic uncertainty estimation, and a differentiable unrolled Projected Gradient Descent (PGD) refinement layer achieves $>95\%$ of optimal continuous-phase Alternating Optimization (AO) sum-rate utility while reducing online inference execution latency by over $100\times$ under hardware phase quantization and imperfect Channel State Information (CSI).

---

## 🏗️ Repository Architecture

```
DL Class Project/
├── configs/
│   ├── default_config.yaml         # Main experiment configuration
│   ├── system_model_64.yaml        # 64-element RIS setting
│   └── system_model_128.yaml       # 128-element RIS setting
├── data/
│   └── dataset_loader.py           # DeepMIMO/Sionna-compatible PyTorch dataset loader
├── src/
│   ├── channel/                    # Physical System & Channel Modeling
│   │   ├── system_model.py         # Hardware model, phase-amplitude coupling beta(theta), ZF beamforming
│   │   ├── ris_channel.py          # Rician/Rayleigh spatial correlation channel generator
│   │   └── csi_noise.py            # Imperfect CSI noise model
│   ├── models/                     # Deep Neural Network Architecture
│   │   ├── backbone_cnn.py         # 2D ResNet spatial feature backbone
│   │   ├── spatial_attention.py    # Inter-element spatial self-attention module
│   │   ├── uncertainty_head.py     # Epistemic variance head sigma^2
│   │   ├── gumbel_quantizer.py     # Temperature-annealed Gumbel-Softmax quantizer
│   │   └── ug_prefine_model.py     # Proposed UG-PRefine unified model
│   ├── losses/                     # Differentiable Loss Functions
│   │   ├── rate_loss.py            # Unsupervised Achievable Rate Loss -E[R_sum]
│   │   ├── uncertainty_loss.py     # Heteroscedastic Gaussian NLL calibration loss
│   │   └── composite_loss.py       # Multi-objective total loss
│   ├── optimizers/                 # Iterative Classical Optimizers
│   │   ├── alternating_optimization.py # Continuous & Quantized AO solver
│   │   ├── projected_gradient.py   # Projected Gradient Descent (PGD)
│   │   └── coordinate_descent.py   # Element-by-element Coordinate Descent (CD)
│   ├── baselines/                  # Comparative Deep Learning Baselines
│   │   ├── dnn_baseline.py         # Vectorized Fully Connected MLP
│   │   ├── cnn_baseline.py         # Standard 2D CNN
│   │   └── gumbelris_baseline.py   # Reproduction of GumbelRIS (Zhang et al., 2025)
│   └── utils/                      # Utilities & Benchmarking
│       ├── metrics.py              # Latency, Spectral Efficiency, Violation Rate
│       └── logger.py               # Console & File Logger
├── experiments/                    # Run Scripts
│   ├── train_proposed.py           # Model training pipeline
│   ├── run_baseline_eval.py        # 11-baseline comparative benchmarking suite
│   ├── run_ablation.py             # Component ablation experiments (M0-M2)
│   ├── run_ood_generalization.py   # Out-of-Distribution CSI noise generalization
│   └── run_complexity_bench.py     # FLOPs and execution latency benchmarking
├── tests/                          # Unit Test Suite
│   ├── test_channel_model.py
│   ├── test_loss_gradients.py
│   └── test_quantization.py
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start & Execution Guide

### 1. Run Unit Test Suite
Verify physical channel formulas, gradient flow, and phase quantization:
```bash
python -m unittest discover tests/
```

### 2. Train the Proposed UG-PRefine Model
Train the neural network using physics-aware sum-rate loss and heteroscedastic uncertainty calibration:
```bash
python experiments/train_proposed.py
```

### 3. Evaluate Against Baseline Suite
Benchmark the trained model against 10 published baselines (AO, PGD, CD, GumbelRIS, DNN, CNN):
```bash
python experiments/run_baseline_eval.py
```

### 4. Run Mandatory Component Ablations
Evaluate contribution of each architectural block:
```bash
python experiments/run_ablation.py
```

### 5. Run Out-of-Distribution Generalization Benchmark
Test model robustness across unseen CSI estimation noise levels ($\delta^2 \in [0.0, 0.20]$):
```bash
python experiments/run_ood_generalization.py
```

### 6. Computational Complexity & FLOPs Benchmark
Calculate parameter counts, theoretical FLOPs, and speedup factors:
```bash
python experiments/run_complexity_bench.py
```

---

## 📊 Summary of Baseline Methods Compared

1. **No RIS (Direct Path Only)**
2. **Uniform Random Phase**
3. **Continuous Phase Alternating Optimization (AO Upper Bound)**
4. **Quantized 2-Bit Alternating Optimization**
5. **Projected Gradient Descent (PGD)**
6. **Coordinate Descent (CD)**
7. **Fully-Connected Deep Neural Network (DNN)**
8. **Standard 2D CNN**
9. **GumbelRIS (Zhang et al., 2025)**
10. **UG-PRefine-RIS (Our Proposed Framework)**

---

## 📜 Citation & License
Developed as a research-engineering framework for 6G RIS-aided wireless communication networks.
