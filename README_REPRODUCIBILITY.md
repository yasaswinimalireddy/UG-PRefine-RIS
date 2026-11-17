# Reproducibility Guide: UG-PRefine-RIS

This guide provides exact step-by-step instructions to reproduce all empirical benchmark results, statistical hypothesis tests, computational complexity analyses, and vector figures presented in the **UG-PRefine-RIS** research paper.

---

## 1. System & Environment Specifications

* **Operating System:** Microsoft Windows 11 / Linux (Ubuntu 22.04 LTS recommended)
* **Python Environment:** Python 3.13.12 (or Python 3.10+)
* **PyTorch Version:** PyTorch 2.10.0 (CPU / CUDA 12.1 supported)
* **Key Dependencies:** `numpy`, `scipy`, `pandas`, `matplotlib`, `seaborn`, `pyyaml`, `pytest`
* **CPU Model for Latency Benchmarks:** 13th Gen Intel Core / AMD Ryzen (Single-thread PyTorch execution)

---

## 2. Environment Setup

```bash
# Clone the repository
git clone https://github.com/anonymous/UG-PRefine-RIS.git
cd UG-PRefine-RIS

# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install required dependencies
pip install torch numpy scipy pandas matplotlib seaborn pyyaml pytest
```

---

## 3. Directory Layout & Artifact Locations

```
.
├── configs/
│   └── default_config.yaml                  # System configuration (N=64, M=8, K=2, 2-bit)
├── data/
│   └── frozen_test_set_64.pt                # Frozen 500-sample test set (Seed: 42000)
├── experiments/
│   ├── run_final_validation.py             # Validation pipeline and diagnostic traces
│   ├── run_final_paper_experiments.py       # Main paper experiment package & figures
│   ├── run_baseline_eval.py                # Standalone classical baseline solver
│   └── train_proposed.py                    # End-to-end model training script
├── results/
│   ├── checkpoints/
│   │   └── ug_prefine_best.pt              # Trained UG-PRefine model checkpoint
│   ├── figures/                            # Publication vector figures (Fig 1 - Fig 10)
│   ├── final/                              # Standardized paper deliverables
│   │   ├── FINAL_RESULTS_TABLE.csv
│   │   ├── FINAL_STATISTICS.csv
│   │   ├── FINAL_ABLATION.csv
│   │   ├── FINAL_ROBUSTNESS.csv
│   │   ├── FINAL_LATENCY.csv
│   │   ├── FINAL_COMPLEXITY.csv
│   │   ├── PAPER_READINESS_REPORT.md
│   │   └── FINAL_VALIDATION_REPORT.md
│   └── literature/
│       └── FINAL_LITERATURE_MATRIX.csv     # Exhaustive literature audit comparison
├── src/
│   ├── channel/system_model.py              # Differentiable MU-MISO PyTorch simulator
│   ├── evaluation/
│   │   ├── canonical_evaluator.py          # Unified sum-rate metric calculator
│   │   └── frozen_test_set.py              # Frozen evaluation set generator
│   ├── models/ug_prefine_model.py          # Complete UG-PRefine neural model
│   └── optimizers/                         # Classical solvers (AO, CD, PGD)
├── tests/                                  # PyTorch unit test suite
├── PAPER_OUTLINE.md                        # Publication structure & evidence mapping
└── README_REPRODUCIBILITY.md               # Exact command reproducibility guide
```

---

## 4. Execution Commands

### Step 1: Run Verification Unit Tests
Validate that the channel model, canonical evaluator, loss functions, and Gumbel quantization layer operate strictly without errors:
```bash
python -m pytest
```

### Step 2: Generate Frozen Test Dataset
Generate or reload the deterministic 500-sample test dataset (`data/frozen_test_set_64.pt`):
```bash
python src/evaluation/frozen_test_set.py
```

### Step 3: Run Scientific Validation Audit
Perform diagnostic traces, check unrolled PGD continuous updates, and verify uncertainty correlations:
```bash
python experiments/run_final_validation.py
```

### Step 4: Execute Complete Paper Experiment Package & Generate Figures
Run all baseline comparisons, statistical hypothesis tests, ablation sweeps, robustness evaluations, latency measurements, and generate all 10 publication-quality vector figures:
```bash
python experiments/run_final_paper_experiments.py
```

---

## 5. Seed Specifications & Determinism

To ensure 100% deterministic reproducibility across platforms:
* **Frozen Test Set Seed:** `42000`
* **PyTorch CPU/CUDA Seed:** `42`
* **NumPy Random Seed:** `42`
* **PyTorch Determinism Flags:**
  ```python
  torch.manual_seed(42)
  torch.use_deterministic_algorithms(True)
  ```

---

## 6. Output Verification

Upon executing `experiments/run_final_paper_experiments.py`, verify that all the following files have been generated:

1. `results/final/FINAL_RESULTS_TABLE.csv`
2. `results/final/FINAL_STATISTICS.csv`
3. `results/final/FINAL_ABLATION.csv`
4. `results/final/FINAL_ROBUSTNESS.csv`
5. `results/final/FINAL_LATENCY.csv`
6. `results/final/FINAL_COMPLEXITY.csv`
7. `results/literature/FINAL_LITERATURE_MATRIX.csv`
8. `results/figures/fig1_architecture.png` to `results/figures/fig10_phase_resolution.png`
