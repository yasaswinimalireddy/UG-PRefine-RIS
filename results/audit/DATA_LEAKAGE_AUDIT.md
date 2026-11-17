# DATA LEAKAGE AUDIT: UG-PRefine-RIS

This document audits data separation, random seeds, target generation, and preprocessing to guarantee zero data leakage between training, validation, and testing.

---

## 1. Data Separation & Seed Isolation

* **Training Dataset Seed:** Seed `42` (Dynamic batch generation in `train_proposed.py`).
* **Validation Dataset Seed:** Seed `100` (Independent channel generation).
* **Frozen Benchmark Test Dataset Seed:** Seed `42000` (Loaded from `data/frozen_test_set_64.pt`).

---

## 2. Leakage Audit Checklist

1. **Test Set Channel Leakage:** **NONE.** Test channels are strictly isolated in `data/frozen_test_set_64.pt` and generated with seed `42000`.
2. **Normalization Statistics Leakage:** **NONE.** Inputs are per-sample normalized: $\mathbf{X}_{\text{norm}} = \mathbf{X} / \|\mathbf{X}\|_F$. No global dataset mean/std is calculated across test samples.
3. **Threshold Selection Leakage:** **NONE.** The 25% uncertainty threshold $\tau_{75}$ is computed purely as a quantile over the test batch's predicted uncertainties without referencing ground-truth AO rate targets.
4. **AO Target Leakage:** **NONE.** Ground-truth AO rate targets are used only during heteroscedastic NLL loss evaluation and benchmark comparison, never as input features to the neural network.
5. **Refinement Parameter Leakage:** PGD step size $\eta=0.10$ and step count $L=10$ were fixed in `configs/default_config.yaml` prior to test evaluation.
6. **Status:** **GREEN (ZERO DATA LEAKAGE VERIFIED)**.
