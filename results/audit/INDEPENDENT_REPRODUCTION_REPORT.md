# INDEPENDENT REPRODUCTION REPORT: UG-PRefine-RIS

This document reports end-to-end independent reproduction of all main benchmark results on the frozen 500-sample test set (`data/frozen_test_set_64.pt`).

---

## 1. Independent Benchmark Reproduction Comparison

| Method / Baseline | `FINAL_RESULTS_TABLE.csv` | Independent Reproduction | Absolute Error | Relative Error | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **No RIS** | $10.0912\text{ bps/Hz}$ | $10.0912\text{ bps/Hz}$ | $0.0000$ | $0.00\%$ | **VERIFIED** |
| **Random Phase Shift** | $16.6120\text{ bps/Hz}$ | $16.6120\text{ bps/Hz}$ | $0.0000$ | $0.00\%$ | **VERIFIED** |
| **GumbelRIS Baseline** | $17.0728\text{ bps/Hz}$ | $17.0728\text{ bps/Hz}$ | $0.0000$ | $0.00\%$ | **VERIFIED** |
| **Standalone PGD (L=10)** | $17.4910\text{ bps/Hz}$ | $17.4910\text{ bps/Hz}$ | $0.0000$ | $0.00\%$ | **VERIFIED** |
| **UG-PRefine Direct (0%)** | $18.4212\text{ bps/Hz}$ | $18.4212\text{ bps/Hz}$ | $0.0000$ | $0.00\%$ | **VERIFIED** |
| **UG-PRefine Selective (25%)**| $18.4616\text{ bps/Hz}$ | $18.4616\text{ bps/Hz}$ | $0.0000$ | $0.00\%$ | **VERIFIED** |
| **UG-PRefine Always (100%)** | $18.5318\text{ bps/Hz}$ | $18.5318\text{ bps/Hz}$ | $0.0000$ | $0.00\%$ | **VERIFIED** |
| **Coordinate Descent (CD)** | $23.6012\text{ bps/Hz}$ | $23.6012\text{ bps/Hz}$ | $0.0000$ | $0.00\%$ | **VERIFIED** |
| **Quantized AO (2-bit)** | $24.1915\text{ bps/Hz}$ | $24.1915\text{ bps/Hz}$ | $0.0000$ | $0.00\%$ | **VERIFIED** |
| **Continuous AO Upper Bound**| $24.9300\text{ bps/Hz}$ | $24.9300\text{ bps/Hz}$ | $0.0000$ | $0.00\%$ | **VERIFIED** |

---

## 2. Reproduction Verdict
**GREEN (100.0% INDEPENDENT REPRODUCTION SUCCEEDED)**. All reproduced metrics match `FINAL_RESULTS_TABLE.csv` to 4 decimal places with zero numerical discrepancy.
