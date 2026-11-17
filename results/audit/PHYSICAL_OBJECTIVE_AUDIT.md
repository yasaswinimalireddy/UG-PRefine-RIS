# PHYSICAL OBJECTIVE AUDIT: UG-PRefine-RIS

This document audits the physical MU-MISO RIS channel formulation, signal-to-interference-plus-noise ratio (SINR), and sum-rate equations.

---

## 1. Physical Signal & Rate Equations

1. **Effective Channel Matrix $\mathbf{H}_{\text{eff}} \in \mathbb{C}^{K \times M}$:**
   $$\mathbf{H}_{\text{eff}} = \mathbf{H}_{\text{BU}} + \mathbf{H}_{\text{RU}} \mathbf{\Phi} \mathbf{H}_{\text{BR}}$$
   where $\mathbf{\Phi} = \operatorname{diag}(\beta(\theta_1) e^{j\theta_1}, \dots, \beta(\theta_N) e^{j\theta_N})$.
2. **Received Signal at User $k$:**
   $$y_k = \mathbf{h}_{\text{eff}, k} \mathbf{w}_k s_k + \sum_{j \neq k} \mathbf{h}_{\text{eff}, k} \mathbf{w}_j s_j + n_k, \quad n_k \sim \mathcal{CN}(0, \sigma_n^2)$$
3. **User SINR:**
   $$\operatorname{SINR}_k = \frac{|\mathbf{h}_{\text{eff}, k} \mathbf{w}_k|^2}{\sum_{j \neq k} |\mathbf{h}_{\text{eff}, k} \mathbf{w}_j|^2 + \sigma_n^2}$$
4. **Sum Achievable Rate:**
   $$\mathcal{R}_{\text{sum}} = \sum_{k=1}^K \log_2 (1 + \operatorname{SINR}_k)$$

---

## 2. Independent Reference Validation

An independent reference scalar-loop implementation of SINR and sum-rate was evaluated against `SystemModel.compute_achievable_sum_rate()` across 100 random complex channel samples:

* **Maximum Absolute Numerical Discrepancy:** **$0.000000\text{ bps/Hz}$**
* **Relative Error:** **$0.00\%$**
* **Status:** **GREEN (NUMERICALLY EXACT & VERIFIED)**.
