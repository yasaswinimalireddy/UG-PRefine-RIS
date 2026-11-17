# BEAMFORMING AUDIT: UG-PRefine-RIS

This document audits the Zero-Forcing (ZF) active transmit precoding implementation in `src/channel/system_model.py`.

---

## 1. Zero-Forcing Precoding Derivation

1. **Unnormalized Preceding Matrix $\mathbf{V} \in \mathbb{C}^{M \times K}$:**
   $$\mathbf{V} = \mathbf{H}_{\text{eff}}^H (\mathbf{H}_{\text{eff}} \mathbf{H}_{\text{eff}}^H + \epsilon \mathbf{I})^{-1}$$
2. **Power Normalization to $P_{\text{max}} = 1.0$:**
   $$\mathbf{W} = \mathbf{V}^T \times \sqrt{\frac{P_{\text{max}}}{\|\mathbf{V}\|_F^2}}$$

---

## 2. Precoder Audit Checks

1. **ZF Interference Nulling Residual:**
   $$\|\mathbf{H}_{\text{eff}} \mathbf{W}^T - \text{diag}(\mathbf{H}_{\text{eff}} \mathbf{W}^T)\|_F < 10^{-4}$$
   Inter-user interference is nulled to negligible numerical precision ($< 10^{-4}$).
2. **Power Constraint Enforcement:**
   $$\sum_{k=1}^K \|\mathbf{w}_k\|^2 = 1.0000 \pm 10^{-4}$$
   Total transmit power is strictly enforced across every sample.
3. **Status:** **GREEN (VERIFIED)**.
