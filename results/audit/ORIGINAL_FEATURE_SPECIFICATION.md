# ORIGINAL FEATURE SPECIFICATION: UG-PRefine-RIS

This document establishes the original intended software and algorithmic specification for **UG-PRefine-RIS** based on code inspection, configuration manifests, and system requirements.

---

## 1. Intended System Model & Wireless Environment

1. **MU-MISO System Model:** $M$ Base Station (BS) antennas, $N$ RIS passive elements, $K$ single-antenna mobile users.
2. **Channel Generation:** Rician LoS link for BS-RIS ($\mathbf{H}_{\text{BR}}$), Rayleigh/correlated NLoS links for RIS-UE ($\mathbf{H}_{\text{RU}}$) and direct path BS-UE ($\mathbf{H}_{\text{BU}}$).
3. **Spatial Correlation:** Spatial correlation matrix $\mathbf{R}_{\text{RIS}}$ with exponential spatial correlation $R_{ij} = \kappa^{|i-j|}$.
4. **Imperfect CSI Error:** Complex Gaussian estimation noise $\mathbf{E} \sim \mathcal{CN}(0, \delta^2 \mathbf{I})$ added to true CSI during dataset generation and model input.
5. **Nonlinear Phase-Amplitude Coupling:** Dynamic element reflection amplitude $\beta(\theta_n) = (1 - \beta_{\text{min}}) \left( \frac{\sin(\theta_n - \phi) + 1}{2} \right)^\gamma + \beta_{\text{min}}$.
6. **Zero-Forcing Beamforming:** BS active transmit precoder $\mathbf{W} = \mathbf{H}_{\text{eff}}^H (\mathbf{H}_{\text{eff}} \mathbf{H}_{\text{eff}}^H)^{-1}$ normalized to total power $P_{\text{max}}$.

---

## 2. Intended Deep Learning Architecture

7. **2D Spatial Representation:** Input tensor $\mathbf{X} \in \mathbb{R}^{B \times 2 \times K \cdot M \times N}$ stacking real and imaginary parts of cascaded CSI.
8. **ResNet Backbone:** 2D Conv stem, 2D Residual Blocks (`ResBlock2D`) with BatchNorm and LeakyReLU activations.
9. **Spatial Self-Attention Module:** Query, Key, Value 2D convolutions computing inter-element spatial phase coupling across the RIS surface.
10. **Epistemic Uncertainty Head:** Linear layers + Softplus activation predicting heteroscedastic variance $\hat{\sigma}^2 > 0$.
11. **Gumbel-Softmax Categorical Quantizer:** Temperature-annealed softmax relaxation during backprop, Straight-Through Estimator (STE) during inference.
12. **Discrete $b$-Bit Phase Constraints:** Hard nearest-neighbor projection to discrete phase alphabet $\mathcal{F}_b = \{0, \frac{2\pi}{2^b}, \dots, \frac{2\pi (2^b - 1)}{2^b}\}$.
13. **Uncertainty-Gated Router:** Conditional execution triggering refinement only for samples where $\hat{\sigma}^2 \ge \tau_{\text{unc}}$.
14. **Differentiable Unrolled PGD Refinement:** $L$ steps of continuous autograd-driven gradient ascent on achievable rate, followed by discrete projection.

---

## 3. Intended Objectives, Solvers, & Baselines

15. **Composite Achievable-Rate Objective:** Negative mean sum-rate loss $-\mathbb{E}[\mathcal{R}_{\text{sum}}]$.
16. **Heteroscedastic Uncertainty Loss:** Gaussian Negative Log-Likelihood $\mathcal{L}_{\text{NLL}} = \frac{|\mathcal{R}_{\text{target}} - \mathcal{R}_{\text{pred}}|^2}{2\hat{\sigma}^2} + 0.5 \log \hat{\sigma}^2$.
17. **Multi-Start Offline AO Solver:** Classical Alternating Optimization solver.
18. **Quantized 2-Bit AO Baseline:** Discrete phase AO optimizer.
19. **Coordinate Descent Baseline:** Discrete non-decreasing rate CD solver.
20. **Standalone PGD Baseline:** Iterative unrolled PGD baseline.
21. **GumbelRIS Baseline:** Deep CNN baseline without unrolled refinement.
22. **Frozen Test Set:** 500 deterministic test samples generated with seed `42000`.
23. **Robustness & OOD Testing:** Sweeps across CSI noise $\delta^2$, array sizes $N$, and quantization bits $b$.
