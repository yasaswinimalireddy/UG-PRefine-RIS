# Research Paper Outline: UG-PRefine-RIS

## Target Journal: IEEE Transactions on Wireless Communications (TWC) / IEEE JSAC
**Title:** UG-PRefine-RIS: An End-to-End Differentiable Predict-and-Refine Neural Framework for Discrete RIS Phase Shift Optimization under Imperfect CSI

---

### Section 1: Abstract
* **Core Problem:** High-dimensional non-convex discrete phase shift design for RIS-assisted MU-MISO networks under hardware quantization and CSI estimation errors.
* **Proposed Solution:** UG-PRefine, a differentiable neural framework combining ResNet spatial feature extraction, Gumbel-Softmax discrete phase relaxation, and unrolled Projected Gradient Descent (PGD) refinement.
* **Key Mechanism:** Epistemic uncertainty head $\hat{\sigma}^2$ quantifies neural prediction confidence, providing a statistically significant signal ($\rho = 0.2254, p < 0.05$) to identify high-difficulty channels for selective refinement.
* **Main Results:** Achieves $0.00\%$ hardware constraint violations, delivers $>140\times$ FLOP speedup over classical Alternating Optimization (AO) at $N=256$, and retains $>99.6\%$ sum-rate at 75% computational savings.
* **Supporting Artifacts:** `results/final/FINAL_RESULTS_TABLE.csv`, `Figure 3`, `Figure 4`.

---

### Section 2: Introduction
* **Background & Motivation:** 6G Reconfigurable Intelligent Surfaces (RIS) require real-time passive beamforming under discrete hardware constraints ($b=2$ bits).
* **Limitations of Existing Work:**
  1. Classical optimization (AO, CD) is computationally intractable for sub-millisecond 6G channels ($>250\text{ ms}$).
  2. Pure neural models lack physical feasibility guarantees or boundary fine-tuning.
  3. Prior predict-and-refine schemes lack uncertainty-aware computational gating.
* **Verified Paper Contributions:**
  1. *Physics-Constrained Neural Predictor:* ResNet + Gumbel-Softmax for discrete 2-bit phase shift prediction.
  2. *Differentiable Unrolled PGD Refinement:* Continuous gradient ascent fine-tuning with nearest-neighbor quantization.
  3. *Epistemic Uncertainty Calibration:* Statistically verified variance estimation correlating with neural optimization gap.
  4. *Pareto Computational Scaling:* $>140\times$ FLOP reduction and low-latency execution.
* **Explicit Claim Boundaries:**
  - *Do NOT claim:* "Uncertainty routing significantly outperforms random selection" ($p = 0.8291$).
  - *Do NOT claim:* "$24.27\text{ bps/Hz}$ direct neural rate" (requires offline AO warm-start).

---

### Section 3: Related Work & Novelty Audit
* **Literature Taxonomy:**
  1. *Classical Iterative Optimization:* Alternating Optimization (Huang et al., 2019), Coordinate Descent.
  2. *Deep Learning for RIS Beamforming:* CNN classification models (Feng et al., 2020), Deep unfolding (Elbir et al., 2020).
  3. *Uncertainty & Conditional Computation:* Neural variance estimation in wireless (Liu et al., 2022).
* **Literature Matrix Reference:** `results/literature/FINAL_LITERATURE_MATRIX.csv`.
* **Differentiated Novelty Statement:** UG-PRefine integrates Gumbel discrete relaxation, unrolled PGD refinement, and epistemic uncertainty gating into a single end-to-end differentiable architecture under imperfect CSI.

---

### Section 4: System Model
* **MU-MISO Wireless System:** BS with $M=8$ antennas, RIS with $N=64$ passive elements, $K=2$ single-antenna users.
* **Channel Formulations:** Direct channel $\mathbf{H}_{\text{BU}} \in \mathbb{C}^{K \times M}$, BS-RIS channel $\mathbf{H}_{\text{BR}} \in \mathbb{C}^{N \times M}$, RIS-User channel $\mathbf{H}_{\text{RU}} \in \mathbb{C}^{K \times N}$.
* **Effective Channel:** $\mathbf{H}_{\text{eff}} = \mathbf{H}_{\text{BU}} + \mathbf{H}_{\text{RU}} \mathbf{\Phi} \mathbf{H}_{\text{BR}}$ where $\mathbf{\Phi} = \operatorname{diag}(e^{j\theta_1}, \dots, e^{j\theta_N})$.
* **Hardware Phase-Amplitude Coupling:** $\beta(\theta_n) = (1 - \beta_{\text{min}}) \left( \frac{\sin(\theta_n - \phi) + 1}{2} \right)^\gamma + \beta_{\text{min}}$.
* **Transmitter Beamforming:** Zero-Forcing (ZF) precoding $\mathbf{W} = \mathbf{H}_{\text{eff}}^H (\mathbf{H}_{\text{eff}} \mathbf{H}_{\text{eff}}^H)^{-1}$ normalized to total power $P_{\text{max}}$.
* **Required Figures:** `Figure 2` (System model diagram).

---

### Section 5: Problem Formulation
* **Achievable Sum-Rate Objective:**
  $$\max_{\mathbf{\theta} \in \mathcal{F}_b^N} \mathcal{R}(\mathbf{\theta}) = \sum_{k=1}^K \log_2 \left( 1 + \operatorname{SINR}_k(\mathbf{\theta}) \right)$$
* **Discrete Phase Constraints:** $\theta_n \in \mathcal{F}_b = \{0, \frac{2\pi}{2^b}, \dots, \frac{2\pi (2^b - 1)}{2^b}\}$.
* **Imperfect CSI Model:** $\mathbf{H}_{\text{noisy}} = \mathbf{H}_{\text{true}} + \mathbf{N}_{\text{CSI}}$, $\mathbf{N}_{\text{CSI}} \sim \mathcal{CN}(0, \delta^2 \mathbf{I})$.

---

### Section 6: Proposed UG-PRefine-RIS Architecture
* **ResNet Feature Extractor:** Input tensor $[2, K \cdot M, N]$, 2D Conv layers + residual blocks.
* **Spatial Self-Attention:** Multi-head attention mechanism capturing long-range inter-element spatial phase correlations.
* **Gumbel-Softmax Discrete Layer:** Continuous relaxation during training $\mathbf{p}_n = \operatorname{softmax}((\mathbf{z}_n + \mathbf{g}) / \tau)$, annealing to discrete selection at inference.
* **Epistemic Uncertainty Head:** Predicts log-variance $\log \hat{\sigma}^2$ trained under negative log-likelihood loss.
* **Unrolled PGD Refinement Layer:** $L=10$ unrolled gradient ascent iterations on continuous phase shifts followed by nearest-neighbor discrete quantization.
* **Selective Computation Router:** Activates PGD refinement only if $\hat{\sigma}^2 \ge \tau_{\text{unc}}$.
* **Required Figures:** `Figure 1` (Architecture schematic).

---

### Section 7: Training Methodology & Loss Functions
* **Differentiable End-to-End Objective:**
  $$\mathcal{L}_{\text{total}} = -\mathbb{E} \left[ \mathcal{R}(\theta_{\text{gumbel}}) \right] + \lambda_1 \mathcal{L}_{\text{NLL}}(\hat{\sigma}^2, |\mathcal{R}_{\text{true}} - \hat{\mathcal{R}}|) + \lambda_2 \mathcal{L}_{\text{PGD}}$$
* **Annealing Schedule:** Temperature $\tau$ annealed from $1.0$ to $0.1$ over 100 epochs.

---

### Section 8: Computational Complexity & Latency Analysis
* **Theoretical FLOP Analysis:**
  - Neural Base: $C_{\text{base}} = \mathcal{O}(M N K + N^2)$ FLOPs ($2.35\text{ M}$).
  - Standalone PGD ($L$ steps): $C_{\text{PGD}} = \mathcal{O}(L \cdot M N K)$ FLOPs ($0.024\text{ M}$).
  - Classical AO: $C_{\text{AO}} = \mathcal{O}(I_{\text{iter}} \cdot N^3)$ FLOPs ($338.18\text{ M}$).
* **FLOP Speedup Factor:** $143.7\times$ at $N=256$.
* **Measured CPU Latency:** Direct neural ($0.45\text{ ms}$), 25% Gated ($1.36\text{ ms}$), AO ($251.93\text{ ms}$).
* **Required Tables & Figures:** `FINAL_COMPLEXITY.csv`, `FINAL_LATENCY.csv`, `Figure 4`, `Figure 5`.

---

### Section 9: Experimental Setup
* **Simulation Parameters:** $N=64$, $M=8$, $K=2$, $b=2$ bits, $\text{SNR} = 10\text{ dB}$, $\delta^2 = 0.05$.
* **Datasets & Seeds:** Frozen 500-sample test set (Seed `42000`).
* **Hardware:** Intel CPU single-thread PyTorch execution.
* **Baselines Evaluated:** No RIS, Random Phase, GumbelRIS, Standalone PGD, Coordinate Descent, Quantized AO (2-bit), Continuous AO.

---

### Section 10: Results & Main Baseline Comparisons
* **Main Sum-Rate Benchmark Table:** `FINAL_RESULTS_TABLE.csv`.
* **Key Metric Results:**
  - Continuous AO (Upper Bound): $24.93\text{ bps/Hz}$
  - Quantized 2-Bit AO: $24.19\text{ bps/Hz}$
  - UG-PRefine Always Refine ($100\%$): $18.53\text{ bps/Hz}$
  - UG-PRefine Selective Refine ($25\%$): $18.46\text{ bps/Hz}$
  - UG-PRefine Direct ($0\%$): $18.42\text{ bps/Hz}$
  - Random Phase: $16.61\text{ bps/Hz}$
  - No RIS: $10.09\text{ bps/Hz}$
* **Statistical Tests:** `FINAL_STATISTICS.csv` (Paired t-tests, CIs, Cohen's d).
* **Required Figures:** `Figure 3` (Main bar benchmark), `Figure 8` (Refinement sweep).

---

### Section 11: Ablation Study & Component Isolation
* **Isolation Table:** `FINAL_ABLATION.csv`.
* **Component Deductions:**
  - Spatial Attention: $+0.83\text{ bps/Hz}$
  - Gumbel-Softmax Quantization: $+0.20\text{ bps/Hz}$
  - Uncertainty Training: $+0.37\text{ bps/Hz}$
  - PGD Refinement ($L=10$): $+0.11\text{ bps/Hz}$
* **Required Figures:** `Figure 9` (Ablation chart).

---

### Section 12: Robustness & Generalization Analysis
* **CSI Estimation Noise Robustness:** `FINAL_ROBUSTNESS.csv` ($\delta^2 \in [0.00, 0.20]$).
* **Phase Quantization Resolution:** 1-bit ($16.10\text{ bps/Hz}$), 2-bit ($18.42\text{ bps/Hz}$), 3-bit ($19.45\text{ bps/Hz}$).
* **Uncertainty Calibration:** `results/final/uncertainty_correlation_breakdown.csv` ($\rho = 0.2254, p = 0.0241$).
* **Required Figures:** `Figure 6` (CSI noise plot), `Figure 7` (Uncertainty scatter), `Figure 10` (Phase resolution bar plot).

---

### Section 13: Limitations & Open Research Questions
1. *Magnitude of Unrolled Gradient Ascent:* $L=10$ PGD unrolling yields $+0.11\text{ bps/Hz}$ average gain on discrete boundary points.
2. *Uncertainty Routing Performance:* Uncertainty gating identifies difficult samples, but under short PGD steps ($L=10$), rate performance is statistically comparable to random selection ($p = 0.8291$).
3. *Autograd Graph Latency Overhead:* PyTorch CPU autograd graph building adds $3.67\text{ ms}$ overhead.

---

### Section 14: Conclusion & Future Work
* **Summary:** UG-PRefine-RIS provides an end-to-end differentiable predict-and-refine framework delivering guaranteed physical feasibility ($0.00\%$ violations), $>140\times$ FLOP speedup, and sub-millisecond execution.
* **Future Work:** GPU C++ custom CUDA kernel deployment for unrolled PGD steps and multi-surface RIS coordination.
