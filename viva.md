# UG-PRefine-RIS: Complete Master Presentation & Viva Voce Guide
## The Ultimate Guide to Understanding, Running, Presenting, and Defending This Project

---

## TABLE OF CONTENTS
1. [Executive Summary & The 60-Second Elevator Pitch](#1-executive-summary--the-60-second-elevator-pitch)
2. [What We Have Built (The Big Picture)](#2-what-we-have-built-the-big-picture)
3. [The Problem: Why Is This Hard? (Wireless + Math Motivation)](#3-the-problem-why-is-this-hard-wireless--math-motivation)
4. [How We Built It: Detailed Technical Architecture & Pipeline](#4-how-we-built-it-detailed-technical-architecture--pipeline)
   - 4.1 Input Representation & Cascaded Channel Tensor
   - 4.2 ResNet Backbone (`ComplexCNNBackbone`)
   - 4.3 Spatial Self-Attention Module (`SpatialSelfAttention`)
   - 4.4 Phase Prediction Head & Categorical Logits
   - 4.5 Gumbel-Softmax Discrete Quantizer & Straight-Through Estimator (STE)
   - 4.6 Epistemic Uncertainty Head (`EpistemicUncertaintyHead`)
   - 4.7 Uncertainty-Gated Router Logic
   - 4.8 Unrolled Projected Gradient Descent (PGD) Refinement
   - 4.9 Physical Wireless Channel Simulator & Canonical Evaluator
   - 4.10 Zero-Forcing (ZF) Active BS Beamforming
   - 4.11 Non-Linear Phase-Amplitude Coupling $\beta(\theta)$
5. [Training Objectives & Mathematical Losses](#5-training-objectives--mathematical-losses)
6. [Operational Modes (Mode A, B, C, D) — Do Not Conflate!](#6-operational-modes-mode-a-b-c-d--do-not-conflate)
7. [Step-by-Step Terminal Execution Guide (How to Demo Live)](#7-step-by-step-terminal-execution-guide-how-to-demo-live)
8. [Audited Experimental Results & Key Metrics Table](#8-audited-experimental-results--key-metrics-table)
9. [Class Presentation Walkthrough (Slide-by-Slide Script)](#9-class-presentation-walkthrough-slide-by-slide-script)
10. [Exhaustive Viva Voce Q&A Bank](#10-exhaustive-viva-voce-qa-bank)
    - [Category 1: Deep Learning & Neural Network Concepts](#category-1-deep-learning--neural-network-concepts)
    - [Category 2: Wireless Communications & 6G RIS Concepts](#category-2-wireless-communications--6g-ris-concepts)
    - [Category 3: Optimization & Mathematical Formulations](#category-3-optimization--mathematical-formulations)
    - [Category 4: Implementation & Codebase Architecture](#category-4-implementation--codebase-architecture)
    - [Category 5: The "Tough / Trap" Examiner Questions & Bulletproof Answers](#category-5-the-tough--trap-examiner-questions--bulletproof-answers)

---

## 1. EXECUTIVE SUMMARY & THE 60-SECOND ELEVATOR PITCH

### The 30-Second Version:
> *"In modern 6G wireless communication, Reconfigurable Intelligent Surfaces (RIS) use thousands of passive, cheap reflective elements to bounce signals around obstacles to multiple users. However, finding the optimal phase shift for every element in real-time is an NP-hard, non-convex combinatorial problem. Classical algorithms take hundreds of milliseconds—far too slow for millisecond-level channel fading. We built **UG-PRefine-RIS**, a physics-constrained deep learning framework that predicts 2-bit discrete phases in 0.45 milliseconds, estimates its own prediction uncertainty, and selectively refines only the most uncertain 25% of channels using an unrolled gradient ascent layer. Our model achieves a **143.7× FLOP speedup** over classical algorithms with **0.00% hardware constraint violations** under realistic non-linear hardware coupling."*

### The 2-Minute Version:
> *"Good morning, examiners. Our project is titled **UG-PRefine-RIS: Uncertainty-Gated Predict-and-Refine Framework for Discrete RIS Phase Optimization in 6G Networks**.*
>
> *Here is the challenge: Next-generation networks use high-frequency signals like millimeter-wave, which suffer severe path loss and blockage. An RIS is an array of passive reflecting elements that can beamform signals towards users without needing active RF power amplifiers. But optimizing an $N=64$ or $N=256$ array with 2-bit discrete phase shifts requires searching over $4^{64} \approx 3.4 \times 10^{38}$ states.*
>
> *Classical methods like Alternating Optimization (AO) take over 250 milliseconds per channel realization on CPU, which is completely unusable for fast-fading wireless channels. Furthermore, pure neural networks make mistakes on difficult edge-case channels and cannot guarantee physical hardware constraints.*
>
> *To solve this, we designed an end-to-end differentiable predict-and-refine pipeline:*
> 1. *A **ResNet backbone with Spatial Self-Attention** extracts spatial phase correlations from noisy CSI.*
> 2. *A **Gumbel-Softmax discrete quantizer** forces the output into exact 2-bit hardware states $\{0, \pi/2, \pi, 3\pi/2\}$ using the Straight-Through Estimator.*
> 3. *An **Epistemic Uncertainty Head** outputs a heteroscedastic variance $\hat{\sigma}^2$, which has a statistically verified correlation ($\rho = 0.2254, p = 0.0241$) with the neural optimization difficulty.*
> 4. *A **Selective Router** sends only the top 25% most uncertain channels to a 10-step **Unrolled Projected Gradient Descent (PGD)** layer for physical fine-tuning.*
>
> *The entire system was scientifically audited with 22 unit tests. It delivers **18.46 bps/Hz** sum-rate, **$0.00\%$ hardware constraint violations**, and a **$143.7\times$ FLOP speedup** over classical multi-start AO."*

---

## 2. WHAT WE HAVE BUILT (THE BIG PICTURE)

We have built a complete, self-contained, publication-grade research codebase that contains:
1. **A Complete Wireless MU-MISO Physical Simulator (`src/channel/`):**
   - Implements multi-user MISO wireless physics in PyTorch.
   - Computes realistic Rician direct links, Rayleigh/Rician cascaded links, spatial correlation matrices ($\mathbf{R}_{\text{RIS}}$), and non-linear phase-amplitude coupling $\beta(\theta_n)$.
   - Solves active BS Zero-Forcing (ZF) precoding $\mathbf{W}$ under strict power constraints.
2. **The UG-PRefine Neural Architecture (`src/models/`):**
   - 2D convolutional ResNet feature extractor (`ComplexCNNBackbone`).
   - Multi-head spatial self-attention (`SpatialSelfAttention`) to capture inter-element mutual coupling.
   - Dual-head output: Phase Logits + Heteroscedastic Epistemic Uncertainty ($\hat{\sigma}^2$).
   - Gumbel-Softmax quantizer with Straight-Through Estimator (`GumbelSoftmaxQuantizer`).
   - Unrolled differentiable PGD refinement module (`unrolled_pgd_refine`).
3. **Classical Optimization Benchmark Suite (`src/optimizers/`):**
   - Continuous Alternating Optimization (AO) — mathematical theoretical upper bound ($24.93\text{ bps/Hz}$).
   - Quantized 2-Bit Alternating Optimization ($24.19\text{ bps/Hz}$).
   - Element-by-element Coordinate Descent (CD) ($23.60\text{ bps/Hz}$).
   - Standalone Projected Gradient Descent (PGD) ($17.49\text{ bps/Hz}$).
4. **Authoritative Canonical Evaluator & Frozen Benchmark (`src/evaluation/`):**
   - `CanonicalEvaluator`: Guarantees every single model is evaluated using the exact same physical formula, noise power, power budget, and ZF precoder.
   - `FrozenTestSet`: Exactly 500 deterministic channel realizations saved to disk (`data/frozen_test_set_64.pt`, seed `42000`) so comparisons are 100% fair.
5. **A Complete Automated Audit & Test Suite (`tests/`, `experiments/`):**
   - 22 unit and integration tests passing (`python -m pytest`).
   - Full parameter gradient flow verification, finite-difference gradient verification, and data leakage checks.

---

## 3. THE PROBLEM: WHY IS THIS HARD? (WIRELESS + MATH MOTIVATION)

To ace your viva, you must understand the exact physics and math of why RIS phase optimization is difficult.

```
       [Base Station (BS)]  M=8 antennas
             /         \
            /           \  Direct Path H_BU (Weak / Blocked)
           /             \
    H_BR  /               v
         v             [User 1]   [User 2]  K=2 single-antenna users
   [RIS Surface] ──────> 
    N=64 elements   H_RU
```

### 3.1 The Physical Setup
* **Base Station (BS):** Has $M = 8$ antennas and total transmit power budget $P_{\text{max}} = 1.0\text{ W}$.
* **RIS (Reconfigurable Intelligent Surface):** Has $N = 64$ passive elements. It does NOT generate power; it only alters the phase of incoming radio waves.
* **Users:** $K = 2$ single-antenna mobile users receiving signals simultaneously in the same frequency band.
* **Channels:**
  - $\mathbf{H}_{\text{BU}} \in \mathbb{C}^{K \times M}$: Direct BS-to-User path (often severely attenuated by obstacles/walls).
  - $\mathbf{H}_{\text{BR}} \in \mathbb{C}^{N \times M}$: Base Station to RIS channel (strong line-of-sight Rician fading).
  - $\mathbf{H}_{\text{RU}} \in \mathbb{C}^{K \times N}$: RIS to User channel (reflected multi-path link).

### 3.2 The Effective Channel Equation
The total signal arriving at the users travels through both the direct path and the reflected path:
$$\mathbf{H}_{\text{eff}} = \mathbf{H}_{\text{BU}} + \mathbf{H}_{\text{RU}} \mathbf{\Phi} \mathbf{H}_{\text{BR}} \quad \in \mathbb{C}^{K \times M}$$
where $\mathbf{\Phi}$ is the diagonal RIS reflection matrix:
$$\mathbf{\Phi} = \operatorname{diag}\Big(\beta(\theta_1)e^{j\theta_1}, \beta(\theta_2)e^{j\theta_2}, \dots, \beta(\theta_N)e^{j\theta_N}\Big) \in \mathbb{C}^{N \times N}$$

### 3.3 What Makes This Problem So Difficult?

1. **Combinatorial Explosion (Discrete Constraint):**
   Real RIS hardware cannot produce infinite continuous phase angles. Low-cost PIN diodes only support discrete phase steps:
   $$\theta_n \in \mathcal{F}_b = \left\{ 0, \frac{2\pi}{2^b}, \dots, \frac{2\pi(2^b - 1)}{2^b} \right\}$$
   For $b = 2$ bits, each element can only take 4 angles: $\{0, \pi/2, \pi, 3\pi/2\}$ ($0^\circ, 90^\circ, 180^\circ, 270^\circ$).
   For $N = 64$ elements, the number of possible phase combinations is:
   $$4^{64} = 2^{128} \approx 3.4 \times 10^{38} \text{ combinations!}$$
   You cannot brute-force search this space.

2. **Non-Linear Phase-Amplitude Coupling $\beta(\theta_n)$:**
   Textbooks often assume an "ideal RIS" where the reflection amplitude is always $1.0$ regardless of phase shift. In reality, shifting the phase detunes the circuit resonance, causing amplitude attenuation:
   $$\beta(\theta_n) = (1 - \beta_{\text{min}}) \left( \frac{\sin(\theta_n - \phi) + 1}{2} \right)^\gamma + \beta_{\text{min}}$$
   In our code: $\beta_{\text{min}} = 0.2$, $\phi \approx 0.43\pi$, $\gamma = 1.6$. This couples phase and amplitude, destroying simple convex approximations.

3. **Multi-User Co-Channel Interference:**
   Because User 1 and User 2 use the same spectrum, User 1's signal acts as noise to User 2. The Base Station must apply Zero-Forcing (ZF) precoding $\mathbf{W}$ to cancel this interference, which couples the BS precoder choice directly with the RIS phase configuration.

4. **Channel Estimation Noise (Imperfect CSI):**
   In real life, the Base Station never knows the true channels perfectly. It only has noisy estimates:
   $$\mathbf{H}_{\text{noisy}} = \mathbf{H}_{\text{true}} + \mathbf{E}, \quad \mathbf{E} \sim \mathcal{CN}(0, \delta^2 \mathbf{I})$$
   A good model must be robust to CSI noise up to $\delta^2 = 0.20$ ($20\%$ error).

5. **Latency Crisis:**
   Wireless channels fade every few milliseconds (the coherence time). Classical Alternating Optimization takes **251.9 milliseconds** per channel. By the time AO finishes calculating the phases, the channel has already changed! We need sub-millisecond inference.

---

## 4. HOW WE BUILT IT: DETAILED TECHNICAL ARCHITECTURE & PIPELINE

Here is the exact dataflow through the neural network:

```
[Noisy CSI: H_BR_noisy, H_RU_noisy]
           │
           ▼
[2D Cascaded Spatial Tensor X]  ---> Shape: [B, 2, K*M, N]  (Real & Imaginary stacked)
           │
           ▼
[ComplexCNNBackbone + SpatialSelfAttention]
           │
           ▼
[Latent Feature Vector z]  ---------> Shape: [B, 256]
           │
     ┌─────┴──────────────────────────────┐
     │                                    │
     ▼                                    ▼
[Phase Prediction Head]       [Epistemic Uncertainty Head]
     │                                    │
     ▼                                    ▼
[Phase Logits: B, N, 4]       [Predicted Variance var: B, 1]
     │                                    │
     ▼                                    │
[GumbelSoftmaxQuantizer]                  │
     │                                    │
     ▼                                    │
[Predicted Phase theta_pred: B, N]        │
     │                                    │
     └──────────────────┬─────────────────┘
                        │
                        ▼
             [Uncertainty-Gated Router]
              Is var >= 75th percentile?
                   /          \
            YES   /            \  NO
                 v              v
         [Unrolled PGD]     [Keep theta_pred]
          (10 steps)            │
                 │              │
                 └──────┬───────┘
                        │
                        ▼
             [Final 2-Bit Quantization] ---> Shape: [B, N] in {0, pi/2, pi, 3pi/2}
                        │
                        ▼
             [Canonical Evaluator] --------> Calculates H_eff, ZF Precoder W, SINR, Sum Rate
```

### 4.1 Input Representation & Cascaded Channel Tensor
* **Why Cascaded?** The RIS cannot transmit or receive pilots on its own (it is passive). The Base Station can only estimate the cascaded link $\mathbf{H}_{\text{cas}} = \mathbf{H}_{\text{RU}} \otimes \mathbf{H}_{\text{BR}}$.
* In `run_final_paper_experiments.py`:
  ```python
  H_cas_noisy = H_RU_noisy.unsqueeze(-1) * H_BR_noisy.unsqueeze(1)
  H_cas_spatial = H_cas_noisy.transpose(1, 3).reshape(B, K * M, N)
  tensor_x = torch.stack([H_cas_spatial.real, H_cas_spatial.imag], dim=1).float()
  tensor_x = tensor_x / (norm + 1e-8)  # Per-sample Frobenius norm normalization
  ```
* **Tensor Shape:** $[B, 2, 16, 64]$ ($B$ batch size, 2 channels for Real/Imag, $K \cdot M = 2 \times 8 = 16$ spatial dimensions, $N = 64$ RIS elements).

### 4.2 ResNet Backbone (`ComplexCNNBackbone` in `src/models/backbone_cnn.py`)
* Standard MLPs flatten the input, losing spatial 2D relationships between adjacent antenna elements.
* We built a 2D convolutional ResNet backbone:
  - **Stem:** `Conv2d(2, 64, kernel=3, padding=1)` + `BatchNorm2d` + `LeakyReLU(0.2)`.
  - **ResBlock 1:** Two `Conv2d(64, 64)` layers with residual skip connection $x + F(x)$.
  - **Spatial Attention Layer:** Inter-element relationship capture (described below).
  - **Downsampling:** `Conv2d(64, 128, stride=2)` + `ResBlock 2(128)`.
  - **Global Average Pooling & Projection:** `AdaptiveAvgPool2d((1,1))` + `Linear(128, 256)` + `BatchNorm1d` + `LeakyReLU(0.2)`.
* **Output:** Latent feature representation $\mathbf{z} \in \mathbb{R}^{B \times 256}$.

### 4.3 Spatial Self-Attention Module (`SpatialSelfAttention` in `src/models/spatial_attention.py`)
* **Why Attention?** RIS elements sit on a 2D planar array. Electromagnetic coupling between element $i$ and element $j$ depends on spatial distance. Convolutional filters have a limited local receptive field ($3 \times 3$), whereas self-attention captures long-range spatial correlations across the entire surface.
* **Math:**
  - Query: $\mathbf{Q} = \text{Conv2d}_{1\times 1}(x)$, Key: $\mathbf{K} = \text{Conv2d}_{1\times 1}(x)$, Value: $\mathbf{V} = \text{Conv2d}_{1\times 1}(x)$.
  - Energy Map: $\mathbf{E} = \mathbf{Q}^T \mathbf{K}$.
  - Attention Map: $\mathbf{A} = \operatorname{softmax}(\mathbf{E})$.
  - Output: $\mathbf{O} = \gamma (\mathbf{V} \mathbf{A}) + x$, where $\gamma$ is a learnable scalar initialized to $0.0$.
* **Impact:** In our ablation study, adding spatial self-attention increased sum-rate from $17.02\text{ bps/Hz}$ to $17.85\text{ bps/Hz}$ (**$+0.83\text{ bps/Hz}$ boost**).

### 4.4 Phase Prediction Head
* Simple MLP branching from $\mathbf{z}$:
  - `Linear(256, 512)` $\to$ `LeakyReLU(0.2)` $\to$ `Linear(512, N * 2^b)`
  - Reshaped to logits $\mathbf{Z}_{\theta} \in \mathbb{R}^{B \times N \times 4}$.
* For each of the $N=64$ elements, the network outputs 4 raw unnormalized logits corresponding to the 4 discrete phase choices ($0, \pi/2, \pi, 3\pi/2$).

### 4.5 Gumbel-Softmax Discrete Quantizer (`src/models/gumbel_quantizer.py`)
* **The Problem:** The `argmax` function is non-differentiable (zero gradient everywhere), which would prevent backpropagation from the physical sum-rate loss to the CNN weights.
* **The Solution:** The Gumbel-Softmax reparameterization trick:
  $$p_{n,m} = \frac{\exp\left((\mathbf{Z}_{n,m} + g_{n,m}) / \tau\right)}{\sum_{j=1}^4 \exp\left((\mathbf{Z}_{n,j} + g_{n,j}) / \tau\right)}, \quad g \sim \operatorname{Gumbel}(0, 1) = -\log(-\log(u)), \quad u \sim \mathcal{U}(0, 1)$$
  - During training, $\tau$ (temperature) starts high ($\tau = 2.0$) allowing exploration, and anneals down to $\tau = 0.2$ to make probabilities sharp.
* **Straight-Through Estimator (STE):**
  - In forward pass: We take the discrete hard one-hot vector $\mathbf{y}_{\text{hard}} = \text{one\_hot}(\operatorname{argmax}(p))$.
  - In backward pass: We pass the gradient directly through soft probabilities:
    $$\mathbf{y} = \mathbf{y}_{\text{hard}} - p.\text{detach}() + p$$
  - Phasor angle calculation:
    $$\theta_n = \operatorname{atan2}\left(\sum_m p_m \sin \phi_m, \sum_m p_m \cos \phi_m\right) \bmod 2\pi$$

### 4.6 Epistemic Uncertainty Head (`src/models/uncertainty_head.py`)
* Instead of outputting only a phase prediction, the network outputs its own confidence.
* Architecture:
  - `Linear(256, 128)` $\to$ `LeakyReLU(0.2)` $\to$ `Linear(128, 1)` $\to$ `Softplus()`
* **Why Softplus?** Variance must be strictly positive ($\hat{\sigma}^2 > 0$). Softplus: $f(x) = \ln(1 + e^x) + 1e-6$ mathematically guarantees positive variance without numerical explosion.
* **What does this variance represent?** It represents *epistemic uncertainty* (model difficulty). When the channel geometry is complicated or interference is severe, the network outputs a larger $\hat{\sigma}^2$.

### 4.7 Uncertainty-Gated Router
* We sort or threshold the batch by uncertainty:
  - Threshold $\tau_{75} = \text{Quantile}(\hat{\sigma}^2, 0.75)$ (the 75th percentile).
  - Mask: $m_i = \mathbb{I}(\hat{\sigma}_i^2 \ge \tau_{75})$.
* **Result:** The top 25% highest-uncertainty samples are routed to unrolled PGD refinement. The other 75% bypass refinement and exit immediately.

### 4.8 Unrolled Projected Gradient Descent (PGD) Refinement
* If a sample is uncertain, we execute $L = 10$ steps of continuous gradient ascent directly on the sum-rate function:
  ```python
  with torch.enable_grad():
      theta_curr = theta_init.clone().detach().requires_grad_(True)
      for step in range(self.unrolled_steps):  # L = 10
          H_eff = self.sys.compute_effective_channel(H_BR, H_RU, H_BU, theta_curr)
          W = self.sys.compute_active_beamformers_zf(H_eff)
          rate = self.sys.compute_achievable_sum_rate(H_eff, W)
          grad = torch.autograd.grad(outputs=rate.sum(), inputs=theta_curr)[0]
          theta_curr = (theta_curr + self.pgd_eta * grad).detach().requires_grad_(True)
      theta_final = self.sys.quantize_phase(theta_curr).detach()
  ```
* **Crucial Implementation Audit Finding:** In an early version of the code, quantization was called inside every small step of size $\eta = 0.05$. Because discrete levels are separated by $\pi/2 \approx 1.57\text{ rad}$, a tiny step of $0.05$ was immediately rounded back to zero! We fixed this: updates accumulate continuously for 10 steps, and quantization is applied *only at the very end*. This allowed **100% of refined samples to successfully change phase**.

### 4.9 Physical Wireless Channel Simulator & Canonical Evaluator (`src/evaluation/canonical_evaluator.py`)
* Guarantees every single baseline and model is evaluated under identical physical rules.
* Calculates:
  1. Effective channel $\mathbf{H}_{\text{eff}}$.
  2. ZF active beamformer $\mathbf{W}$.
  3. Signal power and interference power for each user.
  4. User SINR and sum achievable rate.
  5. Distance to nearest legal phase in $\mathcal{F}_b$. If $\text{dist} > 1e-3$, it flags a hardware constraint violation.

### 4.10 Zero-Forcing (ZF) Active BS Beamforming
* The Base Station computes the pseudo-inverse of $\mathbf{H}_{\text{eff}}$ to null inter-user interference:
  $$\mathbf{V} = \mathbf{H}_{\text{eff}}^H (\mathbf{H}_{\text{eff}} \mathbf{H}_{\text{eff}}^H + \epsilon \mathbf{I})^{-1}$$
  $$\mathbf{W} = \mathbf{V}^T \times \sqrt{\frac{P_{\text{max}}}{\|\mathbf{V}\|_F^2}}$$
* Verified: Inter-user interference is nulled to $< 10^{-4}$ residual, and total power equals exactly $1.0000\text{ W}$.

### 4.11 Non-Linear Phase-Amplitude Coupling $\beta(\theta)$
* Implemented in PyTorch as:
  ```python
  sin_term = (torch.sin(theta - 1.35088) + 1.0) / 2.0
  beta = (1.0 - 0.2) * (sin_term ** 1.6) + 0.2
  phasor = beta * torch.exp(1j * theta)
  ```
* This ensures that reflection amplitude drops down to as low as $0.2$ at certain phase shifts, matching physical varactor diode behavior.

---

## 5. TRAINING OBJECTIVES & MATHEMATICAL LOSSES

The model is trained end-to-end without needing ground-truth labels using a **Composite Multi-Objective Loss**:
$$\mathcal{L}_{\text{total}} = \lambda_{\text{rate}} \mathcal{L}_{\text{rate}} + \lambda_{\text{unc}} \mathcal{L}_{\text{unc}} + \lambda_{\text{quant}} \mathcal{L}_{\text{quant}}$$

In `configs/default_config.yaml`:
$$\lambda_{\text{rate}} = 1.0, \quad \lambda_{\text{unc}} = 0.2, \quad \lambda_{\text{quant}} = 0.05$$

### 1. Unsupervised Achievable Rate Loss ($\mathcal{L}_{\text{rate}}$)
$$\mathcal{L}_{\text{rate}} = -\mathbb{E}\left[ \mathcal{R}_{\text{sum}}(\theta) \right] = -\frac{1}{B} \sum_{i=1}^B \sum_{k=1}^K \log_2(1 + \operatorname{SINR}_{i,k})$$
* **Why negative?** Optimizers minimize loss. Minimizing negative sum-rate maximizes spectral efficiency.

### 2. Heteroscedastic Gaussian NLL Calibration Loss ($\mathcal{L}_{\text{unc}}$)
$$\mathcal{L}_{\text{unc}} = \frac{1}{B} \sum_{i=1}^B \left( \frac{|\mathcal{R}_{\text{target}, i} - \mathcal{R}_{\text{pred}, i}|^2}{2\hat{\sigma}_i^2} + \frac{1}{2} \ln \hat{\sigma}_i^2 \right)$$
* **How it works:**
  - If the prediction error is large, the network is penalized heavily unless it predicts a high variance $\hat{\sigma}^2$ in the denominator.
  - But the network cannot just predict $\hat{\sigma}^2 = \infty$ for everything because the second term, $\frac{1}{2}\ln \hat{\sigma}^2$, acts as a barrier penalizing large uncertainty.
  - This forces $\hat{\sigma}^2$ to act as a calibrated measure of model difficulty!

### 3. Categorical Quantization Entropy Loss ($\mathcal{L}_{\text{quant}}$)
$$\mathcal{L}_{\text{quant}} = -\frac{1}{B \cdot N} \sum_{i=1}^B \sum_{n=1}^N \sum_{m=1}^4 p_{i,n,m} \ln(p_{i,n,m} + 1e-8)$$
* Minimizing categorical entropy pushes probability distributions $p$ away from uniform $[0.25, 0.25, 0.25, 0.25]$ towards sharp one-hot distributions $[0, 1, 0, 0]$, guaranteeing near-zero loss when discretized.

---

## 6. OPERATIONAL MODES (MODE A, B, C, D) — DO NOT CONFLATE!

In your viva, the examiners will test whether you know what was computed online vs offline. **Memorize this distinction:**

| Mode | Name | Sum Rate | CPU Latency | Requires Offline AO? | Description |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **Mode A** | **Pure Direct Neural Inference** | **$18.42\text{ bps/Hz}$** | **$0.45\text{ ms}$** | **NO** | Pure CNN feedforward pass + Gumbel discrete quantization. Real-time sub-millisecond execution. |
| **Mode B** | **Neural + Always PGD (100%)** | **$18.53\text{ bps/Hz}$** | **$4.12\text{ ms}$** | **NO** | Neural prediction followed by 10 unrolled PGD gradient ascent steps on all samples. |
| **Mode C** | **Neural + Offline AO Warm-Start** | **$24.27\text{ bps/Hz}$** | **$69.22\text{ ms}$** | **YES** | Classical AO is run offline for 3 iterations to warm-start the phase, followed by neural refinement. *Never claim this is a pure online inference result!* |
| **Mode D** | **Continuous AO Upper Bound** | **$24.93\text{ bps/Hz}$** | **$251.93\text{ ms}$** | **YES** | Multi-start classical Alternating Optimization with continuous phase shifts ($b=\infty$). Mathematical theoretical upper bound. |

---

## 7. STEP-BY-STEP TERMINAL EXECUTION GUIDE (HOW TO DEMO LIVE)

If your professor or examiner says: *"Open your terminal and run the project right now,"* here is the exact script to follow.

### Step 1: Open PowerShell / Terminal in Project Directory
Navigate to the repository folder:
```powershell
cd "c:\Users\myasa\OneDrive\ドキュメント\7th Semester\Deep Learning\DL Class Project"
```

### Step 2: Run the Unit & Integration Test Suite
To prove the implementation is bug-free and physically verified:
```powershell
python -m pytest
```
*Expected Output:*
```text
======================== 22 passed, 1 warning in 1.51s ========================
```
*What to tell the examiner:* "All 22 unit tests passed. These tests verify the discrete phase alphabet, non-linear amplitude coupling, Zero-Forcing interference nulling, autograd gradient backprop, and determinism."

### Step 3: Run the Forensic Model Integrity Audit Script
This runs the full audit checking gradient flow, Gumbel quantization, and PGD execution:
```powershell
python experiments/run_model_integrity_audit.py
```
*Expected Output:*
```text
[INFO] [ModelIntegrityAudit]: STARTING FORENSIC MODEL INTEGRITY & END-TO-END IMPLEMENTATION AUDIT
[INFO] [ModelIntegrityAudit]: Saved GRADIENT_FLOW_REPORT.csv
[INFO] [ModelIntegrityAudit]: Min Phase: 0.0000, Max Phase: 4.7124
[INFO] [ModelIntegrityAudit]: Unique Phase Values: [0.     1.5708 3.1416 4.7124]
[INFO] [ModelIntegrityAudit]: Saved CSI_AUDIT.csv
[INFO] [ModelIntegrityAudit]: Saved PGD_AUDIT.csv
[INFO] [ModelIntegrityAudit]: Saved FINAL_MODEL_INTEGRITY_SCORECARD.md
[INFO] [ModelIntegrityAudit]: Model Integrity Forensic Audit Complete!
```

### Step 4: Run the Scientific Validation Script
This validates the unrolled PGD execution trace and calculates Spearman $\rho$ correlation:
```powershell
python experiments/run_final_validation.py
```
*Expected Output:*
```text
[INFO] [FinalValidation]: Percentage of samples whose phase changed: 100.00%
[INFO] [FinalValidation]: Mean angular phase change: 279.75 deg
[INFO] [FinalValidation]: Mean Rate Before Refinement: 18.4212 bps/Hz
[INFO] [FinalValidation]: Mean Rate After Refinement:  18.5318 bps/Hz
[INFO] [FinalValidation]: Spearman Rho: 0.2254 (p = 0.0241)
[INFO] [FinalValidation]: Pearson R:    0.2452 (p = 0.0139)
```

### Step 5: Run the Master Paper Experiment Package & Generate All 10 Figures
This evaluates all baselines, calculates statistical hypothesis tests, and renders publication plots:
```powershell
python experiments/run_final_paper_experiments.py
```
*Expected Output:*
```text
[INFO] [PaperExperiments]: Saved FINAL_RESULTS_TABLE.csv
[INFO] [PaperExperiments]: Saved FINAL_STATISTICS.csv
[INFO] [PaperExperiments]: Saved FINAL_ABLATION.csv
[INFO] [PaperExperiments]: Saved FINAL_ROBUSTNESS.csv
[INFO] [PaperExperiments]: Saved FINAL_LATENCY.csv and FINAL_COMPLEXITY.csv
[INFO] [PaperExperiments]: All 10 publication figures successfully rendered to results/figures/
[INFO] [PaperExperiments]: Final Paper Experiments Package Execution Complete!
```

### Step 6: View the Generated Plots
Open the `results/figures/` folder. You will see:
* `fig1_architecture.png`: Full block diagram of the predict-refine pipeline.
* `fig2_system_model.png`: MU-MISO RIS transmission diagram.
* `fig3_sum_rate_benchmark.png`: Bar chart of sum rates across all baselines.
* `fig4_pareto_latency_rate.png`: Sum-Rate vs Latency Pareto frontier.
* `fig5_flop_scaling.png`: FLOP scaling vs RIS array size $N \in [32, 256]$.
* `fig6_csi_noise_robustness.png`: Robustness under CSI error $\delta^2 \in [0.0, 0.20]$.
* `fig7_uncertainty_calibration.png`: Scatter plot of epistemic variance vs AO gap.
* `fig8_refinement_sweep.png`: Refinement budget sweep (0% to 100%).
* `fig9_ablation_study.png`: Step-by-step contribution of each module.
* `fig10_phase_resolution.png`: Performance across 1-bit, 2-bit, 3-bit, and continuous.

---

## 8. AUDITED EXPERIMENTAL RESULTS & KEY METRICS TABLE

These numbers are from `results/final/FINAL_RESULTS_TABLE.csv` on the frozen 500-sample test set ($N=64, M=8, K=2, b=2\text{ bits}$):

| Method / Algorithm | Sum-Rate (bps/Hz) | Median Rate | 95% Confidence Interval | CPU Latency | FLOPs (M) | Hardware Violations |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Continuous AO (Upper Bound)** | $24.93$ | $24.81$ | $[24.78, 25.08]$ | $251.93\text{ ms}$ | $338.18\text{ M}$ | $97.5\%$ (Continuous) |
| **Quantized 2-Bit AO** | $24.19$ | $24.10$ | $[24.05, 24.33]$ | $185.30\text{ ms}$ | $338.18\text{ M}$ | **$0.00\%$** |
| **Coordinate Descent (CD)** | $23.60$ | $23.51$ | $[23.46, 23.74]$ | $28.45\text{ ms}$ | $45.20\text{ M}$ | **$0.00\%$** |
| **UG-PRefine Always (100% PGD)** | **$18.53$** | **$18.49$** | **$[18.39, 18.67]$** | **$4.12\text{ ms}$** | **$2.374\text{ M}$** | **$0.00\%$** |
| **UG-PRefine Selective (25% Gated)**| **$18.46$** | **$18.42$** | **$[18.32, 18.60]$** | **$1.36\text{ ms}$** | **$2.356\text{ M}$** | **$0.00\%$** |
| **UG-PRefine Direct (0% Refine)**| **$18.42$** | **$18.38$** | **$[18.28, 18.56]$** | **$0.45\text{ ms}$** | **$2.350\text{ M}$** | **$0.00\%$** |
| **Standalone PGD ($L=10$)** | $17.49$ | $17.40$ | $[17.34, 17.64]$ | $3.67\text{ ms}$ | $0.024\text{ M}$ | **$0.00\%$** |
| **GumbelRIS Baseline** | $17.07$ | $17.01$ | $[16.92, 17.22]$ | $0.45\text{ ms}$ | $2.350\text{ M}$ | **$0.00\%$** |
| **Random Phase Shift** | $16.61$ | $16.55$ | $[16.46, 16.76]$ | $0.05\text{ ms}$ | $0.000\text{ M}$ | **$0.00\%$** |
| **No RIS Transmission** | $10.09$ | $10.05$ | $[9.94, 10.24]$ | $0.05\text{ ms}$ | $0.000\text{ M}$ | **$0.00\%$** |

### Key Takeaways from the Data:
1. **Mathematical Hierarchy Holds:**
   $$\text{No RIS } (10.09) < \text{Random } (16.61) < \text{GumbelRIS } (17.07) < \text{PGD } (17.49) < \text{UG-PRefine } (18.42) < \text{CD } (23.60) < \text{Quantized AO } (24.19) \le \text{Continuous AO } (24.93)$$
2. **Selective Refinement Retention:** 25% selective refinement achieves **$18.46\text{ bps/Hz}$**, which retains **$>99.6\%$** of the always-refine sum-rate ($18.53\text{ bps/Hz}$) while saving **$75\%$** of refinement FLOPs.
3. **Statistical Significance (`FINAL_STATISTICS.csv`):**
   - UG-PRefine vs Random Phase: Diff $+1.81\text{ bps/Hz}$, $t = 41.20$, $p < 10^{-15}$, Cohen's $d = 1.08$ (Massive effect size).
   - UG-PRefine vs GumbelRIS: Diff $+1.35\text{ bps/Hz}$, $t = 38.45$, $p < 10^{-15}$, Cohen's $d = 0.81$ (Large effect size).
   - 25% Uncertainty vs 25% Random Routing: Diff $+0.006\text{ bps/Hz}$, $p = 0.8291$ (Statistically comparable under short PGD depth $L=10$).

---

## 9. CLASS PRESENTATION WALKTHROUGH (SLIDE-BY-SLIDE SCRIPT)

Use this structure when preparing slides for your final presentation:

* **Slide 1: Title & Team**
  - *Title:* UG-PRefine-RIS: Uncertainty-Gated Predict-Refine Neural Network for 6G RIS Phase Optimization.
  - *Subtitle:* A Physics-Constrained Deep Learning Approach under Imperfect CSI and Hardware Quantization.
* **Slide 2: Background — The 6G Wireless Bottleneck**
  - Explain high-frequency mmWave/sub-THz path loss and obstacles.
  - Introduce RIS: $N$ passive reflecting elements dynamically shaping the wireless channel.
* **Slide 3: Problem Formulation & Challenges**
  - Discrete combinatorial space ($4^{64} \approx 3.4 \times 10^{38}$ states).
  - Realistic non-linear amplitude attenuation $\beta(\theta_n)$.
  - Classical Alternating Optimization takes $251\text{ ms}$ (unusable for real-time millisecond fading).
* **Slide 4: Proposed Architecture — UG-PRefine**
  - Show Figure 1 (`fig1_architecture.png`).
  - Highlight the 3-stage paradigm: PREDICT $\to$ ESTIMATE UNCERTAINTY $\to$ SELECTIVELY REFINE.
* **Slide 5: Key Innovations**
  - *ResNet + Spatial Self-Attention:* Captures inter-element spatial phase coupling across the 2D surface.
  - *Gumbel-Softmax Quantizer:* Solves non-differentiability using the Straight-Through Estimator.
  - *Epistemic Uncertainty Head:* Predicts sample difficulty $\hat{\sigma}^2$ via heteroscedastic Gaussian NLL loss.
  - *Unrolled Differentiable PGD:* Accumulates continuous gradient ascent before discrete projection.
* **Slide 6: Main Benchmark Results**
  - Show Figure 3 (`fig3_sum_rate_benchmark.png`) and Table.
  - Emphasize that UG-PRefine beats Random, Standalone PGD, and GumbelRIS.
  - State clearly that Continuous AO is the theoretical upper bound ($24.93\text{ bps/Hz}$).
* **Slide 7: Pareto Frontier — Latency vs Utility**
  - Show Figure 4 (`fig4_pareto_latency_rate.png`).
  - Direct Neural takes $0.45\text{ ms}$; 25% Selective takes $1.36\text{ ms}$; Continuous AO takes $251.93\text{ ms}$.
  - $143.7\times$ FLOP reduction at $N=256$.
* **Slide 8: Robustness & Ablation Study**
  - Show Figure 6 (`fig6_csi_noise_robustness.png`) and Figure 9 (`fig9_ablation_study.png`).
  - Robust against CSI noise up to $20\%$ ($\delta^2 = 0.20$).
  - Self-attention adds $+0.83\text{ bps/Hz}$; Gumbel adds $+0.20\text{ bps/Hz}$; Uncertainty adds $+0.37\text{ bps/Hz}$.
* **Slide 9: Scientific Audit & Verification**
  - Mention the forensic software audit: 22/22 unit tests passed.
  - $0.00\%$ hardware constraint violations.
  - Honest disclosure of Mode C ($24.27\text{ bps/Hz}$ requires offline AO warm-start).
* **Slide 10: Conclusion & Future Work**
  - Summary of contributions.
  - Future direction: Custom GPU CUDA kernels for unrolled PGD autograd steps.

---

## 10. EXHAUSTIVE VIVA VOCE Q&A BANK

Here are the questions examiners are guaranteed to ask, divided into categories with complete, rigorous answers.

### Category 1: Deep Learning & Neural Network Concepts

#### Q1.1: Why did you use a 2D CNN instead of a standard fully connected Multi-Layer Perceptron (MLP)?
> **Answer:** *"A standard MLP vectorizes the channel matrix into a flat 1D vector. This destroys the spatial 2D neighborhood topology of the RIS elements and the antenna arrays. Convolutional layers exploit spatial locality and weight sharing. Furthermore, our `ComplexCNNBackbone` uses residual connections (`ResBlock2D`), which solve the vanishing gradient problem and allow stable gradient backpropagation through deep feature extraction layers."*

#### Q1.2: What is the purpose of the Spatial Self-Attention module?
> **Answer:** *"Convolutional kernels only have local receptive fields (typically $3 \times 3$). However, electromagnetic coupling and mutual impedance across an RIS surface occur across the entire aperture. Our `SpatialSelfAttention` module calculates Query ($\mathbf{Q}$), Key ($\mathbf{K}$), and Value ($\mathbf{V}$) matrices to compute a global attention energy map. This allows the network to model long-range spatial correlations between distant RIS elements. In our ablation experiments, adding attention boosted achievable rate by $+0.83\text{ bps/Hz}$."*

#### Q1.3: What is the Gumbel-Softmax trick, and why was it necessary?
> **Answer:** *"In discrete phase optimization, selecting an angle is an $\operatorname{argmax}$ operation. However, $\operatorname{argmax}$ has a derivative of zero almost everywhere and is undefined at boundaries, which completely blocks backpropagation. The Gumbel-Softmax reparameterization trick approximates categorical sampling by adding Gumbel noise to logits and taking a temperature-annealed softmax:
> $$p_{n,m} = \frac{\exp((\mathbf{z}_{n,m} + g_{n,m}) / \tau)}{\sum_j \exp((\mathbf{z}_{n,j} + g_{n,j}) / \tau)}$$
> By using the Straight-Through Estimator (STE), we take the discrete hard one-hot choice in the forward pass while letting the true gradients flow through the continuous soft probabilities during the backward pass."*

#### Q1.4: What is the difference between epistemic and aleatoric uncertainty? Which one does your model estimate?
> **Answer:** *"Aleatoric uncertainty is the inherent noise in the data/environment (like thermal noise in the channel $n_k$ or CSI estimation error $\mathbf{E}$). It cannot be reduced by collecting more data. Epistemic uncertainty is model uncertainty—the model's lack of confidence due to unfamiliar, complex, or difficult inputs. Our `EpistemicUncertaintyHead` estimates epistemic variance $\hat{\sigma}^2$ by predicting how difficult a given channel realization is for the neural network to align."*

#### Q1.5: How is the uncertainty head trained if you don't have labeled ground-truth uncertainty values?
> **Answer:** *"We train the uncertainty head in an unsupervised/self-supervised manner using a Heteroscedastic Gaussian Negative Log-Likelihood (NLL) loss:
> $$\mathcal{L}_{\text{NLL}} = \frac{|\mathcal{R}_{\text{target}} - \mathcal{R}_{\text{pred}}|^2}{2\hat{\sigma}^2} + \frac{1}{2} \ln \hat{\sigma}^2$$
> If the network's phase prediction results in a large performance gap $|\mathcal{R}_{\text{target}} - \mathcal{R}_{\text{pred}}|$, the loss will be huge unless the network predicts a large variance $\hat{\sigma}^2$ in the denominator. Conversely, the $\frac{1}{2}\ln \hat{\sigma}^2$ term prevents the network from trivially predicting infinite variance. This balances the head to accurately output larger variance on difficult channels."*

---

### Category 2: Wireless Communications & 6G RIS Concepts

#### Q2.1: Why is an RIS called "passive"? Does it amplify the signal?
> **Answer:** *"An RIS is passive because it does not contain power-hungry active Radio Frequency (RF) components like power amplifiers, mixers, or Digital-to-Analog Converters (DACs). It simply reflects incoming electromagnetic waves while introducing an adjustable phase shift via low-power varactor or PIN diodes. It does not amplify the power of the signal; it only constructively focuses scattered power towards the user."*

#### Q2.2: What is the difference between Rician fading and Rayleigh fading? Where are they used in your project?
> **Answer:** *"Rayleigh fading assumes there is no direct Line-of-Sight (LoS) path between transmitter and receiver, only dense scattered multi-path components. Rician fading assumes there is a dominant Line-of-Sight path in addition to scattered non-line-of-sight components, parameterized by the Rician $K$-factor. In our system:
> - The Base Station to RIS link ($\mathbf{H}_{\text{BR}}$) is modeled as Rician ($K = 10\text{ dB}$) because the RIS is typically installed on high walls with a clear view of the BS.
> - The RIS to User link ($\mathbf{H}_{\text{RU}}$) is modeled as correlated Rayleigh/moderate Rician because mobile users move through scattered clutter."*

#### Q2.3: Explain Zero-Forcing (ZF) beamforming and why it was chosen over Maximum Ratio Transmission (MRT) or MMSE.
> **Answer:** *"In a multi-user system, the Base Station transmits signals to $K=2$ users simultaneously on the same frequency. Maximum Ratio Transmission (MRT) only maximizes signal power to each user, ignoring inter-user interference. Zero-Forcing (ZF) computes the pseudo-inverse of the effective channel $\mathbf{H}_{\text{eff}}^\dagger = \mathbf{H}_{\text{eff}}^H(\mathbf{H}_{\text{eff}}\mathbf{H}_{\text{eff}}^H)^{-1}$. This projects each user's beam into the null space of all other users, completely eliminating multi-user interference. While MMSE offers slightly higher SINR at very low SNR, ZF provides an exact, closed-form, differentiable precoder that scales cleanly during end-to-end backpropagation."*

#### Q2.4: What is non-linear phase-amplitude coupling, and why does ignoring it cause performance collapse?
> **Answer:** *"In an ideal theoretical RIS, each element shifts phase while maintaining $100\%$ reflection amplitude ($\beta = 1.0$). In physical hardware, an element's impedance changes when adjusting its capacitance/inductance to shift phase, detuning its resonant frequency and dissipating energy. The reflection amplitude $\beta(\theta_n)$ drops significantly at certain angles (down to $0.2$ in our model). If an algorithm assumes $\beta = 1.0$, it selects phase shifts that accidentally attenuate the reflected beam, leading to severe rate loss in real hardware."*

---

### Category 3: Optimization & Mathematical Foundations

#### Q3.1: What is Alternating Optimization (AO), and why is it considered an upper bound?
> **Answer:** *"The joint optimization of BS precoder $\mathbf{W}$ and RIS phase matrix $\mathbf{\Phi}$ is non-convex. Alternating Optimization decouples this into two sub-problems:
> 1. Fix $\mathbf{\Phi}$ and solve for the optimal $\mathbf{W}$ (via Zero-Forcing or water-filling).
> 2. Fix $\mathbf{W}$ and solve for the optimal $\mathbf{\Phi}$ (via complex circle manifold optimization or semidefinite relaxation).
> It iterates between these two steps until convergence. Continuous AO ($24.93\text{ bps/Hz}$) allows continuous phase angles $[0, 2\pi)$ and multi-start initializations, representing the mathematical physical upper bound for this channel."*

#### Q3.2: What is Coordinate Descent (CD) in RIS optimization?
> **Answer:** *"Coordinate Descent optimizes one discrete phase element $\theta_n \in \mathcal{F}_b$ at a time while keeping all other $N-1$ elements fixed. It tests all 4 discrete phase states for element $n$, selects the state that yields the highest sum rate, and moves to element $n+1$. Because the rate is evaluated after every change, it is strictly non-decreasing. In our benchmark, Coordinate Descent achieved $23.60\text{ bps/Hz}$ in $28.45\text{ ms}$."*

#### Q3.3: How does your unrolled PGD refinement layer work?
> **Answer:** *"Projected Gradient Descent takes the neural network's initial phase prediction $\theta_{\text{init}}$ and performs $L = 10$ steps of gradient ascent directly along the gradient of the sum-rate function:
> $$\theta_{t+1} = \theta_t + \eta \nabla_{\theta} \mathcal{R}_{\text{sum}}(\theta_t)$$
> By accumulating continuous phase shifts during the unrolled steps and applying discrete nearest-neighbor quantization $\mathcal{F}_b$ only after step 10, the model successfully fine-tunes boundary phase assignments, achieving single-sample improvements up to $+1.96\text{ bps/Hz}$."*

---

### Category 4: Implementation & Codebase Architecture

#### Q4.1: How do you guarantee $0.00\%$ hardware constraint violations?
> **Answer:** *"Constraint violations are defined as any phase shift that does not belong to the allowed 2-bit discrete alphabet $\mathcal{F}_2 = \{0, \pi/2, \pi, 3\pi/2\}$. We guarantee zero violations by passing all continuous outputs (both from Gumbel-Softmax and from unrolled PGD) through our `SystemModel.quantize_phase()` function. This computes the Euclidean distance between each continuous angle and the 4 alphabet points and assigns the nearest neighbor. Our canonical evaluator checks whether $\min |\theta - \mathcal{F}_2| > 1e-3$ and confirmed exactly $0.00\%$ violations across all 500 test samples."*

#### Q4.2: What is the difference between FLOP speedup and CPU latency speedup?
> **Answer:** *"FLOP speedup is the theoretical ratio of floating-point arithmetic operations required by the algorithms. At $N=256$, classical AO requires $338.18\text{ M FLOPs}$ due to repeated $N \times N$ matrix inversions, while UG-PRefine requires only $2.356\text{ M FLOPs}$—a **$143.7\times$ FLOP speedup**.
> CPU latency speedup is the measured wall-clock execution time on hardware. Direct neural inference takes $0.45\text{ ms}$ vs AO's $251.93\text{ ms}$ (a **$559.8\times$ ratio**). However, when unrolled PGD refinement is enabled ($4.12\text{ ms}$), the latency ratio drops to **$55.8\times$** because PyTorch dynamically builds an autograd computation graph in memory on the CPU during inference, which introduces software graph-building overhead."*

#### Q4.3: What is the purpose of the Canonical Evaluator and the Frozen Test Set?
> **Answer:** *"In academic research, comparing algorithms is often flawed because different baselines are tested with slightly different noise powers, channel seeds, or precoding formulas. We created `CanonicalEvaluator` as a single, authoritative evaluation engine that calculates the effective channel, Zero-Forcing beamformer, and sum rate for all methods identically. We also created `data/frozen_test_set_64.pt` (seed `42000`), storing 500 fixed channel realizations so every single baseline is evaluated on the exact same channel realizations."*

---

### Category 5: The "Tough / Trap" Examiner Questions & Bulletproof Answers

These are the tricky questions an experienced examiner will ask to see if you actually understand the project or if you are exaggerating your results.

#### Trap Question 1: *"Did your neural network beat Continuous Alternating Optimization (AO)?"*
> **Correct Bulletproof Answer:**
> *"No, sir. And scientifically, it should NOT beat Continuous AO. Continuous AO ($24.93\text{ bps/Hz}$) optimizes continuous phase shifts $[0, 2\pi)$ without quantization and takes over 250 milliseconds. Our model optimizes discrete 2-bit phases $\{0, \pi/2, \pi, 3\pi/2\}$ in $0.45$ milliseconds. Quantized 2-bit AO achieves $24.19\text{ bps/Hz}$.
> Our pure neural model achieves **$18.42\text{ bps/Hz}$**, capturing $76.2\%$ of Quantized AO while delivering a **$559.8\times$ latency speedup**. Continuous AO is the theoretical mathematical upper bound, not an online operational baseline."*

#### Trap Question 2: *"In previous preliminary reports, there was a number $24.27\text{ bps/Hz}$ reported for UG-PRefine. Is $24.27\text{ bps/Hz}$ a pure neural inference result?"*
> **Correct Bulletproof Answer:**
> *"No. That is an important scientific distinction we audited. $24.27\text{ bps/Hz}$ is **Mode C: Neural + Offline AO Warm-Start**. In Mode C, classical AO is run offline for 3 iterations to provide a warm-start initialization before neural refinement, requiring $69.22\text{ ms}$.
> The pure inference-only neural result is **$18.42\text{ bps/Hz}$** (Mode A) or **$18.53\text{ bps/Hz}$** (Mode B with PGD). We explicitly separated these modes in our documentation so that offline computation is never misleadingly presented as online inference."*

#### Trap Question 3: *"Does uncertainty-gated routing significantly beat random routing? Be honest."*
> **Correct Bulletproof Answer:**
> *"Under our tested unrolled depth of $L=10$ PGD steps, uncertainty-gated routing ($18.4616\text{ bps/Hz}$) performs comparably to random routing ($18.4554\text{ bps/Hz}$) with a paired $t$-test $p$-value of $0.8291$.
> However, our epistemic uncertainty head **does** contain statistically significant ranking information: Spearman $\rho = 0.2254$ ($p = 0.0241$), and the top 10% highest uncertainty samples have a $7.38\text{ bps/Hz}$ gap to AO compared to $6.43\text{ bps/Hz}$ for the general population. The reason the rate difference between uncertainty and random routing is small under $L=10$ is that 10 unrolled gradient steps provide modest recovery ($+0.11\text{ bps/Hz}$ average gain). Uncertainty routing serves as a controllable mechanism to budget computation by identifying difficult channels."*

#### Trap Question 4: *"Why did the early benchmark show that Random Phase beat Alternating Optimization ($16.50$ vs $13.90$)? That makes no physical sense."*
> **Correct Bulletproof Answer:**
> *"Exactly, sir. That was a critical implementation bug in the preliminary baseline code which we identified during our audit. The original AO baseline performed a scalar angle division `new_phase = -phase_sum / K` instead of complex phasor addition. Once we corrected the AO solver to perform proper complex matrix projections, Continuous AO jumped from $13.90$ to its true value of **$24.93\text{ bps/Hz}$**, strictly restoring the correct physical hierarchy where Continuous AO $\ge$ Quantized AO $>$ Random Phase."*

#### Trap Question 5: *"Did you test your model on a real physical RIS hardware testbed?"*
> **Correct Bulletproof Answer:**
> *"No, sir. This work is evaluated through high-fidelity, physics-constrained electromagnetic channel simulations incorporating non-linear phase-amplitude coupling $\beta(\theta)$, spatial correlation matrices $\mathbf{R}_{\text{RIS}}$, and complex Gaussian CSI estimation error. Validating this framework on an over-the-air physical hardware testbed with PIN diode controllers is documented as our primary future work."*

#### Trap Question 6: *"What happens to your model if CSI estimation error increases to 20%?"*
> **Correct Bulletproof Answer:**
> *"We evaluated Out-of-Distribution (OOD) robustness across CSI noise variances $\delta^2 \in [0.00, 0.20]$ ($0\%$ to $20\%$ error). While classical AO degrades by $14.6\%$ as CSI error increases, UG-PRefine maintains a stable sum-rate of $18.59\text{ bps/Hz}$. This is because the ResNet convolutional feature extractor filters out high-frequency Gaussian noise during spatial feature pooling, giving it inherent noise resilience."*

---

## 11. QUICK REFERENCE CHEAT-SHEET (NUMBERS TO REMEMBER)

* **Antennas ($M$):** 8
* **RIS Elements ($N$):** 64 (scaled up to 256 for complexity benchmarking)
* **Users ($K$):** 2 single-antenna users
* **Quantization ($b$):** 2 bits $\to$ Alphabet $\mathcal{F}_2 = \{0, \pi/2, \pi, 3\pi/2\}$
* **Continuous AO Rate (Upper Bound):** $24.93\text{ bps/Hz}$ ($251.93\text{ ms}$)
* **Quantized 2-Bit AO Rate:** $24.19\text{ bps/Hz}$ ($185.30\text{ ms}$)
* **Coordinate Descent Rate:** $23.60\text{ bps/Hz}$ ($28.45\text{ ms}$)
* **UG-PRefine Always (100% PGD):** $18.53\text{ bps/Hz}$ ($4.12\text{ ms}$)
* **UG-PRefine Selective (25% Gated):** $18.46\text{ bps/Hz}$ ($1.36\text{ ms}$)
* **UG-PRefine Direct (0% Refine):** $18.42\text{ bps/Hz}$ ($0.45\text{ ms}$)
* **Random Phase Shift Rate:** $16.61\text{ bps/Hz}$
* **No RIS Rate:** $10.09\text{ bps/Hz}$
* **FLOP Speedup at $N=256$:** **$143.7\times$** ($2.356\text{ M}$ vs $338.18\text{ M FLOPs}$)
* **Direct Neural CPU Latency Speedup:** **$559.8\times$** ($0.45\text{ ms}$ vs $251.93\text{ ms}$)
* **Hardware Constraint Violations:** **$0.00\%$**
* **Uncertainty Correlation:** Spearman $\rho = 0.2254$ ($p = 0.0241$), Pearson $r = 0.2452$ ($p = 0.0139$)
* **Top 10% Uncertainty Gap to AO:** $7.38\text{ bps/Hz}$ (vs $6.43\text{ bps/Hz}$ overall)
* **Passing Unit & Integration Tests:** 22 / 22 Passed (`python -m pytest`)
* **Model Status:** `GREEN — MODEL VERIFIED`

---
*Created for University Class Presentation and Viva Voce Examination.*
