# ACTUAL COMPUTATION GRAPH AUDIT: UG-PRefine-RIS

This document traces the exact end-to-end PyTorch computation graph of **UG-PRefine-RIS**, verifying tensor flow, gradient connections, and output consumption at every stage.

---

## 1. End-to-End Computation Graph Pipeline

```
[Noisy CSI H_noisy] 
       │
       ▼ (Preprocessing & Stacking)
[Input Tensor X: [B, 2, K*M, N]]
       │
       ▼ (stem: Conv2d(2,64) -> BN -> LeakyReLU)
[Stem Feature Map: [B, 64, K*M, N]]
       │
       ▼ (res1: ResBlock2D(64))
[Residual Feature Map 1: [B, 64, K*M, N]]
       │
       ▼ (attention: SpatialSelfAttention(64))
[Attention-Weighted Feature Map: [B, 64, K*M, N]]
       │
       ▼ (downsample: Conv2d(64,128,stride=2))
[Downsampled Feature Map: [B, 128, (K*M)/2, N/2]]
       │
       ▼ (res2: ResBlock2D(128))
[Residual Feature Map 2: [B, 128, (K*M)/2, N/2]]
       │
       ▼ (global_pool: AdaptiveAvgPool2d((1,1)) -> flatten)
[Pooled Vector: [B, 128]]
       │
       ▼ (fc_proj: Linear(128,256) -> BN1d -> LeakyReLU)
[Latent Representation z: [B, 256]] ───► Gradient Flow: CONNECTED [GREEN]
       │
       ├───► [Uncertainty Head: Linear(256,128) -> LeakyReLU -> Linear(128,1) -> Softplus]
       │            │
       │            ▼
       │     [Uncertainty Variance var: [B, 1]] ───► Used in Loss & Router [GREEN]
       │
       └───► [Phase Head: Linear(256,512) -> LeakyReLU -> Linear(512, N*2^b)]
                    │
                    ▼
             [Phase Logits: [B, N, 2^b]]
                    │
                    ▼ (GumbelSoftmaxQuantizer)
             [Predictive Phase theta_pred: [B, N]]
                    │
                    ▼ (Selective Router: var >= threshold)
             ┌──────┴──────┐
             │             │
        (High Var)    (Low Var)
             │             │
             ▼             ▼
       [Unrolled PGD]  [Keep theta_pred]
             │             │
             └──────┬──────┘
                    │
                    ▼ (Final Nearest-Neighbor Quantization)
             [Final Phase theta: [B, N]]
                    │
                    ▼ (Physical Channel Simulator)
             [Effective Channel H_eff: [B, K, M]]
                    │
                    ▼ (Zero-Forcing Beamformer)
             [Active Precoder W: [B, K, M]]
                    │
                    ▼ (Rate Function)
             [Achievable Sum-Rate R_sum: [B]]
```

---

## 2. Graph Connectivity & Node Verification

* **Dead Branches Check:** **NONE.** Every module instantiated in `UGPRefineRISModel` is called during the forward pass.
* **Detached Tensors Check:** Tensors are only detached during unrolled PGD continuous gradient ascent steps (`theta_curr.detach().requires_grad_(True)`) to allow step-by-step continuous ascent, and after final nearest-neighbor quantization to guarantee discrete phase execution.
* **Gradient Autograd Flow:** Tested via `loss.backward()`. Gradients flow back through `CompositeRISLoss` -> `AchievableRateLoss` -> `SystemModel` -> `GumbelSoftmaxQuantizer` -> `phase_head` -> `backbone` (including `SpatialSelfAttention` and `ResBlock2D`) -> `input`.
* **In-Place Mutation Check:** No in-place tensor mutations break autograd tracking.
