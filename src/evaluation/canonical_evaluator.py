import torch
import numpy as np
from src.channel.system_model import SystemModel

class CanonicalEvaluator:
    """
    Single, authoritative, non-negotiable evaluation pipeline for RIS phase optimization.
    Guarantees every algorithm is evaluated using identical:
    - Physical RIS response beta(theta) * exp(j theta)
    - Effective channel H_eff = H_RU @ Theta @ H_BR + H_BU
    - Active Zero-Forcing transmit beamformer scaling
    - Transmit power budget P_max
    - Thermal noise power N_0
    - Achievable sum rate formula sum_k log2(1 + SINR_k)
    """

    def __init__(self, system_model: SystemModel):
        self.sys = system_model

    def evaluate(
        self,
        H_BR: torch.Tensor,
        H_RU: torch.Tensor,
        H_BU: torch.Tensor,
        theta: torch.Tensor
    ) -> dict:
        """
        H_BR: [B, N, M] (complex tensor)
        H_RU: [B, K, N] (complex tensor)
        H_BU: [B, K, M] (complex tensor)
        theta: [B, N] (real tensor of phase angles in radians)

        Returns dict:
            "effective_channel": H_eff [B, K, M]
            "active_beamformers": W [B, K, M]
            "sinrs": [B, K]
            "user_rates": [B, K]
            "sum_rate": [B]
            "mean_sum_rate": float
            "constraint_violations": [B] (boolean mask of elements violating F_b)
            "violation_rate": float
        """
        device = H_BR.device
        theta_tensor = theta.to(device)

        # 1. Physical RIS response & Effective Channel
        H_eff = self.sys.compute_effective_channel(H_BR, H_RU, H_BU, theta_tensor)

        # 2. Active Zero-Forcing Beamforming
        W = self.sys.compute_active_beamformers_zf(H_eff, p_max=1.0)

        # 3. SINR & Rate Computation per user
        B, K, M = H_eff.shape
        sinr_list = []
        rate_list = []

        for k in range(K):
            h_k = H_eff[:, k, :] # [B, M]
            w_k = W[:, k, :]     # [B, M]

            # Desired signal power |h_k w_k|^2
            signal_k = torch.abs(torch.sum(h_k * w_k, dim=-1)) ** 2 # [B]

            # Interference power sum_{j != k} |h_k w_j|^2
            interference_k = torch.zeros(B, device=device)
            for j in range(K):
                if j != k:
                    w_j = W[:, j, :]
                    interference_k = interference_k + torch.abs(torch.sum(h_k * w_j, dim=-1)) ** 2

            sinr_k = signal_k / (interference_k + self.sys.noise_power)
            rate_k = torch.log2(1.0 + sinr_k)

            sinr_list.append(sinr_k)
            rate_list.append(rate_k)

        sinrs = torch.stack(sinr_list, dim=-1) # [B, K]
        user_rates = torch.stack(rate_list, dim=-1) # [B, K]
        sum_rate = torch.sum(user_rates, dim=-1) # [B]

        # 4. Hardware Quantization Constraint Check
        if self.sys.phase_alphabet_torch is not None:
            diffs = torch.abs(theta_tensor.unsqueeze(-1) - self.sys.phase_alphabet_torch.to(device))
            min_diffs = torch.min(diffs, dim=-1)[0]
            constraint_violations = (min_diffs > 1e-3) # [B, N]
            violation_rate = float(constraint_violations.float().mean().item())
        else:
            constraint_violations = torch.zeros_like(theta_tensor, dtype=torch.bool)
            violation_rate = 0.0

        return {
            "effective_channel": H_eff,
            "active_beamformers": W,
            "sinrs": sinrs,
            "user_rates": user_rates,
            "sum_rate": sum_rate,
            "mean_sum_rate": float(torch.mean(sum_rate).item()),
            "std_sum_rate": float(torch.std(sum_rate).item()),
            "constraint_violations": constraint_violations,
            "violation_rate": violation_rate
        }
