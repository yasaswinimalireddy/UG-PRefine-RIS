import torch
import torch.nn as nn
import numpy as np

class SystemModel:
    """
    Physical System Model for MU-MISO Wireless System aided by a Passive RIS.
    
    BS antennas: M
    RIS elements: N
    Users: K
    Quantization bits: b (discrete phase set of size 2^b)
    """

    def __init__(
        self,
        n_ris_elements: int = 64,
        n_bs_antennas: int = 8,
        n_users: int = 2,
        bits: int = 2,
        snr_db: float = 15.0,
        coupled_amplitude: bool = True,
        beta_min: float = 0.2,
        phi_rad: float = 1.35088,
        gamma_exp: float = 1.6,
        device: str = "cpu"
    ):
        self.N = n_ris_elements
        self.M = n_bs_antennas
        self.K = n_users
        self.b = bits
        self.snr_db = snr_db
        self.noise_power = 1.0 / (10.0 ** (snr_db / 10.0))
        self.coupled_amplitude = coupled_amplitude
        self.beta_min = beta_min
        self.phi_rad = phi_rad
        self.gamma_exp = gamma_exp
        self.device = torch.device(device if torch.cuda.is_available() and device != "cpu" else "cpu")

        # Discrete phase alphabet F_b
        if bits > 0:
            self.num_states = 2 ** bits
            self.phase_alphabet = np.linspace(0, 2 * np.pi, self.num_states, endpoint=False)
            self.phase_alphabet_torch = torch.tensor(self.phase_alphabet, dtype=torch.float32, device=self.device)
        else:
            self.num_states = None
            self.phase_alphabet = None

    def compute_amplitude(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Compute element reflection amplitude beta(theta) based on practical phase-amplitude coupling.
        beta(theta) = (1 - beta_min) * ((sin(theta - phi) + 1) / 2)^gamma + beta_min
        """
        if not self.coupled_amplitude:
            return torch.ones_like(theta)
        
        sin_term = (torch.sin(theta - self.phi_rad) + 1.0) / 2.0
        beta = (1.0 - self.beta_min) * (sin_term ** self.gamma_exp) + self.beta_min
        return beta

    def quantize_phase(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Hard nearest-neighbor quantization to discrete phase alphabet F_b.
        """
        if self.b == 0 or self.phase_alphabet_torch is None:
            return torch.remainder(theta, 2 * np.pi)
        
        # Wrap theta to [0, 2*pi)
        theta_wrapped = torch.remainder(theta, 2 * np.pi)
        
        # Reshape for broadcasting
        # theta_wrapped: [..., N], alphabet: [2^b]
        diffs = torch.abs(theta_wrapped.unsqueeze(-1) - self.phase_alphabet_torch)
        # Find minimum distance along last axis
        nearest_idx = torch.argmin(diffs, dim=-1)
        quantized = self.phase_alphabet_torch[nearest_idx]
        return quantized

    def compute_effective_channel(
        self,
        H_BR: torch.Tensor,
        H_RU: torch.Tensor,
        H_BU: torch.Tensor,
        theta: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute effective channel H_eff = H_RU @ Theta @ H_BR + H_BU
        
        Parameters:
            H_BR: [B, N, M] (complex tensor)
            H_RU: [B, K, N] (complex tensor)
            H_BU: [B, K, M] (complex tensor)
            theta: [B, N] (real phase tensor in radians)
            
        Returns:
            H_eff: [B, K, M] (complex tensor)
        """
        beta = self.compute_amplitude(theta)
        # Complex diagonal phasor: [B, N]
        phasor = beta * torch.exp(1j * theta)
        
        # H_RU * Theta: [B, K, N] * [B, 1, N] -> [B, K, N]
        H_RU_theta = H_RU * phasor.unsqueeze(1)
        
        # H_eff = H_RU_theta @ H_BR + H_BU: [B, K, N] @ [B, N, M] -> [B, K, M]
        H_eff = torch.bmm(H_RU_theta, H_BR) + H_BU
        return H_eff

    def compute_active_beamformers_zf(
        self,
        H_eff: torch.Tensor,
        p_max: float = 1.0
    ) -> torch.Tensor:
        """
        Compute Zero-Forcing (ZF) active transmit beamformers w_k at the BS.
        
        H_eff: [B, K, M] (complex tensor)
        Returns:
            W: [B, K, M] (complex transmit beamformer tensor per user)
        """
        B, K, M = H_eff.shape
        # Pseudo-inverse of H_eff: H_eff_dagger = H_eff^H (H_eff H_eff^H)^-1
        # H_eff: [B, K, M] -> H_eff_H: [B, M, K]
        H_eff_H = H_eff.mconjh() if hasattr(H_eff, 'mconjh') else H_eff.conj().transpose(-2, -1)
        
        # Gram matrix G = H_eff @ H_eff^H: [B, K, K]
        Gram = torch.bmm(H_eff, H_eff_H)
        
        # Add small regularization for numerical stability
        reg = 1e-6 * torch.eye(K, device=H_eff.device, dtype=Gram.dtype).unsqueeze(0)
        Gram_inv = torch.linalg.inv(Gram + reg)
        
        # Unnormalized beamforming matrix V = H_eff_H @ Gram_inv: [B, M, K]
        V = torch.bmm(H_eff_H, Gram_inv) # [B, M, K]
        
        # Transpose to [B, K, M]
        W_raw = V.transpose(-2, -1) # [B, K, M]
        
        # Power normalization across all users
        power_raw = torch.sum(torch.abs(W_raw) ** 2, dim=(1, 2), keepdim=True) # [B, 1, 1]
        scale = torch.sqrt(p_max / torch.clamp(power_raw, min=1e-8))
        W = W_raw * scale
        return W

    def compute_achievable_sum_rate(
        self,
        H_eff: torch.Tensor,
        W: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Sum Achievable Rate (Spectral Efficiency in bps/Hz) given H_eff and active beamformers W.
        
        H_eff: [B, K, M] (complex)
        W: [B, K, M] (complex)
        
        Returns:
            sum_rate: [B] (real float tensor)
        """
        B, K, M = H_eff.shape
        sinrs = []
        
        for k in range(K):
            h_k = H_eff[:, k, :] # [B, M]
            w_k = W[:, k, :]     # [B, M]
            
            # Desired signal power |h_k w_k|^2
            signal_k = torch.abs(torch.sum(h_k * w_k, dim=-1)) ** 2 # [B]
            
            # Interference power sum_{j != k} |h_k w_j|^2
            interference_k = torch.zeros(B, device=H_eff.device)
            for j in range(K):
                if j != k:
                    w_j = W[:, j, :]
                    interference_k = interference_k + torch.abs(torch.sum(h_k * w_j, dim=-1)) ** 2
            
            sinr_k = signal_k / (interference_k + self.noise_power)
            sinrs.append(sinr_k)
            
        # Sum rate across users: sum_k log2(1 + sinr_k)
        rates = [torch.log2(1.0 + sinr) for sinr in sinrs]
        sum_rate = torch.stack(rates, dim=-1).sum(dim=-1) # [B]
        return sum_rate
