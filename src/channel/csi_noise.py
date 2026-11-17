import numpy as np
import torch

class CSINoiseModel:
    """
    Imperfect CSI Noise Model adding complex Gaussian error E ~ CN(0, sigma_e^2 I) to channel matrices.
    """

    def __init__(self, nmse_delta_sq: float = 0.05, seed: int = 42):
        self.nmse = nmse_delta_sq
        self.rng = np.random.default_rng(seed)

    def add_noise_numpy(self, H: np.ndarray) -> np.ndarray:
        """
        Add CSI estimation noise to numpy channel matrix H.
        """
        if self.nmse <= 0.0:
            return H.copy()
        
        signal_power = np.mean(np.abs(H) ** 2)
        noise_variance = signal_power * self.nmse
        noise_std = np.sqrt(noise_variance / 2.0)
        
        noise = self.rng.normal(scale=noise_std, size=H.shape) + 1j * self.rng.normal(scale=noise_std, size=H.shape)
        return H + noise

    def add_noise_torch(self, H: torch.Tensor) -> torch.Tensor:
        """
        Add CSI estimation noise to PyTorch complex tensor H.
        """
        if self.nmse <= 0.0:
            return H.clone()
        
        signal_power = torch.mean(torch.abs(H) ** 2)
        noise_variance = signal_power * self.nmse
        noise_std = torch.sqrt(noise_variance / 2.0)
        
        noise_real = torch.randn_like(H.real) * noise_std
        noise_imag = torch.randn_like(H.imag) * noise_std
        noise = torch.complex(noise_real, noise_imag)
        return H + noise
