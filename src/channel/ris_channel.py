import numpy as np
import torch

class RISChannelGenerator:
    """
    Synthetic and DeepMIMO-compatible Channel Generator for RIS-aided MU-MISO networks.
    Generates H_BR (BS-to-RIS), H_RU (RIS-to-UE), and H_BU (BS-to-UE direct path).
    """

    def __init__(
        self,
        n_ris_elements: int = 64,
        n_bs_antennas: int = 8,
        n_users: int = 2,
        rician_k_db: float = 10.0,
        path_loss_exp_br: float = 2.2,
        path_loss_exp_ru: float = 2.8,
        path_loss_exp_bu: float = 3.8,
        direct_path_attenuation_db: float = 20.0,
        seed: int = 42
    ):
        self.N = n_ris_elements
        self.M = n_bs_antennas
        self.K = n_users
        self.rician_k_linear = 10.0 ** (rician_k_db / 10.0)
        self.pl_br = path_loss_exp_br
        self.pl_ru = path_loss_exp_ru
        self.pl_bu = path_loss_exp_bu
        self.direct_attenuation = 10.0 ** (-direct_path_attenuation_db / 20.0)
        self.rng = np.random.default_rng(seed)

    def generate_spatial_correlation(self, size: int, correlation_factor: float = 0.5) -> np.ndarray:
        """
        Generates a exponential spatial correlation matrix R where R_ij = corr^|i-j|.
        """
        i, j = np.indices((size, size))
        R = correlation_factor ** np.abs(i - j)
        # Matrix square root via Cholesky
        R_sqrt = np.linalg.cholesky(R)
        return R_sqrt

    def generate_sample(
        self,
        d_br: float = 50.0,
        d_ru_range: tuple = (10.0, 30.0),
        d_bu_range: tuple = (50.0, 75.0)
    ) -> tuple:
        """
        Generate a single realization of channel matrices (H_BR, H_RU, H_BU).
        
        Returns:
            H_BR: [N, M] complex numpy array
            H_RU: [K, N] complex numpy array
            H_BU: [K, M] complex numpy array
        """
        # Path loss factors
        alpha_br = np.sqrt(1.0 / (d_br ** self.pl_br))
        
        # 1. BS-to-RIS Channel (H_BR) - Dominant Rician LoS
        # LoS component using array response vectors
        a_bs = np.exp(-1j * np.pi * np.arange(self.M) * np.sin(self.rng.uniform(-np.pi/3, np.pi/3)))
        a_ris = np.exp(-1j * np.pi * np.arange(self.N) * np.sin(self.rng.uniform(-np.pi/3, np.pi/3)))
        H_br_los = np.outer(a_ris, a_bs.conj())
        
        # NLoS component
        H_br_nlos = (self.rng.normal(size=(self.N, self.M)) + 1j * self.rng.normal(size=(self.N, self.M))) / np.sqrt(2)
        
        H_BR = alpha_br * (
            np.sqrt(self.rician_k_linear / (1 + self.rician_k_linear)) * H_br_los +
            np.sqrt(1.0 / (1 + self.rician_k_linear)) * H_br_nlos
        )

        # Spatial correlation matrix for RIS elements
        R_ris_sqrt = self.generate_spatial_correlation(self.N, correlation_factor=0.3)

        # 2. RIS-to-UE Channel (H_RU) - NLoS Rayleigh / Moderate Rician per user
        H_RU_list = []
        H_BU_list = []
        
        for k in range(self.K):
            d_ru_k = self.rng.uniform(*d_ru_range)
            d_bu_k = self.rng.uniform(*d_bu_range)
            
            alpha_ru_k = np.sqrt(1.0 / (d_ru_k ** self.pl_ru))
            alpha_bu_k = np.sqrt(1.0 / (d_bu_k ** self.pl_bu)) * self.direct_attenuation
            
            # H_RU user k
            h_ru_raw = (self.rng.normal(size=(self.N,)) + 1j * self.rng.normal(size=(self.N,))) / np.sqrt(2)
            h_ru_k = alpha_ru_k * (R_ris_sqrt @ h_ru_raw)
            H_RU_list.append(h_ru_k)
            
            # H_BU direct path user k
            h_bu_k = alpha_bu_k * (self.rng.normal(size=(self.M,)) + 1j * self.rng.normal(size=(self.M,))) / np.sqrt(2)
            H_BU_list.append(h_bu_k)

        H_RU = np.stack(H_RU_list, axis=0) # [K, N]
        H_BU = np.stack(H_BU_list, axis=0) # [K, M]

        # Normalize channel gains to unit variance so SNR_dB directly sets link SNR
        H_BR = H_BR / (np.linalg.norm(H_BR) / np.sqrt(self.N * self.M) + 1e-8)
        H_RU = H_RU / (np.linalg.norm(H_RU) / np.sqrt(self.K * self.N) + 1e-8)
        H_BU = H_BU / (np.linalg.norm(H_BU) / np.sqrt(self.K * self.M) + 1e-8) * self.direct_attenuation

        return H_BR, H_RU, H_BU

    def generate_batch(self, batch_size: int) -> tuple:
        """
        Generate a batch of complex channel samples.
        
        Returns:
            H_BR: [B, N, M]
            H_RU: [B, K, N]
            H_BU: [B, K, M]
        """
        h_br_list, h_ru_list, h_bu_list = [], [], []
        for _ in range(batch_size):
            h_br, h_ru, h_bu = self.generate_sample()
            h_br_list.append(h_br)
            h_ru_list.append(h_ru)
            h_bu_list.append(h_bu)
            
        H_BR_batch = np.stack(h_br_list, axis=0)
        H_RU_batch = np.stack(h_ru_list, axis=0)
        H_BU_batch = np.stack(h_bu_list, axis=0)
        return H_BR_batch, H_RU_batch, H_BU_batch
