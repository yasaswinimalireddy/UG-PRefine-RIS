import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from src.channel.ris_channel import RISChannelGenerator
from src.channel.csi_noise import CSINoiseModel
from src.channel.system_model import SystemModel

class RISChannelDataset(Dataset):
    """
    PyTorch Dataset for RIS Cascaded Channel Realizations.
    Stores raw complex matrices and prepares real/imag 2D spatial feature tensors.
    """

    def __init__(
        self,
        n_samples: int = 2000,
        n_ris_elements: int = 64,
        n_bs_antennas: int = 8,
        n_users: int = 2,
        csi_nmse: float = 0.05,
        seed: int = 42,
        compute_ao_targets: bool = False
    ):
        self.N = n_ris_elements
        self.M = n_bs_antennas
        self.K = n_users
        
        channel_gen = RISChannelGenerator(
            n_ris_elements=n_ris_elements,
            n_bs_antennas=n_bs_antennas,
            n_users=n_users,
            seed=seed
        )
        noise_model = CSINoiseModel(nmse_delta_sq=csi_nmse, seed=seed + 1)
        
        # Generate perfect channels
        self.H_BR_true, self.H_RU_true, self.H_BU_true = channel_gen.generate_batch(n_samples)
        
        # Apply CSI noise model
        self.H_BR_noisy = noise_model.add_noise_numpy(self.H_BR_true)
        self.H_RU_noisy = noise_model.add_noise_numpy(self.H_RU_true)
        self.H_BU_noisy = noise_model.add_noise_numpy(self.H_BU_true)

        self.compute_ao_targets = compute_ao_targets
        if compute_ao_targets:
            from src.optimizers.alternating_optimization import AlternatingOptimization
            sys_model = SystemModel(n_ris_elements=n_ris_elements, n_bs_antennas=n_bs_antennas, n_users=n_users)
            ao_solver = AlternatingOptimization(sys_model, max_iters=2, num_starts=2)
            t_h_br = torch.tensor(self.H_BR_true, dtype=torch.complex64)
            t_h_ru = torch.tensor(self.H_RU_true, dtype=torch.complex64)
            t_h_bu = torch.tensor(self.H_BU_true, dtype=torch.complex64)
            self.target_theta, self.target_rate, _ = ao_solver.solve_batch(t_h_br, t_h_ru, t_h_bu, discrete=True)

    def __len__(self) -> int:
        return len(self.H_BR_true)

    def __getitem__(self, idx: int) -> dict:
        h_br = self.H_BR_noisy[idx] # [N, M]
        h_ru = self.H_RU_noisy[idx] # [K, N]
        h_bu = self.H_BU_noisy[idx] # [K, M]
        
        # Compute cascaded channel H_cas: H_cas[k, n, m] = H_RU[k, n] * H_BR[n, m]
        # Shape: [K, N, M]
        h_cas = h_ru[:, :, None] * h_br[None, :, :]
        
        # Flatten (K * M) dimension to form spatial tensor [2, K*M, N]
        # Transpose H_cas to [K, M, N] -> reshape to [K*M, N]
        h_cas_spatial = h_cas.transpose(0, 2, 1).reshape(self.K * self.M, self.N)
        
        # Separate Real and Imaginary components -> Shape: [2, K*M, N]
        tensor_x = np.stack([np.real(h_cas_spatial), np.imag(h_cas_spatial)], axis=0).astype(np.float32)
        
        # Normalize tensor
        norm = np.linalg.norm(tensor_x) + 1e-8
        tensor_x = tensor_x / norm

        sample_dict = {
            "x": torch.from_numpy(tensor_x),
            "h_br": torch.from_numpy(self.H_BR_true[idx].astype(np.complex64)),
            "h_ru": torch.from_numpy(self.H_RU_true[idx].astype(np.complex64)),
            "h_bu": torch.from_numpy(self.H_BU_true[idx].astype(np.complex64)),
            "h_br_noisy": torch.from_numpy(self.H_BR_noisy[idx].astype(np.complex64)),
            "h_ru_noisy": torch.from_numpy(self.H_RU_noisy[idx].astype(np.complex64)),
            "h_bu_noisy": torch.from_numpy(self.H_BU_noisy[idx].astype(np.complex64)),
        }

        if self.compute_ao_targets:
            sample_dict["target_rate"] = torch.tensor(self.target_rate[idx], dtype=torch.float32)

        return sample_dict

def get_dataloaders(config: dict) -> tuple:
    """
    Factory function returning (train_loader, val_loader, test_loader).
    """
    cfg_sys = config["system"]
    cfg_train = config["training"]
    cfg_ds = config.get("dataset", {})
    
    n_train = cfg_ds.get("num_train_samples", 2000)
    n_val = cfg_ds.get("num_val_samples", 500)
    n_test = cfg_ds.get("num_test_samples", 500)
    seed = cfg_train.get("seed", 42)
    
    train_ds = RISChannelDataset(
        n_samples=n_train,
        n_ris_elements=cfg_sys["n_ris_elements"],
        n_bs_antennas=cfg_sys["n_bs_antennas"],
        n_users=cfg_sys["n_users"],
        csi_nmse=cfg_sys["csi_noise_nmse"],
        seed=seed
    )
    
    val_ds = RISChannelDataset(
        n_samples=n_val,
        n_ris_elements=cfg_sys["n_ris_elements"],
        n_bs_antennas=cfg_sys["n_bs_antennas"],
        n_users=cfg_sys["n_users"],
        csi_nmse=cfg_sys["csi_noise_nmse"],
        seed=seed + 100
    )
    
    test_ds = RISChannelDataset(
        n_samples=n_test,
        n_ris_elements=cfg_sys["n_ris_elements"],
        n_bs_antennas=cfg_sys["n_bs_antennas"],
        n_users=cfg_sys["n_users"],
        csi_nmse=cfg_sys["csi_noise_nmse"],
        seed=seed + 200
    )
    
    batch_size = cfg_train.get("batch_size", 64)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
