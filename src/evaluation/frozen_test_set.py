import os
import torch
import numpy as np
from src.channel.ris_channel import RISChannelGenerator
from src.channel.csi_noise import CSINoiseModel

class FrozenTestSet:
    """
    Generates, saves, and loads a canonical frozen benchmark test set of 500 channel samples.
    Ensures 100% identical test channels across all baseline algorithms.
    """

    def __init__(
        self,
        n_samples: int = 500,
        n_ris_elements: int = 64,
        n_bs_antennas: int = 8,
        n_users: int = 2,
        csi_nmse: float = 0.05,
        seed: int = 42000,
        file_path: str = "data/frozen_test_set_64.pt"
    ):
        self.n_samples = n_samples
        self.N = n_ris_elements
        self.M = n_bs_antennas
        self.K = n_users
        self.csi_nmse = csi_nmse
        self.seed = seed
        self.file_path = file_path

    def generate_and_save(self) -> dict:
        channel_gen = RISChannelGenerator(
            n_ris_elements=self.N,
            n_bs_antennas=self.M,
            n_users=self.K,
            seed=self.seed
        )
        noise_model = CSINoiseModel(nmse_delta_sq=self.csi_nmse, seed=self.seed + 1)

        h_br_true, h_ru_true, h_bu_true = channel_gen.generate_batch(self.n_samples)
        h_br_noisy = noise_model.add_noise_numpy(h_br_true)
        h_ru_noisy = noise_model.add_noise_numpy(h_ru_true)
        h_bu_noisy = noise_model.add_noise_numpy(h_bu_true)

        data_dict = {
            "h_br_true": torch.tensor(h_br_true, dtype=torch.complex64),
            "h_ru_true": torch.tensor(h_ru_true, dtype=torch.complex64),
            "h_bu_true": torch.tensor(h_bu_true, dtype=torch.complex64),
            "h_br_noisy": torch.tensor(h_br_noisy, dtype=torch.complex64),
            "h_ru_noisy": torch.tensor(h_ru_noisy, dtype=torch.complex64),
            "h_bu_noisy": torch.tensor(h_bu_noisy, dtype=torch.complex64),
            "n_samples": self.n_samples,
            "N": self.N,
            "M": self.M,
            "K": self.K,
            "csi_nmse": self.csi_nmse,
            "seed": self.seed
        }

        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        torch.save(data_dict, self.file_path)
        return data_dict

    def load_or_create(self) -> dict:
        if os.path.exists(self.file_path):
            return torch.load(self.file_path)
        else:
            return self.generate_and_save()
