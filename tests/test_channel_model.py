import unittest
import torch
import numpy as np
from src.channel.system_model import SystemModel
from src.channel.ris_channel import RISChannelGenerator

class TestChannelModel(unittest.TestCase):

    def test_system_model_effective_channel(self):
        sys_model = SystemModel(n_ris_elements=64, n_bs_antennas=8, n_users=2, bits=2)
        gen = RISChannelGenerator(n_ris_elements=64, n_bs_antennas=8, n_users=2)
        
        h_br, h_ru, h_bu = gen.generate_batch(4)
        t_h_br = torch.tensor(h_br, dtype=torch.complex64)
        t_h_ru = torch.tensor(h_ru, dtype=torch.complex64)
        t_h_bu = torch.tensor(h_bu, dtype=torch.complex64)
        
        theta = torch.rand(4, 64) * 2 * torch.pi
        
        H_eff = sys_model.compute_effective_channel(t_h_br, t_h_ru, t_h_bu, theta)
        self.assertEqual(H_eff.shape, (4, 2, 8))

    def test_active_beamformer_power_constraint(self):
        sys_model = SystemModel(n_ris_elements=64, n_bs_antennas=8, n_users=2)
        gen = RISChannelGenerator(n_ris_elements=64, n_bs_antennas=8, n_users=2)
        h_br, h_ru, h_bu = gen.generate_batch(2)
        
        t_h_br = torch.tensor(h_br, dtype=torch.complex64)
        t_h_ru = torch.tensor(h_ru, dtype=torch.complex64)
        t_h_bu = torch.tensor(h_bu, dtype=torch.complex64)
        theta = torch.zeros(2, 64)
        
        H_eff = sys_model.compute_effective_channel(t_h_br, t_h_ru, t_h_bu, theta)
        W = sys_model.compute_active_beamformers_zf(H_eff, p_max=1.0)
        
        power = torch.sum(torch.abs(W)**2, dim=(1,2))
        self.assertTrue(torch.allclose(power, torch.ones_like(power), atol=1e-3))

if __name__ == "__main__":
    unittest.main()
