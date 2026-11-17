import unittest
import torch
import numpy as np
from src.channel.system_model import SystemModel
from src.channel.ris_channel import RISChannelGenerator
from src.evaluation.canonical_evaluator import CanonicalEvaluator

class TestCanonicalEvaluator(unittest.TestCase):

    def setUp(self):
        self.sys_model = SystemModel(n_ris_elements=32, n_bs_antennas=4, n_users=2, bits=2)
        self.evaluator = CanonicalEvaluator(self.sys_model)
        self.gen = RISChannelGenerator(n_ris_elements=32, n_bs_antennas=4, n_users=2, seed=42)

    def test_evaluator_output_shapes_and_types(self):
        h_br, h_ru, h_bu = self.gen.generate_batch(4)
        t_h_br = torch.tensor(h_br, dtype=torch.complex64)
        t_h_ru = torch.tensor(h_ru, dtype=torch.complex64)
        t_h_bu = torch.tensor(h_bu, dtype=torch.complex64)
        
        theta = torch.rand(4, 32) * 2 * np.pi
        theta_quant = self.sys_model.quantize_phase(theta)

        res = self.evaluator.evaluate(t_h_br, t_h_ru, t_h_bu, theta_quant)

        self.assertEqual(res["sum_rate"].shape, (4,))
        self.assertEqual(res["sinrs"].shape, (4, 2))
        self.assertEqual(res["user_rates"].shape, (4, 2))
        self.assertGreaterEqual(res["mean_sum_rate"], 0.0)
        self.assertEqual(res["violation_rate"], 0.0)

    def test_unquantized_phase_detects_violations(self):
        h_br, h_ru, h_bu = self.gen.generate_batch(2)
        t_h_br = torch.tensor(h_br, dtype=torch.complex64)
        t_h_ru = torch.tensor(h_ru, dtype=torch.complex64)
        t_h_bu = torch.tensor(h_bu, dtype=torch.complex64)

        # Unquantized random phase angle (e.g. 0.1234 radians)
        theta_continuous = torch.full((2, 32), 0.1234)
        res = self.evaluator.evaluate(t_h_br, t_h_ru, t_h_bu, theta_continuous)

        self.assertGreater(res["violation_rate"], 0.0)

if __name__ == "__main__":
    unittest.main()
