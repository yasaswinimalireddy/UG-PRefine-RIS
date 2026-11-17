import unittest
import torch
import numpy as np
from src.channel.system_model import SystemModel
from src.models.gumbel_quantizer import GumbelSoftmaxQuantizer

class TestQuantization(unittest.TestCase):

    def test_hard_quantization_discrete_states(self):
        sys_model = SystemModel(n_ris_elements=16, bits=2) # 4 discrete states: 0, pi/2, pi, 3pi/2
        quantizer = GumbelSoftmaxQuantizer(n_ris_elements=16, bits=2)
        
        logits = torch.randn(2, 16, 4)
        theta, probs = quantizer(logits, hard=True)
        
        # Check that phase angles theta belong strictly to phase alphabet F_2
        alphabet = sys_model.phase_alphabet_torch
        for angle in theta.flatten():
            diffs = torch.abs(angle - alphabet)
            self.assertLess(torch.min(diffs).item(), 1e-4)

    def test_gumbel_softmax_probabilities(self):
        quantizer = GumbelSoftmaxQuantizer(n_ris_elements=16, bits=2)
        logits = torch.randn(4, 16, 4)
        theta, probs = quantizer(logits, temperature=1.0)
        
        prob_sums = probs.sum(dim=-1)
        self.assertTrue(torch.allclose(prob_sums, torch.ones_like(prob_sums)))

if __name__ == "__main__":
    unittest.main()
