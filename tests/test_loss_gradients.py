import unittest
import torch
from src.channel.system_model import SystemModel
from src.models.ug_prefine_model import UGPRefineRISModel
from src.losses.composite_loss import CompositeRISLoss

class TestLossGradients(unittest.TestCase):

    def test_loss_backward_gradients(self):
        sys_model = SystemModel(n_ris_elements=32, n_bs_antennas=4, n_users=2, bits=2)
        model = UGPRefineRISModel(sys_model, feature_dim=128)
        loss_fn = CompositeRISLoss(sys_model)
        
        x = torch.randn(2, 2, 8, 32)
        h_br = torch.randn(2, 32, 4, dtype=torch.complex64)
        h_ru = torch.randn(2, 2, 32, dtype=torch.complex64)
        h_bu = torch.randn(2, 2, 4, dtype=torch.complex64)
        
        outputs = model(x, H_BR=h_br, H_RU=h_ru, H_BU=h_bu)
        loss, _ = loss_fn(outputs, h_br, h_ru, h_bu)
        
        loss.backward()
        
        # Check that gradients flow to model parameters
        grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
        self.assertGreater(len(grad_norms), 0, "No gradients accumulated!")
        self.assertTrue(all(not torch.isnan(torch.tensor(g)) for g in grad_norms), "NaN detected in gradients!")

if __name__ == "__main__":
    unittest.main()
