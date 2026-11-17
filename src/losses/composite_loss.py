import torch
import torch.nn as nn
from .rate_loss import AchievableRateLoss
from .uncertainty_loss import HeteroscedasticUncertaintyLoss

class CompositeRISLoss(nn.Module):
    """
    Composite Multi-Objective Loss for Physics-Aware and Uncertainty-Calibrated RIS Training.
    """

    def __init__(
        self,
        system_model,
        lambda_rate: float = 1.0,
        lambda_uncertainty: float = 0.2,
        lambda_phase_circular: float = 0.1,
        lambda_quantization: float = 0.05
    ):
        super(CompositeRISLoss, self).__init__()
        self.rate_loss_fn = AchievableRateLoss(system_model)
        self.uncertainty_loss_fn = HeteroscedasticUncertaintyLoss()
        
        self.lambda_rate = lambda_rate
        self.lambda_unc = lambda_uncertainty
        self.lambda_circ = lambda_phase_circular
        self.lambda_quant = lambda_quantization

    def forward(
        self,
        outputs: dict,
        H_BR: torch.Tensor,
        H_RU: torch.Tensor,
        H_BU: torch.Tensor,
        target_rate: torch.Tensor = None
    ) -> tuple:
        """
        outputs: dictionary returned by UGPRefineRISModel forward pass.
        """
        theta = outputs["theta"]
        var = outputs["uncertainty_var"]
        probs = outputs["phase_probs"]
        
        # 1. Achievable Rate Loss
        loss_rate, sum_rate = self.rate_loss_fn(theta, H_BR, H_RU, H_BU)
        
        # 2. Epistemic Uncertainty Loss
        if target_rate is not None:
            loss_unc = self.uncertainty_loss_fn(sum_rate.detach(), target_rate, var)
        else:
            # Self-supervised baseline target: max of batch
            target_pseudo = sum_rate.max() * torch.ones_like(sum_rate)
            loss_unc = self.uncertainty_loss_fn(sum_rate.detach(), target_pseudo, var)
            
        # 3. Categorical Quantization Entropy Regularization
        # Encourages sharp discrete phase assignments
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        loss_quant = torch.mean(entropy)
        
        # Composite Loss
        total_loss = (
            self.lambda_rate * loss_rate +
            self.lambda_unc * loss_unc +
            self.lambda_quant * loss_quant
        )
        
        return total_loss, {
            "total_loss": total_loss.item(),
            "rate_loss": loss_rate.item(),
            "uncertainty_loss": loss_unc.item(),
            "quant_loss": loss_quant.item(),
            "mean_sum_rate": torch.mean(sum_rate).item()
        }
