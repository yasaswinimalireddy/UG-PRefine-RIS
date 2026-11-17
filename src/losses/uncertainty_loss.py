import torch
import torch.nn as nn

class HeteroscedasticUncertaintyLoss(nn.Module):
    """
    Epistemic Uncertainty Calibration Loss based on Heteroscedastic Gaussian NLL:
    L = |R_target - R_pred|^2 / (2 * var) + 0.5 * log(var)
    """

    def __init__(self):
        super(HeteroscedasticUncertaintyLoss, self).__init__()

    def forward(
        self,
        predicted_rate: torch.Tensor,
        target_rate: torch.Tensor,
        predicted_var: torch.Tensor
    ) -> torch.Tensor:
        """
        predicted_rate: [B]
        target_rate: [B]
        predicted_var: [B, 1] or [B]
        """
        var = predicted_var.view(-1)
        residual_sq = (target_rate - predicted_rate) ** 2
        
        nll = (residual_sq / (2.0 * var)) + 0.5 * torch.log(var)
        return torch.mean(nll)
