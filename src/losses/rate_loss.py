import torch
import torch.nn as nn
from src.channel.system_model import SystemModel

class AchievableRateLoss(nn.Module):
    """
    Unsupervised Communication Utility Loss: Negative Sum Achievable Rate -E[R_sum].
    """

    def __init__(self, system_model: SystemModel):
        super(AchievableRateLoss, self).__init__()
        self.sys = system_model

    def forward(
        self,
        theta: torch.Tensor,
        H_BR: torch.Tensor,
        H_RU: torch.Tensor,
        H_BU: torch.Tensor
    ) -> torch.Tensor:
        """
        theta: [B, N]
        H_BR: [B, N, M]
        H_RU: [B, K, N]
        H_BU: [B, K, M]
        
        Returns:
            loss: scalar loss (negative mean rate)
            sum_rate: [B] (rate tensor in bps/Hz)
        """
        H_eff = self.sys.compute_effective_channel(H_BR, H_RU, H_BU, theta)
        W = self.sys.compute_active_beamformers_zf(H_eff)
        sum_rate = self.sys.compute_achievable_sum_rate(H_eff, W)
        
        loss = -torch.mean(sum_rate)
        return loss, sum_rate
