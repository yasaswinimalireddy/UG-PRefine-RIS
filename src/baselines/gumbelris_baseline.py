import torch
import torch.nn as nn
from src.channel.system_model import SystemModel
from src.models.gumbel_quantizer import GumbelSoftmaxQuantizer

class GumbelRISBaselineModel(nn.Module):
    """
    Reproduction of GumbelRIS Baseline Architecture (Zhang et al., 2025).
    CNN + Gumbel-Softmax Categorical Relaxation for 2-bit discrete phase shifts.
    """

    def __init__(self, system_model: SystemModel):
        super(GumbelRISBaselineModel, self).__init__()
        self.sys = system_model
        self.N = system_model.N
        self.bits = system_model.b
        self.num_states = 2 ** self.bits
        
        self.conv = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1)
        )
        
        self.logits_head = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.N * self.num_states)
        )
        
        self.quantizer = GumbelSoftmaxQuantizer(n_ris_elements=self.N, bits=self.bits)

    def forward(self, x: torch.Tensor, gumbel_temp: float = 1.0, **kwargs) -> dict:
        B = x.size(0)
        feat = self.conv(x)
        logits = self.logits_head(feat).view(B, self.N, self.num_states)
        theta_pred, probs = self.quantizer(logits, temperature=gumbel_temp)
        return {"theta": theta_pred, "theta_pred": theta_pred, "phase_logits": logits, "phase_probs": probs}
