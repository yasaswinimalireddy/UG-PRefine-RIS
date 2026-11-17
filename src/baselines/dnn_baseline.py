import torch
import torch.nn as nn
from src.channel.system_model import SystemModel

class DNNBaselineModel(nn.Module):
    """
    Conventional Fully Connected Deep Neural Network (DNN) Baseline.
    Vectorizes 2D channel tensor X [B, 2, K*M, N] -> MLP -> phase angles [B, N].
    """

    def __init__(self, system_model: SystemModel, hidden_dim: int = 512):
        super(DNNBaselineModel, self).__init__()
        self.sys = system_model
        self.N = system_model.N
        self.M = system_model.M
        self.K = system_model.K
        
        in_dim = 2 * (self.K * self.M) * self.N
        
        self.net = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.N),
            nn.Sigmoid() # Scale to [0, 2*pi]
        )

    def forward(self, x: torch.Tensor, **kwargs) -> dict:
        theta_raw = self.net(x) * 2 * torch.pi
        theta_quant = self.sys.quantize_phase(theta_raw)
        return {"theta": theta_quant, "theta_pred": theta_raw}
