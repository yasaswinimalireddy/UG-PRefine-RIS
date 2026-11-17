import torch
import torch.nn as nn
from src.channel.system_model import SystemModel

class StandardCNNBaselineModel(nn.Module):
    """
    Standard 2D Convolutional Neural Network (CNN) Baseline.
    Processes spatial tensor X [B, 2, K*M, N] with standard convolutions -> predicted phase.
    """

    def __init__(self, system_model: SystemModel):
        super(StandardCNNBaselineModel, self).__init__()
        self.sys = system_model
        self.N = system_model.N
        self.bits = system_model.b
        self.num_states = 2 ** self.bits
        
        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, self.N)
        )

    def forward(self, x: torch.Tensor, **kwargs) -> dict:
        out = self.conv(x)
        theta_raw = torch.sigmoid(self.fc(out)) * 2 * torch.pi
        theta_quant = self.sys.quantize_phase(theta_raw)
        return {"theta": theta_quant, "theta_pred": theta_raw}
