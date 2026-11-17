import torch
import torch.nn as nn
from .spatial_attention import SpatialSelfAttention

class ResBlock2D(nn.Module):
    def __init__(self, channels: int):
        super(ResBlock2D, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(channels)
        self.act   = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + residual)

class ComplexCNNBackbone(nn.Module):
    """
    CNN Spatial Feature Extractor with ResNet blocks and Spatial Self-Attention.
    Accepts input tensor X: [B, 2, K*M, N] representing real/imaginary parts of cascaded CSI.
    Outputs feature vector z: [B, feature_dim].
    """

    def __init__(self, in_channels: int = 2, feature_dim: int = 256):
        super(ComplexCNNBackbone, self).__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.res1 = ResBlock2D(64)
        self.attention = SpatialSelfAttention(64)
        
        self.downsample = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.res2 = ResBlock2D(128)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc_proj = nn.Sequential(
            nn.Linear(128, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 2, K*M, N]
        returns z: [B, feature_dim]
        """
        out = self.stem(x)
        out = self.res1(out)
        out = self.attention(out)
        out = self.downsample(out)
        out = self.res2(out)
        out = self.global_pool(out).flatten(1)
        z = self.fc_proj(out)
        return z
