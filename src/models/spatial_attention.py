import torch
import torch.nn as nn

class SpatialSelfAttention(nn.Module):
    """
    Spatial Self-Attention Module for capturing inter-element coupling across the RIS surface.
    Input tensor shape: [B, C, H, W] where W corresponds to the RIS element dimension N.
    """

    def __init__(self, in_channels: int):
        super(SpatialSelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv   = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]
        """
        B, C, H, W = x.size()
        
        # Query projection: [B, C//8, H*W] -> transpose to [B, H*W, C//8]
        query = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1)
        # Key projection: [B, C//8, H*W]
        key = self.key_conv(x).view(B, -1, H * W)
        
        # Attention map: [B, H*W, H*W]
        energy = torch.bmm(query, key)
        attention = self.softmax(energy)
        
        # Value projection: [B, C, H*W]
        value = self.value_conv(x).view(B, -1, H * W)
        
        # Weighted context: [B, C, H*W]
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        
        # Residual connection scaled by learnable parameter gamma
        return self.gamma * out + x
