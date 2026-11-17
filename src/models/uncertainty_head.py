import torch
import torch.nn as nn

class EpistemicUncertaintyHead(nn.Module):
    """
    Predicts heteroscedastic uncertainty (variance estimate sigma^2 > 0) per input channel realization.
    """

    def __init__(self, feature_dim: int = 256, hidden_dim: int = 128):
        super(EpistemicUncertaintyHead, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Softplus() # Guarantees strictly positive variance output
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: [B, feature_dim]
        returns var: [B, 1]
        """
        # Softplus output + small epsilon to avoid zero division in loss
        var = self.net(z) + 1e-6
        return var
