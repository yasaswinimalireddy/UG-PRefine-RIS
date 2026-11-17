import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GumbelSoftmaxQuantizer(nn.Module):
    """
    Differentiable Gumbel-Softmax Categorical Quantizer for RIS Phase Shifts.
    Maps phase logits [B, N, 2^b] to phase angles theta [B, N].
    """

    def __init__(self, n_ris_elements: int = 64, bits: int = 2):
        super(GumbelSoftmaxQuantizer, self).__init__()
        self.N = n_ris_elements
        self.bits = bits
        self.num_states = 2 ** bits
        
        # Discrete phase states F_b in radians
        alphabet = np.linspace(0, 2 * np.pi, self.num_states, endpoint=False)
        self.register_buffer("phase_alphabet", torch.tensor(alphabet, dtype=torch.float32))

    def forward(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        hard: bool = False
    ) -> tuple:
        """
        logits: [B, N, 2^b]
        
        Returns:
            theta_soft: [B, N] (differentiable phase phasor angle tensor)
            probs: [B, N, 2^b] (categorical probabilities)
        """
        B, N, S = logits.shape
        
        if self.training:
            # Gumbel-Softmax relaxation during backprop
            gumbels = -torch.empty_like(logits).exponential_().log() # ~ Gumbel(0,1)
            gumbel_logits = (logits + gumbels) / temperature
            probs = F.softmax(gumbel_logits, dim=-1)
        else:
            probs = F.softmax(logits / temperature, dim=-1)
            
        if hard or not self.training:
            # Hard quantization via argmax
            index = probs.max(-1, keepdim=True)[1]
            probs_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
            # Straight-Through Estimator (STE) trick
            probs = probs_hard - probs.detach() + probs

        # Expected phase phasor phasor_soft = sum_m p_m * exp(j * phi_m)
        sin_phase = torch.sum(probs * torch.sin(self.phase_alphabet), dim=-1)
        cos_phase = torch.sum(probs * torch.cos(self.phase_alphabet), dim=-1)
        
        # Soft continuous phase angle theta = atan2(sin_phase, cos_phase) wrapped in [0, 2*pi)
        theta = torch.atan2(sin_phase, cos_phase)
        theta = torch.remainder(theta, 2 * np.pi)
        
        return theta, probs
