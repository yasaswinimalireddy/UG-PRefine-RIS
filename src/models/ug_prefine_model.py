import torch
import torch.nn as nn
from .backbone_cnn import ComplexCNNBackbone
from .uncertainty_head import EpistemicUncertaintyHead
from .gumbel_quantizer import GumbelSoftmaxQuantizer
from src.channel.system_model import SystemModel

class UGPRefineRISModel(nn.Module):
    """
    Uncertainty-Gated Differentiable Predict-Refine Neural Network for RIS Phase Optimization.
    """

    def __init__(
        self,
        system_model: SystemModel,
        feature_dim: int = 256,
        unrolled_pgd_steps: int = 3,
        pgd_step_size: float = 0.05
    ):
        super(UGPRefineRISModel, self).__init__()
        self.sys = system_model
        self.N = system_model.N
        self.bits = system_model.b
        self.num_states = 2 ** self.bits
        self.unrolled_steps = unrolled_pgd_steps
        self.pgd_eta = pgd_step_size
        
        # 1. Complex CNN Backbone
        self.backbone = ComplexCNNBackbone(in_channels=2, feature_dim=feature_dim)
        
        # 2. Dual Projection Heads
        self.uncertainty_head = EpistemicUncertaintyHead(feature_dim=feature_dim)
        
        self.phase_head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, self.N * self.num_states)
        )
        
        # 3. Gumbel-Softmax Quantizer
        self.quantizer = GumbelSoftmaxQuantizer(n_ris_elements=self.N, bits=self.bits)

    def unrolled_pgd_refine(
        self,
        theta_init: torch.Tensor,
        H_BR: torch.Tensor,
        H_RU: torch.Tensor,
        H_BU: torch.Tensor
    ) -> torch.Tensor:
        """
        Differentiable Projected Gradient Descent (PGD) Refinement Layer.
        Unrolls L steps of gradient ascent on the Sum Achievable Rate objective.
        """
        with torch.enable_grad():
            theta_curr = theta_init.clone().detach().requires_grad_(True)
            for step in range(self.unrolled_steps):
                H_eff = self.sys.compute_effective_channel(H_BR, H_RU, H_BU, theta_curr)
                W = self.sys.compute_active_beamformers_zf(H_eff)
                rate = self.sys.compute_achievable_sum_rate(H_eff, W)
                
                grad = torch.autograd.grad(
                    outputs=rate.sum(),
                    inputs=theta_curr,
                    create_graph=self.training,
                    retain_graph=True,
                    allow_unused=True
                )[0]
                
                if grad is not None:
                    # Continuous gradient ascent step
                    theta_curr = (theta_curr + self.pgd_eta * grad).detach().requires_grad_(True)
                    
            # Final hard discrete quantization to nearest 2-bit phase state F_b
            theta_final = self.sys.quantize_phase(theta_curr).detach()
                    
        return theta_final

    def forward(
        self,
        x: torch.Tensor,
        H_BR: torch.Tensor = None,
        H_RU: torch.Tensor = None,
        H_BU: torch.Tensor = None,
        gumbel_temp: float = 1.0,
        uncertainty_threshold: float = None,
        force_refine: bool = False
    ) -> dict:
        """
        x: [B, 2, K*M, N] (input spatial channel tensor)
        """
        B = x.size(0)
        
        # Feature extraction
        z = self.backbone(x)
        
        # Dual predictions
        uncertainty_var = self.uncertainty_head(z) # [B, 1]
        
        phase_logits = self.phase_head(z).view(B, self.N, self.num_states) # [B, N, 2^b]
        
        # Gumbel Softmax quantization
        theta_pred, probs = self.quantizer(phase_logits, temperature=gumbel_temp) # [B, N]
        
        # Selective Refinement Routing
        if force_refine or (H_BR is not None and uncertainty_threshold is not None):
            # Gated condition: trigger refinement for samples with variance > threshold
            if force_refine:
                high_uncertainty_mask = torch.ones(B, dtype=torch.bool, device=x.device)
            else:
                high_uncertainty_mask = (uncertainty_var.squeeze(-1) >= uncertainty_threshold)
            
            if high_uncertainty_mask.any():
                theta_refined = theta_pred.clone()
                # Refine only high uncertainty samples
                theta_refined[high_uncertainty_mask] = self.unrolled_pgd_refine(
                    theta_pred[high_uncertainty_mask],
                    H_BR[high_uncertainty_mask],
                    H_RU[high_uncertainty_mask],
                    H_BU[high_uncertainty_mask]
                )
                theta_final = theta_refined
            else:
                theta_final = theta_pred
        else:
            theta_final = theta_pred

        return {
            "theta": theta_final,
            "theta_pred": theta_pred,
            "uncertainty_var": uncertainty_var,
            "phase_logits": phase_logits,
            "phase_probs": probs
        }
