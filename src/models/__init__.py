"""
Deep Neural Network modules for RIS Phase Shift Optimization.
"""

from .spatial_attention import SpatialSelfAttention
from .backbone_cnn import ComplexCNNBackbone
from .uncertainty_head import EpistemicUncertaintyHead
from .gumbel_quantizer import GumbelSoftmaxQuantizer
from .ug_prefine_model import UGPRefineRISModel

__all__ = [
    "SpatialSelfAttention",
    "ComplexCNNBackbone",
    "EpistemicUncertaintyHead",
    "GumbelSoftmaxQuantizer",
    "UGPRefineRISModel",
]
