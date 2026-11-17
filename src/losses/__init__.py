"""
Loss functions for Physics-Aware and Uncertainty-Calibrated RIS phase optimization.
"""

from .rate_loss import AchievableRateLoss
from .uncertainty_loss import HeteroscedasticUncertaintyLoss
from .composite_loss import CompositeRISLoss

__all__ = ["AchievableRateLoss", "HeteroscedasticUncertaintyLoss", "CompositeRISLoss"]
