"""
Iterative mathematical optimization baseline algorithms for RIS phase shift optimization.
"""

from .alternating_optimization import AlternatingOptimization
from .projected_gradient import ProjectedGradientDescent
from .coordinate_descent import CoordinateDescent

__all__ = ["AlternatingOptimization", "ProjectedGradientDescent", "CoordinateDescent"]
