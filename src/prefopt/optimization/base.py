"""
Optimizer for acquisition function.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__all__ = ['OptimizationError']


class OptimizationError(RuntimeError):
    """Optimization did not terminate successfully."""
