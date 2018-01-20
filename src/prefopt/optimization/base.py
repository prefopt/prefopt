"""
Optimizer for acquisition function.
"""

__all__ = ['OptimizationError']


class OptimizationError(RuntimeError):
    """Optimization did not terminate successfully."""
