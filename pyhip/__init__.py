"""PyHIP: Python AMDGPU kernel development toolkit (minimal subset for EvoKernel)"""

from .core.hiptools import module
from .misc import calc_diff

__all__ = ["module", "calc_diff"]
