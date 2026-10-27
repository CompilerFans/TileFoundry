"""Analysis-only MACA targets.

The hardware documents reuse TileFoundry's CUDA schema because the facts an
analysis reads (topology, memory hierarchy, compute/service rates) are the same
shape; no CUDA codegen is claimed by this package.
"""

from .target import MacaTarget

__all__ = ["MacaTarget"]
