"""MACA target composition: CUDA fact machinery, MACA hardware documents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from tilefoundry.target.base import HardwareSpec, register_target
from tilefoundry.utils.python_source import PythonExpr
from tilefoundry_nvidia.cuda.target.spec import (
    ARCHITECTURE_SCHEMA,
    DEVICE_SCHEMA,
    build_cuda_architecture,
    build_cuda_device,
)
from tilefoundry_nvidia.cuda.target.target import CudaTarget


@register_target
@dataclass(frozen=True, init=False)
class MacaTarget(CudaTarget):
    """A MACA device analysed through TileFoundry's generic hardware facts.

    Analysis-only: this class inherits the CUDA target's fact projections and
    topology vocabulary (gpu/cta/thread), but its documents describe MACA APs,
    wave width, shared memory and memory/compute rates.  It must not be used
    for CUDA code generation.
    """

    name: ClassVar[str] = "maca"
    hardware: ClassVar[HardwareSpec] = HardwareSpec(
        package="tilefoundry_maca.hardware",
        schemas={
            ARCHITECTURE_SCHEMA: build_cuda_architecture,
            DEVICE_SCHEMA: build_cuda_device,
        },
    )

    def __repr__(self) -> str:
        return f'MacaTarget("{self.identity}")'

    def to_python(self) -> PythonExpr:
        return PythonExpr(
            ("from tilefoundry_maca import MacaTarget",),
            f'MacaTarget("{self.identity}")',
        )


__all__ = ["MacaTarget"]
