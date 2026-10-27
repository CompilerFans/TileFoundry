"""Python 3.10 compatibility shims.

The public source tree targets Python >=3.10.  ``StrEnum`` (3.11) and
``tomllib`` (3.11) are the only standard-library names the tree uses that the
older interpreter does not provide.
"""
from __future__ import annotations

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib

try:
    from enum import StrEnum
except ImportError:  # pragma: no cover - Python 3.10
    from enum import Enum

    class StrEnum(str, Enum):
        """The Python 3.11 ``StrEnum`` surface used by this tree."""

        def __str__(self) -> str:
            return self.value

        def __format__(self, spec: str) -> str:
            return format(str(self), spec)

def tuple_annotation(item):
    """A ``tuple[item]`` annotation that accepts a non-type (a Pattern).

    Python 3.12's ``typing.Tuple[pattern]`` stores the object as-is; 3.10's
    ``typing.Tuple.__getitem__`` rejects it.  ``types.GenericAlias`` has the
    same ``get_origin``/``get_args`` surface and does not validate.
    """
    from types import GenericAlias

    return GenericAlias(tuple, (item,))

__all__ = ["StrEnum", "tomllib", "tuple_annotation"]
