"""In-process domain capture (Strategy B).

Drives the openseespy domain via ``ops.nodeDisp(...)``,
``ops.nodeEigenvector(...)`` etc., writing native HDF5 directly.

See :class:`DomainCapture` for usage.
"""
from ._domain import DomainCapture

__all__ = ["DomainCapture"]
