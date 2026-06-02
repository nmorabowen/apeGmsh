"""Typed ``damping <Type>`` objects (ADR 0053).

Frequency-band viscous damping objects, attached to elements via a
``region -damp`` line.  See :mod:`apeGmsh.opensees.damping.damping`.
"""
from __future__ import annotations

from .damping import SecStif, Uniform

__all__ = ["Uniform", "SecStif"]
