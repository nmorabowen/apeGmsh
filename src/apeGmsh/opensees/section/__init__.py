"""
``apeGmsh.opensees.section`` — typed primitives for OpenSees
``section <Type>`` commands.

Per ADR 0004, sections live at the top level of the package (not
under ``material/``). The module is split by section family:

* :mod:`.beam`  — :class:`ElasticSection` (linear-elastic beam line)
* :mod:`.computed` — :class:`ComputedSection` (lazy ``section
  Elastic`` lowered from a :class:`~apeGmsh.sections.SectionProperties`
  analyzer, ADR 0078)
* :mod:`.plate` — :class:`ElasticMembranePlateSection`,
  :class:`LayeredShell`, :class:`LayeredShellFiberSection`
* :mod:`.fiber` — :class:`Fiber`, plus value objects
  :class:`RectPatch`, :class:`StraightLayer`, :class:`FiberPoint`

Constructing these classes outside a bridge is supported (P11). The
typed methods that auto-register them with an :class:`apeSees`
bridge live on :class:`apeGmsh.opensees._internal.ns.section._SectionNS`
and are exposed via ``ops.section.<Type>(...)``.
"""
from __future__ import annotations

from .aggregator import AGGREGATOR_DOF_CODES, Aggregator
from .beam import ElasticSection
from .computed import ComputedSection
from .fiber import Fiber, FiberPoint, RectPatch, StraightLayer, W_fiber
from .plate import (
    ElasticMembranePlateSection,
    LayeredShell,
    LayeredShellFiberSection,
    ShellLayer,
)


__all__ = [
    # beam
    "ElasticSection",
    # computed (ADR 0078 — lazy analyzer lowering)
    "ComputedSection",
    # plate
    "ElasticMembranePlateSection",
    "LayeredShell",
    "LayeredShellFiberSection",
    "ShellLayer",
    # fiber
    "Fiber",
    "FiberPoint",
    "RectPatch",
    "StraightLayer",
    "W_fiber",
    # aggregator (composes other sections + uniaxials)
    "Aggregator",
    "AGGREGATOR_DOF_CODES",
]
