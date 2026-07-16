"""Errors and warnings for the section-properties analyzer (ADR 0078).

All are re-exported from :mod:`apeGmsh.sections`.
"""
from __future__ import annotations


class SectionMeshError(ValueError):
    """The ``FEMData`` handed to :class:`SectionProperties` fails an
    input gate — non-2-D elements, a non-planar mesh, or a physical-group
    material map that does not cover every element exactly once."""


class CompositeSectionError(ValueError):
    """An unprefixed accessor (``Ixx_c``, ``J``, ``Sxx``, …) was read on a
    multi-modulus (composite) section.  Rigidity-form fields (``EIxx_c``,
    ``GJ``, ``Mp_xx``) are always valid; for the classic numbers pick an
    explicit reference modulus via ``transformed(e_ref=...)``."""


class SectionAnalysisError(RuntimeError):
    """A section solve failed — e.g. the warping domain is disconnected
    under the default ``disconnected="raise"`` policy, or the plastic
    neutral-axis search could not bracket equilibrium."""


class SectionAccuracyWarning(UserWarning):
    """Linear elements (``tri3``/``quad4``) were used for an analysis that
    converges poorly on them (warping / stress).  Guidance: raise the mesh
    to second order with ``g.mesh.generation.set_order(2)``."""
