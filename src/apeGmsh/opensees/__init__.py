"""
apeGmsh.opensees — Pythonic, statically-typed wrapper around OpenSees.

Public API: :class:`apeSees`. See ``architecture/README.md`` for the
design charter, the 14 principles, and the per-phase implementation
roadmap.

Example:

.. code-block:: python

    from apeGmsh.opensees import apeSees

    fem = g.mesh.queries.get_fem_data(dim=1)
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=6)
    steel = ops.uniaxialMaterial.Steel02(fy=420e6, E=200e9, b=0.01)
    ...
    ops.tcl("frame.tcl")
    ops.run()

The package coexists with the legacy :mod:`apeGmsh.solvers` during
the migration period (ADR 0009 — no back-compat shim).
"""
from __future__ import annotations

from .apesees import apeSees

__all__ = ["apeSees"]
