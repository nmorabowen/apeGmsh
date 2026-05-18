"""apeGmsh._kernel.resolvers — defs→records resolution layer.

Relocated verbatim from ``apeGmsh.mesh._{load,mass,constraint}_resolver``
and ``apeGmsh.mesh._constraint_resolver`` (plan P1-K).  Each resolver
translates the pure :mod:`apeGmsh._kernel.defs` intents into the
resolved :mod:`apeGmsh._kernel.records` dataclasses by attaching mesh
data (node tags, coordinates, connectivity).

:mod:`apeGmsh._kernel.resolvers._mass_resolver` is the sole module
permitted to import ``apeGmsh.fem`` (leaf-pure HRZ / shape-function
helpers — plan HT10).  No other ``apeGmsh.*`` package is imported.
"""

from __future__ import annotations
