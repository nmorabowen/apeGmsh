# OpenSees bridge

apeGmsh's OpenSees deck is constructed via the explicit-constructor
pattern **after** the session closes:

```python
from apeGmsh.opensees import apeSees

fem = g.mesh.queries.get_fem_data(dim=3)
ops = apeSees(fem)
ops.model(ndm=3, ndf=6)
# … typed-primitive declarations, explicit fix / mass / patterns …
ops.tcl("model.tcl")     # or ops.py(...), ops.h5(...), ops.run()
```

The legacy ``g.opensees`` session composite and its sub-composites
(``materials`` / ``elements`` / ``ingest`` / ``inspect`` /
``export``) were removed in Phase 8 of the bridge teardown (ADR
0009).  `apeSees` has **no ingest and no auto-resolution** — loads,
masses, and SPs must be re-declared explicitly on ``ops``; MP
constraints are deferred (see the gold reference in
``skills/apegmsh/references/opensees-bridge.md``).

## Public surface

::: apeGmsh.opensees.apeSees

## Orientation helpers

Used as the ``orientation=`` argument on the typed geom_transf
primitives (``Linear`` / ``PDelta`` / ``Corotational``).

::: apeGmsh.opensees.Cartesian

::: apeGmsh.opensees.Cylindrical

::: apeGmsh.opensees.Spherical

## Recorders

Standalone recorder declaration helper. Recorder declarations live on
`ops.recorder.*` in the `apeSees` bridge.

<!-- TODO(apeSees migration): apeGmsh.opensees.recorder — verify new module path under apeGmsh.opensees after Phase 8.3b lands -->
::: apeGmsh.opensees.recorder

## Numberer

::: apeGmsh.mesh._numberer.Numberer

::: apeGmsh.mesh._numberer.NumberedMesh
