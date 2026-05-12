# OpenSees bridge

apeGmsh's OpenSees deck is constructed via the explicit-constructor
pattern:

```python
fem = g.mesh.queries.get_fem_data(dim=3)
from apeGmsh.opensees import apeSees
ops = apeSees(fem)
ops.model(ndm=3, ndf=6)
# … primitives, patterns, recorders, analysis chain …
ops.tcl("model.tcl")     # or ops.py(...), ops.h5(...), ops.run()
```

The legacy ``g.opensees`` session attribute and its
``OpenSees``-class sub-composites (``materials`` / ``elements`` /
``ingest`` / ``inspect`` / ``export``) were removed in Phase 8 of
the bridge teardown.  Migration patterns live in the EOS
curriculum notebooks under ``examples/EOS Examples/curriculum/``.

## Public surface

::: apeGmsh.opensees.apeSees

## Coordinate-system helpers

Used as the ``csys=`` argument on the typed geom_transf primitives
(``Linear`` / ``PDelta`` / ``Corotational``).

::: apeGmsh.opensees.Cartesian

::: apeGmsh.opensees.Cylindrical

::: apeGmsh.opensees.Spherical

## Recorders

Standalone recorder declaration helper (canonical home moves in
Phase 8.3b).

::: apeGmsh.solvers.Recorders.Recorders

## Numberer

::: apeGmsh.mesh._numberer.Numberer

::: apeGmsh.mesh._numberer.NumberedMesh
