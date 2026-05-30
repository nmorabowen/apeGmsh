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
``export``) were removed in Phase 8 of the bridge teardown
([ADR 0009](https://github.com/nmorabowen/apeGmsh/blob/main/src/apeGmsh/opensees/architecture/decisions/0009-no-backwards-compat-with-solvers.md)).
`apeSees` does **selective ingest**: **loads** declared via
``g.loads.*`` (which resolve onto ``fem.nodes.loads``) **emit
automatically** as synthesized ``Plain`` patterns — no re-declaration
needed — while **masses** and **support fixities / SPs** must be
re-declared explicitly on ``ops``
(``ops.fix(pg=, dofs=)``, ``ops.mass(pg=, values=)``).

> **Don't double-declare a load.** If a load is already declared
> via ``g.loads.*`` it reaches the solver on its own; re-declaring
> the same load on the bridge (``ops.pattern.Plain(...) as p:
> p.load(pg=, forces=)``) **doubles** it (reactions come out at 2×).
> Pick one channel per load — either the broker (``g.loads``) or a
> bridge pattern, not both.

Since the teardown, the bridge has been progressively widened:

- **Loads emit automatically** from ``fem.nodes.loads``. Loads
  declared via ``g.loads.*`` are synthesized into ``Plain``
  patterns by the broker-load emitter and land in the runnable
  Tcl/Py deck (and the live/run path) without an ``ingest`` step —
  purely additive on top of any bridge-registered pattern
  primitives. (See the double-declaration caveat above.)
- **MP constraints emit automatically** from ``fem.nodes.constraints``
  / ``fem.elements.constraints``
  ([ADR 0022](https://github.com/nmorabowen/apeGmsh/blob/main/src/apeGmsh/opensees/architecture/decisions/0022-mp-constraint-emission-fanout.md),
  Phase 7b) — ``equalDOF`` / ``rigidLink`` / ``rigidDiaphragm`` /
  ``ASDEmbeddedNodeElement`` lines land in the runnable Tcl/Py
  deck without an ``ingest`` step. The ``apeSees.h5(path)`` write
  target persists per-record details under
  ``/opensees/constraints/`` (additive minor schema bump 2.6.0 →
  2.7.0). Auto-emits a ``Transformation`` constraint handler when
  MP constraints are present and the user has not declared one.
- **Staged analysis** ships via ``ops.stage(name)`` —
  see [Staged analysis](#staged-analysis) below for the user-
  surface walkthrough, and the in-repo internals doc at
  [staged-analysis.md](https://github.com/nmorabowen/apeGmsh/blob/main/src/apeGmsh/opensees/architecture/staged-analysis.md)
  for the per-stage emit pipeline.
- **Read-side broker.**
  ``OpenSeesModel.from_h5(path, fem_root=)`` provides a frozen
  read-only view of the persisted ``/opensees/`` zone with the
  embedded ``FEMData`` lazily attached
  ([ADR 0019](https://github.com/nmorabowen/apeGmsh/blob/main/src/apeGmsh/opensees/architecture/decisions/0019-opensees-model-read-side-broker.md)).
  Re-emit via ``om.build("tcl", path)`` / ``om.build("py", path)``
  / ``om.build("live")`` without rehydrating the apeSees
  primitives.

For the full user-facing surface (typed materials, sections,
elements, recorders, patterns, analysis chain, staged analysis,
SSI helpers, cuts and sweeps), see the in-repo
[api-design.md](https://github.com/nmorabowen/apeGmsh/blob/main/src/apeGmsh/opensees/architecture/api-design.md).

## Public surface

::: apeGmsh.opensees.apeSees

## Staged analysis

Multi-stage workflows (in-situ stress install → excavate →
lining install → dynamic shake, or any other sequence of
``analyze`` blocks with Domain mutations between them) use
the ``ops.stage(name)`` context manager:

```python
with ops.stage(name="excavate") as s:
    s.activate(pgs=["Lining"])               # bring new elements online
    s.fix(pg="LiningAnchor", dofs=(1, 1, 1)) # stage-bound BC
    s.embedded(name="lining_embed")          # claim MP constraint by name
    s.analysis(test=…, algorithm=…, integrator=…,
               constraints=…, numberer=…, system=…, analysis=…)
    s.run(n_increments=20, dt=0.05)
```

Each stage emits its own analysis chain + analyze loop with an
explicit inter-stage cleanup (``loadConst -time 0.0`` +
``wipeAnalysis``). Between-stage Domain mutators (``s.remove_sp``
/ ``s.remove_element`` / ``s.set_time`` / ``s.set_creep`` /
``s.reset`` / ``s.mass(overwrite=True)``) lift the append-only
constraint from earlier phases and unlock the atomic-replace
pattern (release prior support + re-fix the same DOF in one
stage).

Five validators gate stage-bound BCs at build time (H1 / V1 / V2
/ V3 / V4) and two more cover the SSI-2.E removal verbs (V5 /
V6). Each raises ``BridgeError`` with a clear offender list when
a stage references topology that doesn't yet exist or has
already been removed.

Tcl + Py text emit are the supported execution paths for staged
decks today. Live execution (``ops.analyze`` / ``ops.eigen``)
refuses staged models with ``NotImplementedError`` — emit via
``ops.tcl(p, run=True)`` / ``ops.py(p, run=True)`` for the
OpenSees subprocess. H5 archival of staged structure is also
deferred (``apeSees.h5(path)`` is fail-loud on a staged build per
PR [#313](https://github.com/nmorabowen/apeGmsh/pull/313)).

The full lifecycle table, builder verbs, validator surface, MP
partitioned + staged emit (Phase SSI-2.C), and the SSI-1
initial-stress ramp live in
[architecture/api-design.md](https://github.com/nmorabowen/apeGmsh/blob/main/src/apeGmsh/opensees/architecture/api-design.md)
§"Staged analysis"; the internals (deck layout, ownership
computation, hook dispatcher, per-emitter dialect divergence,
cleanup contract) live in
[architecture/staged-analysis.md](https://github.com/nmorabowen/apeGmsh/blob/main/src/apeGmsh/opensees/architecture/staged-analysis.md)
and
[architecture/emitter.md](https://github.com/nmorabowen/apeGmsh/blob/main/src/apeGmsh/opensees/architecture/emitter.md).

## Orientation helpers

Used as the ``orientation=`` argument on the typed geom_transf
primitives (``Linear`` / ``PDelta`` / ``Corotational``).

::: apeGmsh.opensees.Cartesian

::: apeGmsh.opensees.Cylindrical

::: apeGmsh.opensees.Spherical

## Recorders

Standalone recorder declaration helper. Recorder declarations live on
`ops.recorder.*` in the `apeSees` bridge (typed primitives —
``Node`` / ``Element`` / ``MPCO`` / declarative fan-out via
``ops.recorder.declare(...)``).

::: apeGmsh.opensees.recorder

## Numberer

::: apeGmsh.mesh._numberer.Numberer

::: apeGmsh.mesh._numberer.NumberedMesh
