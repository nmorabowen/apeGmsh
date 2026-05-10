# `apeGmsh.opensees` — Architecture

This folder is the design charter for the OpenSees bridge that replaces
`apeGmsh.solvers`. It captures the principles, the layout, the API
shape, and the rationale behind each major decision so future work
extends the same skeleton instead of inventing a new one beside it.

## How to read these docs

Read in this order on first pass:

1. **[charter.md](charter.md)** — mission, principles, non-goals.
   The 14 principles are the rules we judge new code against.
2. **[layout.md](layout.md)** — folder structure, naming conventions,
   where each kind of class lives.
3. **[api-design.md](api-design.md)** — the user-facing surface:
   namespace API, static typing, typed instances with capabilities,
   Node aggregator pattern.
4. **[emitter.md](emitter.md)** — the Tcl/py/live abstraction. How one
   model definition produces three execution targets.
5. **[patterns-and-loads.md](patterns-and-loads.md)** — the
   pattern-explicit decision and its grounding in the OpenSees source.
6. **[h5-schema.md](h5-schema.md)** — the bridge enrichment HDF5
   format spec. Authoritative for what `ops.h5(path)` produces.
7. **[viewer-integration.md](viewer-integration.md)** — contract for
   the viewer team consuming the H5. Required reads, optional UI
   features per group, performance budget, fixtures.

For the implementation phase (multi-agent parallel execution):

8. **[testing.md](testing.md)** — test architecture, the seven
   test layers, conventions every agent follows, what every PR
   adds.
9. **[parallel-execution.md](parallel-execution.md)** — work
   breakdown and dependency graph. Identifies what can run in
   parallel and the sync points where slices come together.
10. **[agent-onboarding.md](agent-onboarding.md)** — prompt
    templates the coordinator uses to spin up slice agents.
    Common pitfalls and the "do not invent" rule.

For specific decisions, see the ADRs in
[decisions/](decisions/) — one file per major call, with the
context, the decision, the alternatives we rejected, and the
consequences.

For things we agreed to defer, see
[_DEFERRED.md](_DEFERRED.md).

## Status

**Phase: design.** No primitive code, no `apeSees` class, no emitters
yet. The legacy `apeGmsh.solvers` package remains the production path
until skeletons land in this package and migration begins.

## TL;DR

```python
from apeGmsh.opensees import apeSees
from apeGmsh.opensees.transform import Cartesian

fem = g.mesh.queries.get_fem_data(dim=1)
ops = apeSees(fem)
ops.model(ndm=3, ndf=6)

steel = ops.uniaxialMaterial.Steel02(fy=420e6, E=200e9, b=0.01)
core  = ops.uniaxialMaterial.Concrete02(
    fpc=-30e6, epsc0=-0.002, fpcu=-25e6, epsu=-0.006,
    lambda_val=0.1, ft=2.5e6, Ets=200e6,
)
sec   = ops.section.Fiber(patches=[...], fibers=[...], GJ=1e9)
trans = ops.geomTransf.PDelta(csys=Cartesian())

ops.element.forceBeamColumn(pg="Cols", section=sec, transf=trans, n_ip=5)
ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))

with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
    p.load(pg="RoofNode", forces=(100e3, 0, 0))

ops.constraints.Transformation()
ops.numberer.RCM()
ops.system.BandGeneral()
ops.test.NormDispIncr(tol=1e-6, max_iter=10)
ops.algorithm.Newton()
ops.integrator.LoadControl(increment=0.05)
ops.analysis.Static()
ops.analyze(steps=20)

ops.tcl("frame.tcl")          # write
ops.py("frame.py", run=True)  # write + invoke python
ops.run()                     # live via openseespy
```

The deck reads as OpenSees Tcl, with full static typing, named
parameters, automatic tag allocation, and physical-group integration.
