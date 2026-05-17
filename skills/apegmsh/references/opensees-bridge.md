# OpenSees bridge — `apeSees(fem)`

The legacy in-session `g.opensees.*` composite (and the
`apeGmsh.solvers` package) was **removed** in the Phase-8 teardown
(ADR 0009 — no back-compat shim). The OpenSees surface is now a
single class, constructed **after** the session from a `FEMData`
snapshot:

```python
from apeGmsh.opensees import apeSees

fem = g.mesh.queries.get_fem_data(dim=3)
ops = apeSees(fem)
ops.model(ndm=3, ndf=3)
```

Signatures are read from `src/apeGmsh/opensees/apesees.py`,
`opensees/_internal/ns/*.py`, `opensees/pattern/pattern.py`,
`opensees/element/*.py`, and `opensees/emitter/base.py`.

## The big behavioural change: no ingest

The old bridge had an `ingest` step that pulled session-declared
`g.loads` / `g.masses` / `g.constraints` into the OpenSees deck.
**The new bridge has no ingest and no auto-resolution.** ADR 0009
calls this "FEM-direct, explicit patterns".

- `apeSees` reads the `fem` snapshot only to resolve `pg=` /
  `label=` selectors to node/element tags and to get
  coordinates/connectivity.
- Loads, masses and SPs you want in OpenSees must be **re-declared
  explicitly** on `ops` (below).
- Session declarations still flow into the **`model.h5` neutral
  zone** (`ops.h5(path)`), so the **viewer / `Results`** see them
  — they are just not in the runnable Tcl/Py/Live deck.

## Lifecycle

```python
ops = apeSees(fem)                 # snapshot bound at construction
ops.model(ndm=3, ndf=3)            # must come first
# ... typed-primitive declarations + explicit fix/mass/patterns ...
ops.tcl("out/model.tcl")           # build() is implicit in emit
```

`ndm` / `ndf` the emitter understands:

| ndm | ndf | typical use |
|-----|-----|-------------|
| 2   | 2   | 2-D solid (ux, uy) |
| 2   | 3   | 2-D frame (ux, uy, θz) |
| 3   | 3   | 3-D solid (ux, uy, uz) |
| 3   | 6   | 3-D frame / shell (ux, uy, uz, θx, θy, θz) |

`apeSees(fem, default_orientation=Cartesian())` sets a model-wide
`geomTransf` orientation default (Z-up). Pass `None` for 2-D
models; pass a custom `Orientation` for a Y-up CAD import.

`ops.build()` returns an **immutable `BuiltModel`**; emitters
consume it. You rarely call it directly — `ops.tcl/py/h5/run`
build internally. It raises early with a pointed error if the
model is inconsistent (missing transf, ndm/ndf mismatch, …).

## Materials / sections / transforms — typed primitives

Constructors **return handles**; pass the handle by reference
(no string names, no registry). They auto-register on the bridge.

```python
conc  = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2, rho=2400)
steel = ops.uniaxialMaterial.Steel02(fy=420e6, E=200e9, b=0.01)

from apeGmsh.opensees.section.fiber import FiberPoint
sec   = ops.section.Fiber(
    fibers=(FiberPoint(material=steel, y=0.0, z=0.0, area=0.01),),
)
slab  = ops.section.ElasticMembranePlateSection(E=30e9, nu=0.2,
                                                h=0.2, rho=2400)
transf = ops.geomTransf.Linear(vecxz=(1, 0, 0))   # or .PDelta / .Corotational
integ  = ops.beamIntegration.Lobatto(section=sec, n_ip=5)
```

Which namespace:

- **`ops.nDMaterial`** → solid elements (`FourNodeTetrahedron`,
  `stdBrick`, `SSPbrick`, `quad`, `tri31`, `SSPquad`).
- **`ops.uniaxialMaterial`** → `truss`, `corotTruss`, `zeroLength`
  springs, and fiber-section beams.
- **`ops.section`** → shell elements
  (`ElasticMembranePlateSection`) and fiber sections for beams.

Every method has an explicit, fully-typed signature (no
`**kwargs`). Steel02 uses `fy=` (not the legacy `Fy=`).

## Elements

`ops.element.<Type>(pg="PG", ...)` writes every mesh element in
that physical group as `<Type>`. PG resolution is "FEM-direct"
against `fem`. Returns a typed `ElementGroup`.

```python
# Solid — nd material; gravity/body force is an ELEMENT parameter
ops.element.FourNodeTetrahedron(
    pg="Body", material=conc, body_force=(0.0, 0.0, -9.81 * 2400),
)
# 2-D solids also take a `pressure=` arg.

# Beam — transf + beamIntegration handles
ops.element.forceBeamColumn(pg="Cols", transf=transf, integration=integ)

# Shell — section handle
ops.element.ShellMITC4(pg="Deck", section=slab)
```

There is **no `eleLoad` pattern verb** — distributed/body loads
are element parameters (`body_force=`, `pressure=`), not loads.

## Boundary conditions, masses, loads — explicit

The `Emitter` protocol exposes exactly `node / fix / mass /
element / sp` for the model side. Re-declare on `ops`:

```python
# Homogeneous SP (fixities). dofs length = ndf.
ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))
# ops.fix(nodes=[...], dofs=(...))            # explicit-node form

# Lumped mass.
ops.mass(pg="Roof", values=(m, m, m, 0.0, 0.0, 0.0))

# Nodal loads + prescribed (non-zero) SP — pattern-scoped.
ts = ops.timeSeries.Linear()                  # also Constant/Path/Trig/Pulse
with ops.pattern.Plain(series=ts) as p:       # also UniformExcitation
    p.load(pg="Tip", forces=(0.0, 0.0, -5e4))
    p.sp(pg="LoadingPin", dof=3, value=0.01)  # prescribed displacement
```

`p.load` / `p.sp` fan a `pg=` across the group's nodes at build
time; `node=` takes an explicit tag or a `Node` from
`ops.nodes.get(...)`. Homogeneous SPs are model-level (`ops.fix`);
only non-zero prescribed values go inside a pattern via `p.sp`.

Migration of the old ingest call:

| Old `g.opensees.ingest.X(fem)` | New |
|---|---|
| `.loads(fem)` | `with ops.pattern.Plain(series=ts) as p: p.load(pg=..., forces=...)` |
| `.masses(fem)` | `ops.mass(pg=..., values=...)` |
| `.sp(fem)` (homogeneous `face_sp`) | `ops.fix(pg=..., dofs=...)` |
| `.sp(fem)` (prescribed `face_sp`) | `p.sp(pg=..., dof=..., value=...)` |
| `.constraints(fem, tie_penalty=)` | **deferred — no path** (see below) |

## ⚠️ Multi-point constraints are DEFERRED

There is **no OpenSees-emission path** for `tie`, `rigid_link`,
`equal_dof`, `rigid_diaphragm`, `node_to_surface`, `tied_contact`,
`mortar`, or embedded rebar. The `Emitter` protocol has no
`equalDOF`, `rigidLink`, `rigidDiaphragm`, or
`ASDEmbeddedNodeElement`. This is deferred by design (ADR 0009;
`src/apeGmsh/opensees/architecture/_DEFERRED.md`).

Consequences:

- Declare constraints on the session as usual — they resolve into
  `FEMData` and are persisted into the **`model.h5` neutral zone**
  by `ops.h5(path)`, so the **viewer / `Results`** render them.
- They are **not** written to any runnable Tcl/Py/Live deck. A
  model whose load path depends on a tie / rigid link will be
  wrong if you run the emitted deck as-is.
- To run such a model today, hand-emit the constraint commands by
  iterating `fem.nodes.constraints` / `fem.elements.constraints`
  into raw openseespy yourself (see `references/fem-broker.md`),
  or wait for the deferred feature.

## Emit / run / inspect

```python
ops.tcl("out/model.tcl")          # classic OpenSees deck
ops.py("out/model.py", run=False) # openseespy script (run=True subprocesses)
ops.h5("out/model.h5")            # native: bridge /opensees/ + broker neutral zone
ops.run()                         # in-process openseespy (LiveOpsEmitter)
ops.analyze(steps=10, dt=0.01)    # drive the analysis chain
```

These are **separate statements** — not a fluent chain. Each
`tcl/py/h5/run` calls `build()` internally.

Inspection is broker-side (`fem.inspect.summary()`,
`fem.inspect.node_table()`) or post-emit via the reference reader
`apeGmsh.opensees.emitter.h5_reader.open("model.h5")`.

## Canonical skeleton

```python
from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees

with apeGmsh(model_name="solid") as g:
    box = g.model.geometry.add_box(0, 0, 0, 10, 5, 2, label="body")
    g.physical.add(3, [box], name="Body")
    g.physical.add_surface([1], name="Base")

    g.mesh.sizing.set_global_size(0.5)
    g.mesh.generation.generate(dim=3)
    g.mesh.partitioning.renumber(dim=3, method="rcm", base=1)
    fem = g.mesh.queries.get_fem_data(dim=3)

# OpenSees — post-session, explicit declarations.
ops = apeSees(fem)
ops.model(ndm=3, ndf=3)
conc = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2, rho=2400)
ops.element.FourNodeTetrahedron(
    pg="Body", material=conc, body_force=(0.0, 0.0, -9.81 * 2400),
)
ops.fix(pg="Base", dofs=(1, 1, 1))
with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
    p.load(pg="Tip", forces=(0.0, 0.0, -5e4))

ops.tcl("out/model.tcl")
ops.py("out/model.py")
```

## When things go wrong

- **`apeSees.model(...) must be called before ...`** — call
  `ops.model(ndm=, ndf=)` first.
- **Loads / masses / constraints missing from the deck** — the
  bridge does not ingest `g.loads` / `g.masses` / `g.constraints`.
  Re-declare loads/masses/SP explicitly on `ops`; MP constraints
  are deferred (above).
- **Ambiguous `pg=`** — the same name exists at multiple
  dimensions. Keep PG names dimension-unique.
- **`len(dofs) != ndf`** — `ops.fix` needs a mask of length
  `ndf`, not the element's node count.
- **Tie/rigid-link model gives wrong results when run** — those
  constraints are not emitted (deferred); the runnable deck has no
  coupling.
