# OpenSees bridge — `apeSees(fem)`

The OpenSees surface is a single class, constructed **after** the
session from a `FEMData` snapshot. The legacy in-session
`g.opensees.*` composite was **removed** — there is no back-compat
shim.

```python
from apeGmsh.opensees import apeSees   # verified: tests/opensees/unit/test_apesees_class.py::test_apesees_constructs_with_fem

fem = g.mesh.queries.get_fem_data(dim=3)
ops = apeSees(fem)                      # snapshot bound at construction
ops.model(ndm=3, ndf=3)                # must come first
```

Signatures read from `src/apeGmsh/opensees/apesees.py`,
`opensees/_internal/ns/*.py`, and `opensees/emitter/base.py`.

## What the bridge auto-emits vs what you re-declare

The bridge reads the `fem` snapshot to resolve `pg=` / `label=`
selectors and to get coords/connectivity. Two different policies
apply to what ends up in the runnable deck:

| Surface | Auto-emitted from `fem`? |
|---|---|
| **MP constraints** (`fem.nodes.constraints` / `fem.elements.constraints`) | **YES — auto-emit** (ADR 0022, shipped v2.0.0) |
| **Loads** (`fem.nodes.loads`, declared via `g.loads.*`) | **YES — auto-emit** as synthesized Plain patterns (`_emit_broker_loads`, additive on top of bridge patterns) |
| Masses, homogeneous SPs (fixities) | **NO — re-declare explicitly on `ops`** |

So the bridge does **selective ingest**: nodal/element loads you
declared on the *session* (`g.loads.*`) **are** pulled into the deck
automatically (as Plain patterns, purely additive on top of any
bridge-registered patterns), as are multi-point constraints (equalDOF,
rigid link, rigid diaphragm, embedded). But lumped masses and fixities
you declared on the session (`g.masses` / `g.constraints` SPs) are
**not** ingested — re-declare those on `ops` (§ Boundary conditions,
masses, loads).

> **Don't double-declare loads.** A load declared via `g.loads.*` is
> already emitted by the bridge. If you *also* re-declare that same
> load on a bridge pattern (`p.load(...)`), it lands **twice** —
> verified: reactions come out at exactly 2×. Pick **one** channel
> per load: either `g.loads.*` (session) **or** `p.load` (bridge),
> never both.

Session declarations all still flow into the **`model.h5` neutral
zone** (`ops.h5(path)`), so the **viewer / `Results`** see everything
regardless.

## Lifecycle

```python
ops = apeSees(fem)
ops.model(ndm=3, ndf=3)            # must be called first
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

`ops.build()` returns an **immutable `BuiltModel`** and **requires
`ops.model(...)` first** — it is usually implicit because
`ops.tcl/py/h5/run` build internally. It raises early
(`BridgeError`) on an inconsistent model (missing transf, ndm/ndf
mismatch, …).
<!-- verified: tests/opensees/unit/test_apesees_class.py::test_apesees_build_requires_model_first -->

## Materials / sections / transforms — typed primitives

Constructors **return handles**; pass the handle by reference (no
string names, no registry). They auto-register on the bridge.

```python
conc  = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2, rho=2400)   # verified: tests/opensees/unit/test_apesees_class.py::test_apesees_namespace_is_present
steel = ops.uniaxialMaterial.Steel02(fy=420e6, E=200e9, b=0.01)     # fy= not Fy=

from apeGmsh.opensees.section.fiber import FiberPoint
sec    = ops.section.Fiber(
    fibers=(FiberPoint(material=steel, y=0.0, z=0.0, area=0.01),),
)
slab   = ops.section.ElasticMembranePlateSection(E=30e9, nu=0.2, h=0.2, rho=2400)
transf = ops.geomTransf.Linear(vecxz=(1, 0, 0))    # or .PDelta / .Corotational
integ  = ops.beamIntegration.Lobatto(section=sec, n_ip=5)
```

Which namespace:

- **`ops.nDMaterial`** → solid elements (`FourNodeTetrahedron`,
  `stdBrick`, `SSPbrick`, `quad`, `tri31`, `SSPquad`).
- **`ops.uniaxialMaterial`** → `truss`, `corotTruss`, `zeroLength`
  springs, and fiber-section beams.
- **`ops.section`** → shell elements (`ElasticMembranePlateSection`)
  and fiber sections for beams.

Every method has an explicit, fully-typed signature (no `**kwargs`).

## Elements

`ops.element.<Type>(pg="PG", ...)` writes every mesh element in that
physical group as `<Type>`. PG resolution is FEM-direct against
`fem`. Returns a typed `ElementGroup`.

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

There is **no `eleLoad` pattern verb** — distributed/body loads are
element parameters (`body_force=`, `pressure=`), not loads.

## Boundary conditions, masses, loads — explicit

The `Emitter` protocol exposes `node / fix / mass / element / sp`
for the model side. **Masses and homogeneous SPs (fixities)** are
**not** auto-pulled from the session — re-declare those on `ops`.
**Loads**, by contrast, *are* auto-emitted from `g.loads.*`; the
pattern verbs below are the **bridge channel** for loads you'd
rather declare here — don't declare the same load on both channels
(it doubles):

```python
# Homogeneous SP (fixities). dofs length = ndf.
ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))     # verified: tests/opensees/unit/test_emitter_protocol.py::test_fix_records_tag_and_dofs
# ops.fix(nodes=[...], dofs=(...))              # explicit-node form

# Lumped mass.
ops.mass(pg="Roof", values=(m, m, m, 0.0, 0.0, 0.0))

# Nodal loads + prescribed (non-zero) SP — pattern-scoped.
ts = ops.timeSeries.Linear()                    # also Constant/Path/Trig/Pulse
with ops.pattern.Plain(series=ts) as p:         # also UniformExcitation   # verified: tests/opensees/unit/test_emitter_protocol.py::test_pattern_open_close_pair
    p.load(pg="Tip", forces=(0.0, 0.0, -5e4))
    p.sp(pg="LoadingPin", dof=3, value=0.01)    # prescribed displacement
```

`p.load` / `p.sp` fan a `pg=` across the group's nodes at build
time; `node=` takes an explicit tag or a `Node` from
`ops.nodes.get(...)`. Homogeneous SPs are model-level (`ops.fix`);
only non-zero prescribed values go inside a pattern via `p.sp`.

## ✅ Multi-point constraints ARE emitted (ADR 0022)

> This reverses the old skill claim that MP emission is "deferred /
> the Emitter protocol has no MPC verbs." It shipped in **v2.0.0**.

Declare MP constraints on the session as usual (`g.constraints.*`);
they resolve into `fem.nodes.constraints` / `fem.elements.constraints`,
and `apeSees(fem)` **auto-emits** them into the runnable Tcl/Py/Live
deck. The `Emitter` protocol now defines the verbs
(`opensees/emitter/base.py:153`):

| Session constraint | Emitter verb | OpenSees command |
|---|---|---|
| `g.constraints.equal_dof(...)` | `equalDOF(master, slave, *dofs)` | `equalDOF` |
| `g.constraints.rigid_link(...)` (`rigid_beam`/`rigid_rod`) | `rigidLink(kind, master, slave)` | `rigidLink` |
| `g.constraints.rigid_diaphragm(...)` | `rigidDiaphragm(perp_dir, master, *slaves)` | `rigidDiaphragm` |
| `g.constraints.embedded(...)` / `tied_contact(...)` | `embeddedNode(ele_tag, cnode, *masters, stiffness=, stiffness_p=, rotational=, pressure=)` | `ASDEmbeddedNodeElement` |

<!-- verified: tests/opensees/unit/test_emitter_protocol.py::test_equalDOF_records_master_slave_dofs, ::test_rigidLink_records_kind_master_slave, ::test_rigidDiaphragm_records_perp_master_slaves, ::test_embeddedNode_records_ele_tag_cnode_args -->

Emission order (INV-3/INV-5, `build.emit_mp_constraints`): phantom
nodes (synthesized by `NodeToSurfaceRecord` ties, `node(..., ndf=6)`)
emit first, then MP constraints, **after** element emission and
**before** pattern emission, so every referenced node already exists.

**Auto-Transformation handler.** When MP constraints are present and
you did **not** declare a constraints handler, the bridge auto-emits
`constraints("Transformation")` (with a one-time warning;
`apesees.py:3434`). A user-declared `Plain` handler with MP present
warns ("did you mean Transformation/Lagrange/Penalty?"); `Penalty`
/ `Transformation` / `Lagrange` pass silently.

The `ASDEmbeddedNodeElement` options (`stiffness`, `stiffness_p`,
`rotational`, `pressure`) are exposed on
`g.constraints.embedded/tie/tied_contact(...)` (ADR 0035, opensees
schema 2.12.0).

## Per-node ndf — shell-on-solid (`g.node_ndf`, ADR 0032/0033)

apeGmsh **never infers** ndf from element class. For mixed-ndf
models (e.g. shells coupled to solids) you declare per-node DOF
counts on the session **before** `get_fem_data()`:

```python
g.node_ndf.set_default(ndf=3)          # solids default to 3 DOF
g.node_ndf.set("Top", ndf=6)           # shell-coupled face gets 6 DOF (last-wins)

fem = g.mesh.queries.get_fem_data(dim=3)   # _ndf vector resolved here
fem.nodes.ndf_for(some_top_nid)            # -> 6
```
<!-- verified: tests/test_node_ndf.py::test_single_region_override_plus_default -->

Surface: `g.node_ndf.set(target, *, ndf, name=None)`,
`.set_default(*, ndf, name=None)`, `.list()`, `.clear()` (composite
at `core/NodeNDFComposite.py`). `ndf` must be an int in `[1, 6]`
(out of range → `ValueError`; non-int / `bool` → `TypeError`).

**Fail-loud `ndf_for`.** `fem.nodes.ndf_for(nid)` raises `LookupError`
(message naming both `set` and `set_default` fixes) when the node
exists but is undeclared (sentinel 0, or `_ndf is None`); raises
`KeyError` when `nid` is not a known node id. Targeted defs apply in
declaration order (last wins on overlap); `set_default` fills only
slots still at the sentinel after targeted defs resolve.
<!-- verified: tests/test_node_ndf.py::test_ndf_for_undeclared_raises_helpful_lookuperror -->

**Envelope validator.** `ops.model(ndm=, ndf=K)` raises `BridgeError`
at call time if `K < max(declared per-node ndf)`. The envelope `ndf`
**wins** on any node without a declaration — those nodes emit with
**no** per-node `-ndf` token (byte-identical backcompat). Only
declared nodes get an explicit `-ndf K`.
<!-- verified: tests/test_node_ndf.py::test_validator_at_apesees_model_raises_bridgeerror, ::test_emit_mixed_ndf_shell_on_solid_flat -->

> Mutating `g.node_ndf` *after* a `get_fem_data()` build emits a
> `UserWarning` (cached broker won't see it — re-run `get_fem_data()`).
> `from_msh` models have no composite → `_ndf is None` → every node
> falls back to the envelope.

## Staged analysis — `ops.stage(name)` (ADR 0028/0029/0030/0034)

Open a stage as a **context manager**. Stages accumulate by
registration order; they do **not** nest (a second open while one is
live raises `RuntimeError`). Every stage MUST call `s.analysis(...)`
(all 7 chain kwargs) **and** `s.run(...)` before the `with` exits, or
`__exit__` raises `ValueError`.

```python
ops = apeSees(fem)
ops.model(ndm=2, ndf=2)
# ... materials / elements / transforms ...
ops.fix(pg="Base", dofs=(1, 1))        # global (pre-stage) BCs use the flat verbs

with ops.stage(name="insitu") as s:
    s.initial_stress(name="rock", pg="Rock",                 # PUSH
                     sigma_xx=-100.0, sigma_yy=-200.0, sigma_zz=-150.0,
                     ramp_steps=10)
    s.analysis(test=ops.test.NormDispIncr(tol=1e-4, max_iter=150),
               algorithm=ops.algorithm.Newton(),
               integrator=ops.integrator.LoadControl(dlam=0.1),
               constraints=ops.constraints.Plain(),
               numberer=ops.numberer.RCM(),
               system=ops.system.UmfPack(),
               analysis=ops.analysis.Static())
    s.run(n_increments=10, dt=0.1)

with ops.stage(name="install_lining") as s:
    s.activate(pgs=["Lining"])              # bring elements online
    s.fix(pg="LiningAnchor", dofs=(1, 1))   # stage-bound BC (PUSH)
    s.embedded(name="lining_embed")         # CLAIM MP constraint by name
    s.analysis(test=..., algorithm=..., integrator=...,
               constraints=..., numberer=..., system=..., analysis=...)
    s.run(n_increments=20, dt=0.05)

ops.tcl("model.tcl", run=True)   # staged decks emit via Tcl/Py text ONLY
```
<!-- verified: tests/opensees/unit/test_stages.py::test_stage_builder_records_complete_stage, tests/opensees/unit/test_stages.py::test_stage_builder_missing_analysis_raises -->

### Stage verbs — PUSH vs PULL vs CLAIM

The three semantics are distinct (don't confuse them):

- **PUSH** (creates an inert record on the stage):
  `s.fix(*, pg=|nodes=, dofs)`, `s.mass(*, pg=|nodes=, values, overwrite=False)`,
  `s.region(*, name, pg=|nodes=)`, `s.initial_stress(*, name, pg=|elements=, sigma_xx, sigma_yy, sigma_zz, ramp_steps, lambda_install=1.0)`.
- **PULL** (record/spec must already be globally registered):
  `s.recorder(spec)` (spec from `ops.recorder.X`),
  `s.add(record)` (`InitialStressRecord` only).
- **CLAIM** (constraint declared at apeGmsh time via
  `g.constraints.X(..., name=...)`, claimed by name from the broker):
  `s.embedded(name=)`, `s.tie(name=)`, `s.distributing(name=)`,
  `s.equal_dof(name=)`, `s.rigid_link(name=)`,
  `s.rigid_diaphragm(name=)`, `s.kinematic_coupling(name=)`,
  `s.node_to_surface(name=)`, `s.node_to_surface_spring(name=)`. A
  typo / missing `name=` → `ValueError`; double-claim across stages
  → `ValueError`.

<!-- verified: tests/opensees/unit/test_stage_bound_fix_mass.py::test_s_fix_populates_stage_record_fix_records, tests/opensees/unit/test_stage_embedded_claim.py::test_embedded_claim_populates_stage_pool, tests/opensees/unit/test_stage_initial_stress_push.py -->

Between-stage Domain mutators (SSI-2.E): `s.remove_sp(*, pg=|nodes=, dofs)`,
`s.remove_element(*, pg=|elements=)`, `s.set_time(t)`,
`s.set_creep(on)`, `s.reset()`.

### Stage gotchas

- **`s.remove_sp` `dofs=` are 1-based DOF INDICES** (one `remove sp
  $node $dof` line each), **not** the 0/1 fixity-flag vector that
  `s.fix` / `s.mass` use. Same kwarg name, different meaning.
- `s.remove_element` `elements=` are **FEM eids** (recorder.Element
  convention), not OpenSees ops tags — the bridge translates at emit.
- **`s.tied_contact` and `s.mortar` are intentionally NOT stage
  verbs** (`apesees.py:5228`); tied_contact's slaves can't be
  suppressed by the global exclusion filter, mortar is not
  kernel-implemented.
- **Live execution refuses staged models.** `ops.analyze()` and
  `ops.eigen()` raise `NotImplementedError` when any stage is
  registered. Only `ops.tcl(path, run=)` / `ops.py(path, run=)` drive
  a staged deck.
- **H5 archival refuses staged models** — `apeSees(fem).h5()` raises
  on a staged build (`apesees.py:4665`); `split='parts'` also refuses.
- `s.mass` re-applying mass to a node already massed in another tier
  raises (validator V2) unless you pass `overwrite=True` to ack it.
  Same region `name=` across scopes raises (V3).

## Recorders — three declaration surfaces

1. **Typed primitives** — `ops.recorder.Node(...)` /
   `ops.recorder.Element(...)` / `ops.recorder.MPCO(...)`.
2. **`ops.recorder.declare(...)`** — canonical fan-out, per-category
   kwargs (`nodes=` / `elements=` / `line_stations=` / `gauss=`) plus
   a `raw=` escape hatch.
3. **`DomainCaptureSpec`** (`apeGmsh.results.capture.spec`) for
   in-process capture: `ops.domain_capture(spec, path=)`.

The canonical component vocabulary is the neutral top-level
`apeGmsh._vocabulary` (`expand_shorthand` / `expand_many`). The old
`Recorders` fluent helper was deleted (legacy paths raise
`ImportError`).

## Native model.h5 — two zones + schema constants

`ops.h5(path, *, model_name=None, cuts=(), sweeps=())` writes a
**two-zone** `model.h5`: the *neutral* zone (broker-written —
nodes/elements/PGs/labels/constraints/loads/masses, same as
`g.save()`) **and** the `/opensees/` zone (bridge-written —
transforms, recorders, cuts/sweeps). `g.save()` / `FEMData.to_h5`
write **only** the neutral zone; `apeSees(fem).h5(path)` writes both.

Two **independent** per-zone schema constants (ADR 0023):

- bridge `SCHEMA_VERSION` (`opensees/emitter/h5.py:252`) = **2.12.0**
  — stamps `/opensees/…`.
- broker `NEUTRAL_SCHEMA_VERSION` (`mesh/_femdata_h5_io.py:151`) =
  **2.10.0** — stamps the root neutral zone.

A reader at `X.Y` accepts only `X.Y.*` and `X.(Y-1).*`; anything
newer / older / different-major raises `SchemaVersionError`.

## Section cuts / sweeps — model.h5 persistence

`SectionCutDef` / `SectionSweepDef` persist under `/opensees/cuts/`
and `/opensees/sweeps/` (element_ids carry OpenSees tags, not FEM
eids — hence under `/opensees/`). Schema = bridge **2.12.0**. Three
writer paths:

- in-shot — `apeSees.h5(path, cuts=[…], sweeps=[…])`;
- append — `cuts.persist_to_h5(path, …)` (deletes only supplied groups);
- primitive — `cuts._h5_io.write_cuts_into(f, …)` (raises if groups exist).

The viewer auto-loads persisted cuts; an explicit `viewer(cuts=[…])`
kwarg **wins** over persisted (no merge).
<!-- verified: tests/opensees/h5/test_h5_apesees_cuts.py::test_apesees_h5_with_cuts_writes_groups, ::test_apesees_h5_bumps_schema_version_to_2_5_0 -->

## Emit / run / inspect

```python
ops.tcl("out/model.tcl", run=False)   # classic OpenSees deck (run=True subprocesses)
ops.py("out/model.py", run=False)     # openseespy script
ops.h5("out/model.h5")                # native: /opensees + neutral zone
ops.run(wipe=True)                    # in-process openseespy (LiveOpsEmitter)
ops.analyze(steps=10, dt=0.01)        # drive the analysis chain (NON-staged only)
ops.eigen(...)                        # NON-staged only
```

`tcl` / `py` take `analyze_steps=` / `analyze_dt=` (append an
`analyze` line) and `split=` (ADR 0043 split emit). These are
**separate statements** — not a fluent chain; each `tcl/py/h5/run`
calls `build()` internally. Post-emit inspection is broker-side
(`fem.inspect.summary()`, `fem.inspect.node_table()`) or via
`apeGmsh.opensees.emitter.h5_reader.open("model.h5")`.

## Canonical skeleton

```python
from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees

with apeGmsh(model_name="solid") as g:
    box = g.model.geometry.add_box(0, 0, 0, 10, 5, 2, label="body")
    g.physical.add_volume("body", name="Body")
    g.physical.add_surface([1], name="Base")
    g.mesh.sizing.set_global_size(0.5)
    g.mesh.generation.generate(dim=3)
    g.mesh.partitioning.renumber(dim=3, method="rcm", base=1)
    fem = g.mesh.queries.get_fem_data(dim=3)

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
- **Masses / fixities missing from the deck** — these are NOT
  auto-ingested; re-declare them on `ops` (`ops.fix` / `ops.mass`).
  (Loads from `g.loads.*` and MP constraints, by contrast, *are*
  auto-emitted.)
- **Reactions / loads come out at 2×** — you declared the same load
  on **both** `g.loads.*` (auto-emitted) **and** a bridge pattern
  (`p.load`). Pick one channel per load.
- **Ambiguous `pg=`** — same name at multiple dimensions. Keep PG
  names dimension-unique.
- **`len(dofs) != ndf`** — `ops.fix` needs a mask of length `ndf`.
- **`BridgeError` at `ops.model(...)`** — the envelope `ndf` is
  smaller than the max per-node ndf you declared via `g.node_ndf`.
- **`NotImplementedError` on `ops.analyze()`/`ops.eigen()`** — the
  model has stages; drive it via `ops.tcl(path, run=True)` instead.
