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

## What the bridge auto-emits vs what you import / re-declare

The bridge reads the `fem` snapshot to resolve `pg=` / `label=`
selectors and to get coords/connectivity. Three policies apply to what
ends up in the runnable deck (ADR 0051):

| Surface | How it reaches the deck |
|---|---|
| **MP constraints** (`fem.nodes.constraints` / `fem.elements.constraints`) | **AUTO-EMIT** (ADR 0022, shipped v2.0.0) |
| **Loads** (`fem.nodes.loads`, via `g.loads.case(...)`) + **prescribed displacements** (`fem.nodes.sp`, via `g.displacements.case(...)`) | **OPT-IN** — `p.from_model(case)` inside a bridge pattern (or author with `p.load` / `p.sp`). **No auto-emit.** |
| Masses, homogeneous SPs (fixities) | **RE-DECLARE explicitly** on `ops` (`ops.mass` / `ops.fix`) |

So loads are **opt-in** (ADR 0051 reversed the old auto-emit): a
geometry **case** reaches the deck only when a bridge **pattern**
imports it. The geometry groups by `g.loads.case("dead")` (a label, no
temporal meaning); the OpenSees pattern + `timeSeries` is born on the
bridge; `p.from_model("dead")` is the seam that replays the resolved
nodal records as `load` / `sp` lines. MP constraints still auto-emit
(equalDOF, rigid link, rigid diaphragm, embedded). Lumped masses and
fixities are re-declared on `ops`.

> **No double-counting; the deck is authoritative.** Nothing
> loads-related auto-emits, so importing a case once with
> `p.from_model(case)` is the single channel — the old "session
> `g.loads` **and** bridge `p.load` → 2×" trap is gone. The bridge
> applies exactly the cases you import and does NOT audit the geometry's
> case-list against the deck (no `WarnUnconsumedModelLoads` / no
> `ops.ignore_model_loads` — both were removed). A case you don't import
> is simply not applied; an import of a non-existent case name is a
> no-op (check `fem.nodes.loads.patterns()`).

Session declarations all still flow into the **`model.h5` neutral
zone** (`ops.h5(path)`), so the **viewer / `Results`** see everything
regardless of whether a pattern imported it.

> **Two execution modes — no mixing (ADR 0051 §5).** Non-staged (a
> global `ops.pattern.*` + the chain + `ops.analyze`/`ops.eigen`) OR
> staged (every pattern stage-scoped via `s.pattern(series=...)`, run
> through `ops.tcl`/`ops.py`). A global pattern + a stage raises
> `BridgeError`. In a staged deck, open the pattern inside the stage:
> `with s.pattern(series=ts) as p: p.from_model("live")`.

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
- **`ops.uniaxialMaterial`** → `truss`, `corotTruss`, `zeroLength` /
  `twoNodeLink` / `CoupledZeroLength` springs, and fiber-section beams.
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

### Choosing an element (modeling judgment)

The bridge is agnostic; pick by mechanics. One-clause defaults (for the
*why*, see the `fem-mechanics-expert` skill — locking, B-bar, integration):

| Mesh | Default | Reach for instead |
|---|---|---|
| 3-D solid (hex/tet) | `stdBrick` / `FourNodeTetrahedron` | `SSPbrick` (cheaper, stabilized hourglass), `bbarBrick` (near-incompressible / plasticity, beats volumetric locking) |
| 2-D solid | `quad` (`SSPquad` stabilized) | `tri31` only where a quad mesh won't form |
| Shell | `ShellMITC4` (general 4-node) | `ASDShellQ4` (drilling DOF, large-disp), `ShellDKGQ` (thin, no shear) |
| Beam/column | `forceBeamColumn` (force-based, fewer elems for spread plasticity) | `dispBeamColumn` (displacement-based, stiff/short members or when force-based won't converge) |

Fiber sections (`ops.section.Fiber`) make beams/shells nonlinear; `elasticBeamColumn`
stays linear. Fork-only continuum (`LadrunoBrick` etc.) is in `ladruno.md`.

### Springs, hinges, dashpots (ZeroLength family)

`ops.uniaxialMaterial` springs ride 2-node (typically coincident) PGs.
Each `(material, dof)` pair is a 1-DOF spring (`dof` 1-based: 1=Ux…6=Rz);
reuse a material across DOFs, or stack two on the same DOF to add them.

```python
from apeGmsh.opensees.element.zero_length import ZeroLengthMatDir

# Rotational hinge — moment-rotation material on dir 6 (Mz)
k = ops.uniaxialMaterial.Steel02(fy=420e6, E=200e9, b=0.01)
ops.element.ZeroLength(pg="Hinges",
    mat_dirs=(ZeroLengthMatDir(material=k, dof=6),))

# Viscous dashpot / Lysmer absorbing boundary — Viscous ∥ stabilising elastic
c  = ops.uniaxialMaterial.Viscous(C=rho*cp*A, alpha=1.0)   # ZERO static stiffness
ks = ops.uniaxialMaterial.ElasticMaterial(E=k_stab)
ops.element.ZeroLength(pg="AbsBnd",
    mat_dirs=(ZeroLengthMatDir(material=c,  dof=1),
              ZeroLengthMatDir(material=ks, dof=1)))        # both dir 1 → stiffness adds

# Node-pair form (ADR 0049) — spring to a g.decouple_node ground, NO meshed line.
# Pass nodes=(node_i, node_j) instead of pg=; each endpoint is a decouple_node
# handle, a single-node label, or an int tag. The dominant SSI case: a boundary
# mesh node ↔ a coincident decoupled ground.
gnd = g.decouple_node(coords=(x, y, z), label="pile_ground")   # session-side
ops.ndf(gnd, ndf=3)                                            # size the element-less ground
ops.element.ZeroLength(nodes=("boundary_node", gnd),           # mesh-node label + ground handle
    mat_dirs=(ZeroLengthMatDir(material=ks, dof=1),
              ZeroLengthMatDir(material=ks, dof=2),
              ZeroLengthMatDir(material=ks, dof=3)))
```

- **Node-pair `nodes=(i, j)`** works on `ZeroLength` / `CoupledZeroLength` /
  `TwoNodeLink` (mutually exclusive with `pg=`). Both ends must carry **equal
  `ndf`** (G1 fails loud otherwise) and resolve to **distinct** nodes. Not for
  `ZeroLengthSection` (non-adaptive — use `pg=` or a plain `ZeroLength`). v1
  limits: global-only (no stage binding), fails loud under partitioned/MPI emit,
  and not drawn in the viewer (no mesh cell). Int endpoints are a non-compose-safe
  escape hatch — prefer a handle or label.
- **`Viscous` / `ViscousDamper` / `Maxwell`** are the rate-dependent (dashpot)
  uniaxials — they produce a velocity-proportional force with **no `-doRayleigh`**.
  A pure `Viscous` has zero static stiffness → parallel it with an elastic spring
  on the same DOF, or the static tangent is singular.
- **`ZeroLengthSection`** (fiber / `section.Aggregator` hinge, real P–M coupling)
  requires `ndf` 3 (2D) or 6 (3D), and its `do_rayleigh` defaults **ON** — the
  inverse of plain `ZeroLength`. Pass `do_rayleigh=False` to disable.
- **`TwoNodeLink`** — finite-length link with `-mass` / `-pDelta` / `-shearDist`
  (the only spring that carries mass). **`CoupledZeroLength`** — one material on
  the resultant of two dirs (bidirectional / bearing spring).
- For frequency-band damping on a group of springs, attach a `ops.damping.*`
  object by PG (`on="MyPG"`) rather than a per-element flag.

## Boundary conditions, masses, loads — explicit / opt-in

The `Emitter` protocol exposes `node / fix / mass / element / sp`
for the model side. **Masses and homogeneous SPs (fixities)** are
**re-declared** on `ops` (`ops.mass` / `ops.fix`). **Loads** are
**opt-in** (ADR 0051): a `g.loads.case(...)` reaches the deck only when
a bridge pattern imports it with `p.from_model(case)`, or you author it
directly with `p.load(...)`:

```python
# Homogeneous SP (fixities). dofs length = ndf.
ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))     # verified: tests/opensees/unit/test_emitter_protocol.py::test_fix_records_tag_and_dofs
# ops.fix(nodes=[...], dofs=(...))              # explicit-node form

# Lumped mass.
ops.mass(pg="Roof", values=(m, m, m, 0.0, 0.0, 0.0))

# Nodal loads + prescribed (non-zero) SP — pattern-scoped.
ts = ops.timeSeries.Linear()                    # also Constant/Path/Trig/Pulse
with ops.pattern.Plain(series=ts) as p:         # also UniformExcitation   # verified: tests/opensees/unit/test_emitter_protocol.py::test_pattern_open_close_pair
    p.from_model("dead")                         # import the resolved g.loads case
    p.load(pg="Tip", forces=(0.0, 0.0, -5e4))   # + ad-hoc bridge-authored load
    p.sp(pg="LoadingPin", dof=3, value=0.01)    # prescribed displacement
```

`p.from_model(case)` replays the resolved nodal records tagged that
case (loads → `load`, prescribed disp → `sp`; homogeneous fixes are
never imported — use `ops.fix`). `p.load` / `p.sp` fan a `pg=` across
the group's nodes at build time; `node=` takes an explicit tag or a
`Node` from `ops.nodes.get(...)`. In a staged deck, open the pattern
inside the stage with `s.pattern(series=...)` — a global pattern + a
stage raises `BridgeError` (no-mixing, ADR 0051 §5). To release a
prior-tier BC inside a stage use `s.remove_bc(...)` (the
`g.constraints.bc`-reading alias of `s.remove_sp`; `dofs=` are 1-based
DOF indices, not the fix flag vector).

## Moment-tensor seismic source (ADR 0062)

Embed an earthquake **point source** inside the continuum; the bridge emits the
representation-theorem equivalent **nodal forces** `M0·m_ij·∂N_a/∂x_j` — a pure
right-hand-side load (no stiffness, no constraint, integrator-agnostic).
Declared inside a `Plain` pattern whose `timeSeries` carries the normalized
moment function `S(t)` (rising `0 → 1`; slip-rate is `S'(t)`).

```python
# S(t) helpers (ADR 0062). Both ride ops.timeSeries.* and emit as a Path.
ts = ops.timeSeries.MomentStep(half_duration=0.5, t_total=20.0, dt=0.005, t0=2.0)
#   erf ramp; slip-rate is a Gaussian of std=half_duration centred at t0.
# or the finite-fault kinematic source-time function:
ts = ops.timeSeries.Yoffe(rise_time=1.2, peak_time=0.3, t_total=20.0, dt=0.005)
#   regularized modified-Yoffe; peak_time MUST be < rise_time/2.
#   Both also take f_max= (band-limit warn) and factor= (Path -factor scale).

with ops.pattern.Plain(series=ts) as p:                  # verified: src/apeGmsh/opensees/pattern/pattern.py:338
    p.moment_tensor(
        position=(x, y, z),          # mesh coords (or (x, y) in 2D); host element found automatically
        frame="z-up",                # REQUIRED — "z-up" | "z-down"; wrong frame mirrors the radiation pattern
        M0=3.2e16,                   # scalar seismic moment M0=μ·A·D̄ in deck moment units (default 1.0)
        mech=dict(strike=30, dip=80, rake=-10),   # degrees; OR m_ij=<3×3 unit tensor, A&R z-down> — exactly one
        method="consistent",         # host ∂N/∂x (any solid mesh); or "dipole" (±neighbour couples, structured grid)
        region="soil",               # optional: restrict host search to a PG/label (else fails loud on a skin/outside point)
    )
ops.tcl("eq.tcl", run=True)          # resolution happens in the bridge BUILD (it owns the FEM snapshot)
```

- Supply **exactly one** of `mech=dict(strike, dip, rake)` (degrees) or `m_ij=`
  (a 3×3 unit moment tensor, Aki & Richards z-down). `t0` only supports `0.0`
  for a single source — per-source onset is the finite-fault path below.
- `frame=` has **no default**: `"z-up"` is typical apeGmsh (free surface on top,
  depth positive downward; flips the A&R z-down tensor).
- The pattern primitive only **records** — `p._emit` raises. Always emit via the
  bridge (`ops.tcl` / `ops.py` / `ops.build`), which owns the FEM snapshot
  needed to locate the host and evaluate `∂N/∂x`. `p.moment_tensors` is the
  recorded-source snapshot.

### Finite fault — `ops.fault.from_shakermaker(...)` (MT-4)

Turn a ShakerMaker `FaultSource` into **one `Plain` pattern per `PointSource`**
(each a moment-tensor source with its own onset, `Yoffe` STF, and position):

```python
patterns = ops.fault.from_shakermaker(          # verified: src/apeGmsh/opensees/_internal/ns/fault.py:110
    fault,                       # duck-typed: iterates PointSources (.x [km], .angles [RADIANS], .tt onset)
    frame="z-up",
    M0=3.2e16,                   # scalar (shared) OR a per-source sequence
    rise_time=1.2, peak_time=0.3,# scalar OR per-source sequence (Yoffe params)
    dt=0.005, t_total=40.0,
    length_scale=1000.0,         # deck-length per km — REQUIRED
    method="consistent",
    region="soil",               # optional host restriction (MT-5a)
)
# Returns a list[Plain] — still emit via the bridge. Warns (subclasses of
# UserWarning) on subfaults skipped (non-positive M0/rise/peak), clamped
# (peak_time > rise_time/2), truncated (rise spills past t_total), or never
# fired (onset ≥ t_total).
```

## Damping — `ops.damping` (ADR 0053)

Domain-level (sibling of `fix`/`mass`/`region`), **not** in the analysis
chain. Every member is a declaration resolved at emit — no `assign`, no
held tag. Owns 4 channels; material dashpots stay in
`ops.uniaxialMaterial.*` (`Viscous`/`ViscousDamper`/…), numerical damping
in `ops.integrator.*`.

```python
# Rayleigh — raw OR ratio; global (no on=) OR region-scoped (on=PG/list).
ops.damping.rayleigh(alpha_m=0.1, beta_k=0.01)                  # raw, global
ops.damping.rayleigh(ratio=0.05, f_i=1.0, f_j=10.0)             # ratio fit (α, β)
ops.damping.rayleigh(ratio=0.05, f_i=1.0, f_j=10.0, on="Soil")  # region (-ele)
# ratio β lands by stiffness= : default "initial"=βK0 (nonlinear-safe);
# "current"/"committed" explicit. This is what -doRayleigh opts INTO.
# Element Rayleigh OVERWRITES per element → global+on= overlap warns
# (RayleighOverwriteWarning); resolver emits global before region.

# Modal — bundles its own eigen; domain-wide (no on=). NO modal_q
# (modalDampingQ is an upstream anti-damping bug).
ops.damping.modal(0.05, modes=4)                 # uniform; or [..] len==modes

# Tagged objects → region -damp attach (on= required, OR element damp=).
ops.damping.uniform(ratio=0.03, freq_lower=0.5, freq_upper=10.0, on="Soil")
ops.damping.sec_stif(beta=0.002, on="Soil")
ops.damping.urd(points=[(0.5,0.02),(5,0.03),(20,0.05)], on="Soil")   # N>=2 asc
ops.damping.urd_beta(points=[(0.5,1e-3),(10,2e-3)], on="Soil")
# uniform ratio = PHYSICAL ζ (OpenSees ×2 internally — don't pre-divide).
# All four: activate_time/deactivate_time (off during gravity) + factor=
# (an ops.timeSeries.* → -factor). Element-flag attach instead of a region:
damp = ops.damping.uniform(ratio=0.03, freq_lower=0.5, freq_upper=10.0)
ops.element.elasticBeamColumn(pg="Cols", transf=t, A=.01, E=2e11, Iz=1e-4, damp=damp)
```

`damp=` only on `-damp`-capable elements (elasticBeamColumn /
forceBeamColumn / dispBeamColumn / stdBrick / FourNodeQuad / Shell family /
ZeroLength) — else `TypeError`. An object attached to nothing (no `on=`,
no element) raises at `build()` (no global `-damp`). Persists to
`/opensees/dampings` (schema 2.15.0, folds into `model_hash`); element-flag
attach round-trips, region `-damp`/`-rayleigh` attach does not (archival
`/opensees/regions`, like all region state). **Staged:** `s.damping.*`
(same verbs, resolve inside the stage after `domainChange`); `s.damping.modal`
raises (per-stage modal deferred).

## ✅ Multi-point constraints ARE emitted (ADR 0022, shipped v2.0.0)

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

## Per-node ndf — inferred from elements + `ops.ndf` (ADR 0048/0049)

Per-node `ndf` is **inferred** from the declared element classes — you do
**not** set it on the session (the old `g.node_ndf` composite was removed). You
declare elements + `ndm`; the bridge resolves each node's `ndf` from the
incident element classes' DOF floor (e.g. a `ShellMITC4` node → 6, a
`stdBrick` node → 3, a 2D `elasticBeamColumn` node → 3), validates it against
every incident element's `ndf_ok` (the shell-on-solid `∩` gate), and emits a
per-node `-ndf K` token — **elided** when `K` equals the `ops.model(..., ndf=)`
envelope, so homogeneous decks stay token-free.

```python
ops.model(ndm=3, ndf=3)                 # envelope; solid nodes infer 3 → elided
ops.element.ShellMITC4(pg="Deck", ...)  # Deck nodes infer 6 → emit "-ndf 6"
```

A node shared by two elements with disjoint `ndf_ok` (shell `{6}` vs solid
`{3}`) fails loud — the fix is separate coincident nodes + `equal_dof` / `tie`,
never a shared node (see the shell-on-solid idiom / ADR 0046).

**`ops.ndf` — the one explicit channel (element-less decoupled nodes only).**
A node inference cannot reach — an SSI spring/dashpot **ground**, a control
node, a mass anchor created via `g.decouple_node(...)` that no element touches —
has its `ndf` stated on the bridge:

```python
gnd = g.decouple_node(coords=(x, y, z), label="pile_ground")  # session: identity
...
ops.ndf(gnd, ndf=3)          # bridge: DOF count (handle or int tag)
```
<!-- verified: tests/opensees/unit/test_ops_ndf.py::test_ops_ndf_emits_stated_ndf_on_decoupled_node -->

`ops.ndf` targets a `g.decouple_node` handle or its int tag (a `label=`/`pg=`
grammar is deferred). It **fails loud** on a mesh node or any element-touched
node — inference owns those, and restating would create a two-headed model. The
stated value is checked by three gates (all `BridgeError` at build): **G1**
adaptive 2-node-link endpoints must agree, **G2** a `rigidDiaphragm`/`rigidLink`
master must carry the exact ndf (6 in 3D / 3 in 2D) and every constrained DOF
must fit the endpoint ndf, **G3** a `mass`/`load` vector must EQUAL the node ndf
and a `fix`/`sp` DOF must fit it — OpenSees silently drops each of these
otherwise. The resolved per-node `ndf` (inferred ∪ stated) persists to
`/opensees/nodes_ndf` and round-trips through `model.h5`.

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

> **Ladruno fork recorder (`.ladruno`).** On the **Ladruno fork**
> (`nmorabowen/OpenSees@ladruno`) there is a fork-only recorder distinct from
> STKO's `.mpco`: command **`recorder ladruno …`**, writing **`.ladruno`** HDF5
> (`GENERATOR="Ladruno"`, `FORMAT_VERSION=1`). It is self-describing (elements
> declare basis/quadrature) and — unlike `.mpco` — **writes `MODEL/LOCAL_AXES`**
> (per-class quaternion `FRAME`) for beams, so beam `line_force`/section diagrams
> can be oriented straight from the file. apeGmsh can emit it via `ops.tcl/py`
> (`recorder ladruno`); a typed `ops.recorder.Ladruno` and a `Results.from_ladruno`
> reader are the **recommended** apeGmsh-side additions (not yet shipped). Full
> emit/read contract: `Ladruno_implementation/ladruno_apegmsh_contract.md` in the
> fork. The fork is opt-in; stock `openseespy` (no `.ladruno`) stays first-class.

## Native model.h5 — two zones + schema constants

`ops.h5(path, *, model_name=None, cuts=(), sweeps=())` writes a
**two-zone** `model.h5`: the *neutral* zone (broker-written —
nodes/elements/PGs/labels/constraints/loads/masses, same as
`g.save()`) **and** the `/opensees/` zone (bridge-written —
transforms, recorders, dampings, cuts/sweeps). `g.save()` / `FEMData.to_h5`
write **only** the neutral zone; `apeSees(fem).h5(path)` writes both.

Two **independent** per-zone schema constants (ADR 0023):

- bridge `SCHEMA_VERSION` (`opensees/emitter/h5.py:379`) = **2.19.0**
  — stamps `/opensees/…` (2.15.0 added `/opensees/dampings`, ADR 0053 D3b;
  later bumps carry coupling-knob + staged-partition work).
- broker `NEUTRAL_SCHEMA_VERSION` (`mesh/_femdata_h5_io.py:165`) =
  **2.13.0** — stamps the root neutral zone.

A reader at `X.Y` accepts only `X.Y.*` and `X.(Y-1).*`; anything
newer / older / different-major raises `SchemaVersionError`.

## Section cuts / sweeps — model.h5 persistence

`SectionCutDef` / `SectionSweepDef` persist under `/opensees/cuts/`
and `/opensees/sweeps/` (element_ids carry OpenSees tags, not FEM
eids — hence under `/opensees/`). They are stamped with the current bridge
`SCHEMA_VERSION` (see above). Three writer paths:

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

**Per-rank Tcl decks — `ops.tcl(path, per_rank=True)` (ADR 0061).** For a
**partitioned** model, write the driver deck at `path` plus one guard-free
`ranks/rank<K>_<seq>.tcl` fragment per `if {[getPID]==K} {…}` block; the
driver `source`s each rank's fragment. Every MPI rank then parses only the
driver + its own fragments — `O(global + model/np)` instead of `O(model)` per
rank. Deck semantics are unchanged (layout-only). Requires `len(fem.partitions)
> 1` and is **mutually exclusive with `split=True`** — either condition raises
`ValueError`.

```python
ops.tcl("main.tcl", per_rank=True)   # → main.tcl + ranks/rank0_0.tcl, ranks/rank1_0.tcl, …
```

## Which OpenSees runs — `OpenSeesTarget`

The bridge talks to **three** distinct runtimes — live in-process
openseespy (`run`/`analyze`/`eigen`), a Tcl binary (`tcl(run=True)`),
and an openseespy subprocess (`py(run=True)`). By default each resolves
from env vars + `PATH`, so you usually do nothing. To pin them
explicitly, pass one `OpenSeesTarget` on construction:

```python
from apeGmsh.opensees import apeSees, OpenSeesTarget   # verified: tests/opensees/unit/test_opensees_target.py

fork = OpenSeesTarget(
    binary="C:/Program Files/Ladruno/OpenSees/bin/OpenSees.exe",  # ops.tcl(run=True)
    python="C:/Users/nmora/venv/opensees_venv/Scripts/python.exe",# ops.py(run=True)
    require_fork=True,   # assert the LIVE build is the Ladruno fork
)
ops = apeSees(fem, opensees=fork)
ops.opensees                 # → the bound OpenSeesTarget, or None
```

All three fields are optional. Resolution precedence (consistent for
both subprocess paths):

| Path | precedence |
|---|---|
| `tcl(run=True)` | `bin=` arg → `target.binary` → `$OPENSEES_BIN` → `which("OpenSees")` |
| `py(run=True)` | `python=` arg → `target.python` → `$OPENSEES_VENV` → `which("python")` → `sys.executable` |

`ops.py(..., python=...)` mirrors the existing `ops.tcl(..., bin=...)`
per-call override.

**`binary` / `python` are inert for the LIVE path** — you cannot swap
`import openseespy` under a running interpreter, so for live fork
features you still launch your script *under the fork's venv*. The only
live-relevant field is `require_fork`: it turns "fork expected" into a
loud, early failure at `run()` / `analyze()` / `eigen()` instead of a
cryptic one three primitives deep.

```python
ops = apeSees(fem, opensees=OpenSeesTarget(require_fork=True))
ops.element.BezierTet10(pg="Body", material=m)
ops.run()
# RuntimeError: OpenSeesTarget(require_fork=True) but the in-process
# openseespy build does not look like the Ladruno fork ...
```

**A target is not a fork switch.** Pointing `binary=` at the fork build
does *not* tell apeGmsh the build has `BezierTet10` — fork-only features
stay gated by capability detection at the point of use. To branch on it
yourself, probe the **live** build:

```python
caps = ops.capabilities()
# OpenSeesCapabilities(source='live', has_fork=True, has_profiler=True, version='3.8.0')
if caps.has_fork:
    ops.element.BezierTet10(pg="Body", material=m)
else:
    ops.element.FourNodeTetrahedron(pg="Body", material=m)
```

`has_fork` tracks the fork-only `profiler` command (the same gate the
live emitter uses). `capabilities()` introspects the **live** runtime
only — the subprocess paths bind their own interpreter / binary.

## Remote SLURM runs — `ops.run_remote` + `apeGmsh.hpc` (ADR 0060)

Emit the deck into a job dir, push it to a cluster over SSH, `sbatch` it, and
(optionally) block until it finishes — one call:

```python
job = ops.run_remote(                # verified: src/apeGmsh/opensees/apesees.py:6091
    "./run/tunnel-v3",              # job dir (created if missing; deck emitted here)
    cluster="esmeralda",           # name in ~/.apegmsh/clusters.toml, OR a Cluster object
    np=64, name="tunnel-v3",
    analyze_steps=6000, analyze_dt=0.002,
    wait=True, timeout=7200.0,     # block on the job; logs + results always fetched back
)                                  # raises HPCError (with stderr tail) if status != COMPLETED
```

Lower-level primitives (`from apeGmsh.hpc import Cluster, Job, JobStatus`):

```python
cluster = Cluster.load("esmeralda")           # reads ~/.apegmsh/clusters.toml [esmeralda]
ops.tcl("./run/main.tcl", analyze_steps=6000, analyze_dt=0.002)
job = cluster.submit("./run", np=64, name="tunnel-v3", deck="main.tcl")  # returns immediately
job.status()                                  # JobStatus.{PENDING,RUNNING,COMPLETED,FAILED,CANCELLED,UNKNOWN}
job.tail(30, stream="out")                    # last n lines of stdout ("out") / stderr ("err")
job.wait(poll=15.0, timeout=7200)             # block to terminal state (TimeoutError if exceeded)
job.fetch()                                   # pull the remote dir back to local_dir
job = Job.load("./run")                       # rehydrate from the .apegmsh_job.json sidecar later
```

Two-layer config (apeGmsh never touches credentials): `~/.ssh/config` owns the
SSH connection (host / user / key / `BatchMode`); `~/.apegmsh/clusters.toml`
owns cluster facts (`remote_root`, `slurm_bin`, `partition`, `opensees_bin`,
`launcher`, `env`). The sbatch script is rendered **locally** before push (LF +
UTF-8 — CRLF breaks the remote shebang) and sits next to the deck for
inspection / replay. Set `slurm_bin` when SLURM is only on the login-shell PATH
(a non-interactive `ssh host cmd` does not source `/etc/profile`); job
completion is read from an `.exit_code` sentinel when `sacct` is broken. The
default launcher is `srun --cpu-bind=cores --mpi=pmix_v3 {binary} {deck}`
(override per cluster). Run-verified end-to-end on the Esmeralda cluster.

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
  auto-emitted; re-declare them on `ops` (`ops.fix` / `ops.mass`).
- **Loads missing from the deck** — `g.loads.*` are **opt-in** (ADR
  0051): import the case with `p.from_model(case)` inside a pattern (or
  author `p.load`). The bridge applies only what a pattern imports and
  does NOT warn about un-imported geometry cases — if a load is missing,
  you didn't import its case (or typo'd the case name; check
  `fem.nodes.loads.patterns()`). MP constraints, by contrast, *do*
  auto-emit.
- **`BridgeError: cannot mix a global ops.pattern.* with stages`** —
  a staged model must keep every pattern stage-scoped
  (`s.pattern(series=...)`); don't register a global `ops.pattern.Plain`
  (or `ops.imposed_displacement`, which builds one) alongside
  `ops.stage(...)`.
- **Ambiguous `pg=`** — same name at multiple dimensions. Keep PG
  names dimension-unique.
- **`len(dofs) != ndf`** — `ops.fix` needs a mask no longer than the node
  `ndf`; a `mass`/`load` vector must EQUAL it (ADR 0049 G3).
- **`BridgeError` about an element class "not in the capability registry"** —
  per-node `ndf` is inferred from element classes; an unregistered element
  can't be inferred. **`BridgeError: ops.ndf … not a decoupled node`** — `ops.ndf`
  only sizes element-less `g.decouple_node` nodes; mesh-node ndf is inferred.
- **`NotImplementedError` on `ops.analyze()`/`ops.eigen()`** — the
  model has stages; drive it via `ops.tcl(path, run=True)` instead.
