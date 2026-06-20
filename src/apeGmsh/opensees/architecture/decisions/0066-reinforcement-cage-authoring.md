# ADR 0066 — Reinforcement-cage authoring: a `g.rebar` geometry layer above `g.reinforce`

**Status:** Proposed (2026-06-19; refined by a 14-agent design workflow,
every signature cross-checked against `src/apeGmsh`).  Builds on — and
explicitly **delegates to** — the already-shipped `g.reinforce`
composite (`ReinforcementsComposite`).  Threads through ADR 0022
(MP-constraint emission), ADR 0035/0036 (`ASDEmbeddedNodeElement`
exposure + host decomposition), ADR 0038 (`g.compose`), and ADR 0041
(chain-phase routing for geometry-intensive constraints).  No OpenSees
C++ change.  No H5 schema bump for the **inline** path; the
**composed-Part** path is gated on lifting an existing deferral (see
§6.3 and Open Items).

## Context

Reinforced-concrete 3D solid models on the Ladruno fork have a mature
mechanics stack (`LadrunoBrick`, `BezierTet10`, `ASDConcrete3D`,
`LadrunoEmbeddedRebar` ELE 33005 + `LadrunoBondSlip` MAT 33002) but a
thin **authoring** layer for the steel.  Two surfaces exist today:

1. **`OpenSees/Ladruno_scripts/ladruno_rc.py`** (`RCColumnSpec`) — a
   pure-openseespy helper that builds a *conformal* rectangular column:
   longitudinal `corotTruss` bars on the concrete mesh's perimeter
   node-lines (shared nodes ⇒ perfect bond) plus `corotTruss` tie rings
   on z-planes.  It documents its own wall: *"a bar/tie can only sit on
   a mesh node-line… that couples reinforcement layout to mesh
   density."*  No real bar geometry, no hooks, no bend radii, locked to
   the grid.

2. **`g.reinforce` (`ReinforcementsComposite`,
   `src/apeGmsh/core/ReinforcementsComposite.py`)** — the *binding*
   composite.  `reinforce(host, bars, *, bond, perfect, bar_diameter,
   bar_area, kt, kt_alpha, enforce='penalty', bipenalty, dtcr,
   tolerance=1e-6, snap=False, …)` takes a **pre-meshed** rebar line PG,
   inverse-maps each bar node into the host solid, resolves at
   `get_fem_data`, and emits `LadrunoEmbeddedRebar` (ELE 33005) via
   `emit_reinforce_ties` (`opensees/_internal/build.py:3160`).  It
   already carries a compose-aware `ReinforceTieRecord` and an
   Option-B "bond-by-name" layering (the geometry composite never holds
   an OpenSees tag).

What is missing is the **geometry-authoring** layer between a designer's
intent ("a column cage: 8 #8 verticals with seismic 90° hooks, #4 ties
at 100 mm densifying to 50 mm in the hinge zones") and the curve PGs +
coupling calls those two surfaces consume.  That layer must:

- produce arbitrary polyline bars and stirrups (not grid-locked),
- detail hooks and bends with a code-aware standard (ACI 318),
- be serializable/reusable (a cage authored once, stamped into many
  members), and
- route to **either** conformal (shared-node perfect bond, the
  `ladruno_rc.py` behaviour generalised) **or** embedded
  (`LadrunoEmbeddedRebar`, the non-matching path) coupling — without
  the author hand-writing `embed()` or `reinforce()` calls.

This ADR specifies that layer as a new first-class composite, `g.rebar`.

## Decision

### §1 — Position: `g.rebar` authors geometry, delegates coupling

`g.rebar` is a **new geometry-authoring composite** that sits *above*
the shipped `g.reinforce` binding composite.  It owns reinforcement
*specs* and *geometry*; it owns **no** coupling emitter.  Delegation map
(the ADR pins which surface emits which OpenSees primitive):

| Coupling | `g.rebar` action | Emits |
|---|---|---|
| **conformal** (inline) | `g.mesh.editing.embed(bar_pg, host, dim=1, in_dim=host_dim)` before `generate()` | nothing — shared nodes; bar `Truss`/`CorotTruss`/`DispBeamColumn` emitted at bridge time on shared nodes |
| **embedded** (inline) | forward to `g.reinforce(host, bars, …)` | `LadrunoEmbeddedRebar` (ELE 33005), per-rebar-node, bond-by-name |
| **embedded** (composed Part) | assembly-seam `g.reinforce(...)` (fork) / `g.constraints.embedded(...)` (stock) | `LadrunoEmbeddedRebar` / `ASDEmbeddedNodeElement` |

`g.reinforce` → `LadrunoEmbeddedRebar` and `g.constraints.embedded` →
`ASDEmbeddedNodeElement` (Kuhn corner decomposition, ADR 0035/0036) are
**different elements**; `g.rebar` selects the fork path
(`g.reinforce`) by default and exposes the stock path only on the
cross-Part seam for non-fork targets.

### §2 — Three layers, three homes

| Layer | What | Home (mirrors the `loads` split) |
|---|---|---|
| **L1** specs | frozen, serializable, unitless data: `Hook`, `Path`, `Bar`, `Stirrup`, `Cage`, `DetailingStandard`, `BarCatalog` | `_kernel/defs/rebar.py` (beside `constraints.py`), detailing in `apeGmsh/rebar/detailing.py` |
| **L2** composite | `RebarComposite` — geometry generation + standardized members + coupling delegation | `core/RebarComposite.py`, registered in `_core.py` `_COMPOSITES` as `('rebar', '.core.RebarComposite', 'RebarComposite', False)` + a `rebar: RebarComposite` type decl |
| **L3** fluent | `BarBuilder` sugar (`g.rebar.bar(db=,material=).through(...).hook_end(...).as_(name)`; `.as_()`/`.build()` terminals) | `_kernel/defs/rebar.py` |

The `Cage` is the serializable source of truth (`to_dict`/`from_dict`);
the same `Cage` object feeds both binding surfaces (inline and
composed-Part).  L1 never imports a units singleton (none exists — the
unit-agnostic invariant is preserved); L1 never resolves an OpenSees
tag (Option-B, as `ReinforceDef` already does).

### §3 — Spec objects

```python
# _kernel/defs/rebar.py  — all frozen, all pure data
@dataclass(frozen=True)
class Hook:
    angle: float                              # 90 | 135 | 180 (deg)
    tail: float | str                         # absolute length OR "<k>db"
    bend_radius: float | str | None = None    # None ⇒ standard fills from db+size
    turn: str | tuple[float, float, float] = "centroid"   # bend-plane selector
    true_arc: bool = False                     # False ⇒ polyline+metadata
    name: str | None = None
    # factories (std-injected at bind; "<k>db" resolved lazily):
    #   Hook.seismic_135(tail="6db")  Hook.standard_90(tail="12db")
    #   Hook.standard_180(tail="4db", min_tail=2.5)

@dataclass(frozen=True)
class Path:
    points: tuple[Vec3, ...]
    corner_radius: float | str = "metadata"   # default ⇒ sharp polyline + radius metadata

@dataclass(frozen=True)
class Bar:
    path: Path
    db: float | str                            # designation ("#8") or raw length
    material: str                              # by NAME (bridge resolves)
    role: str = "longitudinal"
    element: str = "truss"                     # "truss" | "beam"  (see §7)
    start_hook: Hook | None = None
    end_hook: Hook | None = None
    name: str | None = None

@dataclass(frozen=True)
class Stirrup:                                  # NOT a closed loop — see below
    path: Path                                  # open corner polyline
    db: float | str
    material: str
    closure_hook: Hook = field(default_factory=Hook.seismic_135)
    name: str | None = None
    # factories: Stirrup.rect / .circular / .diamond / .cross_tie

@dataclass(frozen=True)
class Cage:
    bars: tuple[Bar, ...]
    stirrups: tuple[Stirrup, ...]
    standard: "DetailingStandard | None" = None
```

**Hook bend-plane resolution (3D).**  A hook at a free end has an
incoming tangent `t̂` (from the parent polyline — *not* stored on the
Hook).  The bend plane is fixed at **bind time** by a normal `n̂`:
pick a turn target `p*` (default `"centroid"` ⇒
`g.model.queries.center_of_mass(host_label, dim=3)`; override = explicit
3-vector or `"up"`/world-axis token), form the in-plane direction
`û = normalize((I − t̂t̂ᵀ)(p* − anchor))`, then `n̂ = normalize(t̂ × û)`.
Degenerate-collinear (`‖û‖≈0`) falls back to a deterministic seed-axis
ladder (+Z then +Y), re-projects, and **warns at resolve time** — never
random.

**Stirrup = open path + two hooked tails.**  A real tie is *not* a
closed `curve_loop`.  It is an ordered open corner polyline that returns
near its origin with two 135° hooked tails overlapping at one corner.
The closure overlap is real geometry: two distinct free end-nodes at the
closure corner, with a `seam_stagger > 0` (default `max(db,
1.5·mesh_size)`) offsetting the second tail so OCC cannot weld the lap
into a single point.  Factories resolve clear-cover → centerline offset
(`cover + db/2` inward), corner radius, and expand `spacing` (with
hinge-zone densification) into a flat list of stations.

### §4 — DetailingStandard + BarCatalog (opt-in code intelligence)

```python
# apeGmsh/rebar/detailing.py   (NET-NEW; no units module exists today)
@runtime_checkable
class DetailingStandard(Protocol):
    name: str
    def bar_diameter(self, designation) -> float        # db, model units
    def bar_area(self, designation) -> float             # As (ASTM table for #N; π·d²/4 for raw/mm)
    def min_bend_diameter(self, db, *, kind="primary") -> float   # INSIDE diameter
    def hook_tail(self, angle, db) -> float
    def default_corner_radius(self, db, *, kind="primary") -> float  # scalar; NOT fed to add_arc
    def resolve_length(self, spec, db) -> float          # "4db" → 4·db ; float passthrough
    def make_hook(self, kind, db, **kw) -> Hook          # Raw() raises DetailingError

class Raw(DetailingStandard):           ...  # explicit-only: every code method raises DetailingError
class ACI318(DetailingStandard):        ...  # ACI 318-19 Table 25.3.1 / 25.3.2
class ACI318_seismic(ACI318):           ...  # §18.8.5 / §25.3.4 seismic 135° hoops
```

`bar_diameter`/`bar_area` are the **upstream producers** of
`ReinforceDef.bar_diameter`/`bar_area` (already live floats, validated
`>0`).  Area convention (Gate-A correction): an imperial `"#N"`
designation returns the **ASTM A615 nominal area** (the design area
engineers expect, e.g. #8 → 0.79 in²), while a metric `"<N>mm"`
designation or a raw-float diameter returns the geometric `π·d²/4`.
`ReinforceDef` stores diameter and area independently, so the `(db, As)`
pair stays ASTM-consistent (it need not satisfy `As = π·d²/4`).

Tables encoded (cited so the implementer can verify):

- **Min inside bend diameter** — ACI 318-19 Table 25.3.1: #3–#8 → 6db,
  #9–#11 → 8db, #14/#18 → 10db.  Table 25.3.2 (stirrups/ties): #3–#5 →
  4db, #6–#8 → 6db.
- **Standard hook tail** — primary §25.3.1: 90° → 12db, 180° →
  max(4db, 2.5 in).  **Non-seismic** stirrup/tie §25.3.2 is **bar-size
  dependent**: 90° and 135° → 6db (#3–#5) / 12db (#6–#8).  **Seismic**
  hook §25.3.4 is a flat 6db with a 3 in floor → max(6db, 3 in),
  independent of bar size (do NOT apply the 25.3.2 12db split to a
  seismic hoop).

Units: a single `unit_length` knob lives on `BarCatalog` (model length
per inch / per mm); the only place an absolute imperial constant enters
is scaling the 2.5 in / 3 in floors.  ACI bend/size buckets key off the
**bar diameter converted to inches** (`catalog.to_inches`) — unit-safe,
exactly equal to the size-number bucket for imperial designations, and
the documented nearest-equivalent for metric/raw diameters.
`DetailingError(ValueError)` is introduced in
`apeGmsh/rebar/detailing.py` (subclassing `ValueError` keeps existing
call-site `except` clauses working).

### §5 — Bend fidelity: metadata default, true-arc opt-in

- **Default (`true_arc=False`, `corner_radius="metadata"`).**  Bars and
  stirrups are straight-segment polylines; the bend radius / hook detail
  is carried as **metadata** on the `Cage` spec and the emitted curve
  PG.  The bend is a sharp vertex.  This is exactly the fidelity
  `ladruno_rc.py` already assumes (its `corotTruss` segments have no arc
  geometry), and it keeps the host mesh clean.

- **Opt-in (`true_arc=True`).**  Each bend becomes a true fillet: a
  straight lead-in line, one `add_arc(p_t1, center, p_t2,
  through_point=False)` (a 180° hook → **two** 90° arcs sharing a mid
  vertex, to dodge the shorter-arc ambiguity), and a straight tail line.
  Because `add_line`/`add_arc` take **point references, not
  coordinates**, the L2 builder must `add_point` the lead/center/tangent
  points first and compute `center` so that `‖center−p_t1‖ =
  ‖center−p_t2‖ = R` (rotate `p_t1` about `n̂` through `center` by the
  hook angle) — the equidistant invariant `add_arc` requires.  **Do not
  pass `label=` to `add_wire`** (it raises `ValueError`; the wire is a
  transient non-meshable OCC object).  Instead label/PG the member
  curves: `g.model.select([l_lead, c_arc, l_tail]).to_physical(name=…)`,
  then `g.labels.promote_to_physical` for a solver-facing PG.

- **Welding.**  A true-arc fillet leaves duplicate junction vertices.
  Run `g.model.queries.make_conformal(dims=[1])` **once per cage, early**
  (before any Part/Assembly build and before the chain-phase boundary),
  **never per-hook** — `make_conformal` renumbers OCC entities globally,
  so per-hook calls would dangle prior tag refs and corrupt a
  composed-Part cage.  After welding, re-resolve curve PG labels to fresh
  tags.

`true_arc=True` + `conformal` is fragile (the host mesh must conform to
a curved 1D entity, and is forbidden on a transfinite/structured host);
the recommended combination is `true_arc=True` + `embedded`.

### §6 — Coupling and the two binding surfaces

```
g.rebar.place(cage, into, *, coupling="conformal"|"embedded",
              per_member_coupling=None,
              # embedded-coupling forwarding (one regime per place(), applied to
              # every embedded member; bond XOR perfect required):
              bond=None, perfect=None, kt=None, kt_alpha=None,
              enforce="penalty", bipenalty=False, dtcr=None,
              tolerance=1e-6, snap=False,
              host_dim=None, true_arc=False,
              on_conformal_infeasible="fail"|"embedded",
              name=None) -> RebarPlacement
```

The embedded forwarding kwargs are a **single coupling regime per
`place()`** (all embedded members share one bond/perfect law); per-member
bond laws are a future extension (would add fields to `Bar`/`Stirrup`).
`host_entities`/`bars_entities` are intentionally not exposed — `g.rebar`
couples whole-label host/bars (drop down to `g.reinforce` directly for
sub-entity restriction). Validation is **Pass-0** (entire cage + host
checked before any gmsh mutation): unique member names, embedded bond-XOR-
perfect + resolvable db, the `element="beam"` gate, stirrup ≥3 distinct
corners, single + un-meshed + same-session host for conformal, PG host for
embedded.

`place()` runs at **geometry time** and calls `chain_phase_guard` at its
own entry (the `g.reinforce` sub-call is *not* guarded; only the
geometry/`embed` sub-calls self-guard).

**§6.1 Inline conformal.**  After labeling the bar curve PG,
`g.mesh.editing.embed(bar_pg, host_label, dim=1, in_dim=host_dim)` once
per (cage, host-volume) pair **before** `generate()`.  Host nodes then
lie on the bar curves ⇒ shared nodes ⇒ perfect bond.  No constraint def;
rebar elements emit at bridge time on the shared nodes.  `host_dim` is
auto-detected (dim=2 shell hosts allowed, not only dim=3).  **Works
under MPI.**

**§6.2 Inline embedded.**  Bars mesh independently; `place()` forwards to
`g.reinforce(host=into, bars=bar_pg, bond=…, perfect=…,
bar_diameter=<from L1, REQUIRED for bond>, bar_area=…, kt=…, kt_alpha=…,
enforce=…, bipenalty=…, dtcr=…, tolerance=1e-6, snap=False, …)` →
`LadrunoEmbeddedRebar`.  **Forward all `ReinforceDef` params** (do not
drop `bar_diameter`/`bar_area`/`bipenalty`/`dtcr`/`host_entities`/
`bars_entities`); derive per-member `bar_diameter` from each `Bar` spec.
**Single-process only today** — `apesees.py:1954-1966` raises on
partitioned `LadrunoEmbeddedRebar` emission (per-rank routing deferred,
fork ADR 20/R2).  So embedded is *not* unconditionally "better": for
partitioned models, conformal is the only working path.

**§6.3 Composed-Part (binding surface B).**  A cage saved as its own
Part / `model.h5` and stamped via `g.compose` is **embedded-only**,
declared at the assembly seam.  The resolved tie already has a
first-class compose-aware home: `ReinforceTieRecord` carries a working
`tag_rewrite_spec = {'tag_fields_scalar': ('rebar_node',),
'tag_fields_array': ('host_nodes',), 'name_fields': ('name', 'bond')}`,
so compose offset-rewrites the tie and re-prefixes the bond material
name.  **Do not invent a `/opensees/rebar/` sidecar or `RebarBarRecord`;
reuse (and, if needed, additively extend) `ReinforceTieRecord`.**

**The real blocker (gates the composed-Part path):**
`apeSees(fem).h5(path)` *drops* every `LadrunoEmbeddedRebar` tie —
`H5Emitter.embedded_rebar` is a deliberate no-op raising
`H5ReinforceDeviationWarning` (native round-trip deferred, fork ADR
20/R2; `emitter/h5.py:117-121, 900-907`).  `g.compose` is H5-source-only
(ADR 0038).  So a composed cage arrives today as **free-floating bars
with no host attachment** (warning at save, no error after compose).
ADR 0066 chooses **option (B)**: scope v1 composed-Part cages to
**Tcl/openseespy emit only**, and **raise** if a cage carrying reinforce
ties is saved for the H5-compose path — with option (A) as the named
follow-on (lift the R2 deferral: persist/read-back `ReinforceTieRecord`,
schema bump 2.19.0 → 2.20.0, forward-only).  **Inline cages are
unaffected** (they never round-trip through the lossy save).

**§6.4 Legality matrix + guards.**

| | inline | composed Part |
|---|---|---|
| conformal | ✅ (`embed`) | ❌ **raise** `ComposeUnsupportedError` (ADR 0038) |
| embedded | ✅ (`g.reinforce`; single-process) | ✅ assembly seam; v1 Tcl/py-only (H5 gated) |

Conformal-across-Part raises with an actionable message ("conformal
bond requires same-session authoring; use `coupling='embedded'`") —
reuse the existing `ComposeUnsupportedError`, do not mint a new name.
Mixed coupling (longitudinal conformal + hoops embedded) is legal
**inline** via `per_member_coupling`: conformal members fan out to
`embed()` calls, embedded members accumulate `ReinforceDef`s.  A shared
endpoint between a conformal bar and an embedded hoop yields two
coincident-but-distinct nodes — **do not auto-dedup** (that would
silently make the hoop conformal).  `on_conformal_infeasible` policy:
`"fail"` (default, raise naming the offending curve) or `"embedded"`
(auto-fall-back that member + one warning).

### §7 — Binding and realization (bridge-side)

- **db → area.**  `Bar`/`Stirrup` store diameter as the source of truth;
  area is derived `A = π·d²/4`.  Any `"<k>db"` string is resolved by the
  `DetailingStandard` at bind, *before* the bridge.  An unresolved
  `"<k>db"` surviving to the bridge is a fail-loud error.

- **Material by name.**  Every cage object references its material by
  **name** (str); the bridge resolves name → tag via the existing
  alias table `name_to_tag` (`apesees.py:4756`, populated at `build()`),
  exactly as `emit_reinforce_ties` does for the bond material.  There is
  **no** `bridge.materials.by_name()` — do not reference it.  A missing
  name raises the `build.py`-style "no primitive with that name…"
  error.  Name (not object/tag) is what makes the composed-Part path
  round-trip.

- **Element realization** is bridge-side, keyed by `Bar.element`
  (default `"truss"` ⇒ `CorotTruss`, matching `ladruno_rc.py`).  This is
  a **deliberate departure** from `g.reinforce` (which expects a
  user-declared `corotTruss` and emits only the coupling): the cage
  **auto-realizes** the bar element via the PG-fanned typed primitives
  (`Truss`/`CorotTruss` with `pg + A + material`, or `DispBeamColumn`
  with `pg + transf + integration`).

- **`element="beam"` is gated.**  The ADR-0010 Phase-4 orientation
  fan-out is **not implemented** (`transform.py` raises
  `NotImplementedError` when `orientation=` is set with `vecxz=None`).
  Therefore: straight single-tangent bars may use `beam` *with an
  explicit constant `vecxz`*; curved/hooked/bent bars with `beam` must
  **raise** a clear message ("beam rebar on curved segments requires the
  ADR-0010 orientation fan-out; use `element='truss'` or wait for Phase
  4").  **Ship truss-first.**

- **Bar-axis-twist stabilization** (beam only, solid host).
  `LadrunoEmbeddedRebar` couples *translations only* (ELE 33005), so a
  beam rebar's local-axis torsional spin is a zero-energy mode.
  Auto-emit a soft torsional `zeroLength` to a grounded ghost node,
  default-ON for `element="beam"` at unrestrained rotational DOFs;
  `element="truss"` (ndf=3) gets nothing, and a numeric
  `twist_stabilization` on a truss-only cage is rejected.  **Open
  caveat:** the synthetic ghost nodes + `zeroLength` need tag allocation
  through the canonical allocator and their own `(fem_eid → ops_tag)`
  H5 handling — not covered by `set_phantom_node_tags` (that is
  NodeToSurface-only).  Scope twist hardware to solid hosts first; its
  H5 round-trip is an Open Item, not assumed solved.

### §8 — Standardized members

L2 generators emit a `Cage`, then `place` it:

```python
# all keyword-only; section = ("rect", b1, b2); BarLayout/TieLayout carry the
# bar counts, diameters, spacing + hinge densification (no hinge_zones param):
g.rebar.column(*, section, height, cover, longitudinal: BarLayout,
               ties: TieLayout, base_z=0.0, origin=(0,0), standard=None,
               top_hook=None, bottom_hook=None, end_cover=None) -> Cage
g.rebar.beam(*, section, length, cover, top: BarLayout, bottom: BarLayout,
             stirrups: TieLayout, base_x=0.0, origin=(0,0), standard=None,
             end_cover=None) -> Cage
g.rebar.use_standard(std)
```

Hinge-zone tie densification comes from `TieLayout(hinge_spacing=,
hinge_length=)`. Bars/ties are inset interior (section faces by
`cover + tie + db/2`; member ends by `end_cover`, default `cover`) so the
flagship column meshes under **conformal** coupling without a boundary-facet
PLC error. They generalise `RCColumnSpec` off the grid.

**v1 detailing gaps (warned + Open Items):** only a single perimeter hoop is
generated — ACI 318 §25.7.2.3 **cross-ties / supplementary legs** for
intermediate bars (`n>2` per face) are not yet built (a warning fires). Hinge
densification is data-driven (not auto-derived from the seismic standard); a
warning fires when an `ACI318_seismic` column omits hinge params. Stirrup
closure twin-tail overlap is simplified to a single closure hook.

### §9 — Emission grain and chain-phase

Geometry is emitted **eagerly** inside `place()`/`column()`/`beam()`
(matching `g.sections`' eager `Instance` grain), **not** via the
pre-mesh validate hook — `_mesh_generation.py` iterates a *hardcoded*
tuple (`'loads'`, `'displacements'`, `'constraints'`, `'masses'`) plus
`model.geometry`, which `g.rebar` is not in.  Curves + PGs appear
immediately (inspectable); the `embed()`/`g.reinforce` forwarding is
also done eagerly at `place()` (both run pre-mesh; the chain-phase guard
already blocks post-snapshot mutation via `Model._register →
chain_phase_guard`).  Mutation guarding follows the established
`LoadsComposite._add_def` pattern (`try_chain_phase_route` +
`_bump_fem_counter`), not a bare `_check_chain_phase`.  `RebarPlacement`
(with `RebarMember` children carrying pg + resolved diameter/area +
material + element + coupling) is the declare-side record (curve PGs +
per-member coupling + spawned
`ReinforceDef` refs); element/material realization happens at bridge
time on resolved nodes.  **No H5 schema bump for the inline path**
(`ReinforceDef`/`EmbeddedDef` are pre-mesh declarations, never
serialized; conformal emits no def).

## Alternatives rejected

- **Reinvent the embedded emitter inside `g.rebar`.**  Rejected — the
  design workflow found `g.reinforce` already ships the entire
  `LadrunoEmbeddedRebar` path (resolver, inverse-map, `ReinforceTieRecord`,
  compose rewrite, bond-by-name).  Duplicating it would fork the
  coupling logic and the H5/compose records.  `g.rebar` forwards.

- **A new `/opensees/rebar/` H5 sidecar + `RebarBarRecord`.**  Rejected —
  `ReinforceTieRecord` is already the neutral-zone, compose-aware home;
  bar elements persist through normal `element_meta`.  Missing fields
  (db, role, hook id) are *additive* optional fields on the existing
  record, not a parallel dataset.

- **Extend `ladruno_rc.py` in the OpenSees tooling layer.**  Rejected —
  it is post-mesh openseespy with no geometry kernel and is intrinsically
  grid-conformal.  The cage layer needs OCC curves, labels, `embed`, and
  the compose machinery, all of which live in apeGmsh.

- **Default to true-arc bend geometry.**  Rejected as the default —
  meshing tight fillets distorts/refines the host badly (and breaks
  conformal embedding); the shipped `ladruno_rc.py` fidelity is sharp
  polyline corners.  True-arc is opt-in.

- **`coupling=` as a free flag everywhere.**  Rejected — conformal is a
  same-session `embed` operation that cannot cross a Part boundary.
  Conformal-across-Part raises; the composed path is embedded-only.

## Consequences

### Positive

- One authoring surface produces arbitrary, code-detailed cages and
  routes to *either* coupling, reusing the shipped `g.reinforce` /
  `embed` / `compose` machinery with no new emitter.
- `Cage` is serializable and reusable: author once, drop inline or
  `save()` as a Part-library module.
- No OpenSees C++ change; no H5 bump for the inline path — inline cages
  ship immediately.
- Generalises `RCColumnSpec` off the mesh grid; conformal cages even run
  under MPI.

### Negative (acknowledged)

- **Composed-Part embedded cages cannot persist to `model.h5` today**
  (the R2 deferral); v1 scopes them to Tcl/py emit and raises on the
  H5-compose path until option (A) lifts the deferral.
- **Embedded coupling is single-process only** (`LadrunoEmbeddedRebar`
  partitioned emission raises); partitioned models must use conformal.
- **`element="beam"` is partially blocked** by the unbuilt ADR-0010
  Phase-4 orientation fan-out: curved/hooked beam rebar raises;
  truss-first ships.
- **Auto-emitted twist-stabilization hardware** (ghost nodes +
  `zeroLength`) needs tag/H5 handling that is not yet wired — an Open
  Item, scoped to solid hosts.
- **`g.rebar` auto-realizes bar elements**, a departure from
  `g.reinforce`'s "coupling-only" contract; the two must be documented
  as complementary (author with `g.rebar`, or hand-declare bars + call
  `g.reinforce` directly).

### Neutral

- Adds one row to `_core.py` `_COMPOSITES` and a new `_kernel/defs/rebar.py`
  + `apeGmsh/rebar/detailing.py` package; no change to existing
  composites.
- `DetailingError(ValueError)` is a new error type (ledgered), caught by
  existing `ValueError` handlers.

## Open items (resolve before / during implementation)

1. **H5 persistence of `ReinforceTieRecord`** (option A) — the gate for
   composed-Part cages.  Schema 2.19.0 → 2.20.0, forward-only.
2. **Twist-stabilization tag + H5 handling** — ghost node / `zeroLength`
   allocation through the canonical allocator and `(fem_eid → ops_tag)`
   map.
3. **ADR-0010 Phase-4 orientation fan-out** — unblocks curved/hooked
   `element="beam"` rebar.
4. **`embed` blast radius for `make_conformal(dims=[1])`** — it fragments
   *all* dim-1 curves globally; document ordering vs the host volume's
   edges, or scope it.

## Phasing / implementation plan

- **P0** — L1 specs (`Hook`/`Path`/`Bar`/`Stirrup`/`Cage`) + `Raw()`
  detailing + `BarCatalog`, pure data, fully unit-tested off-session.
- **P1** — `RebarComposite` + eager geometry emission (polyline default)
  + `place(coupling="conformal")` inline via `embed`.  Generalises
  `ladruno_rc.py`; conformal column/beam smoke test.
- **P2** — `place(coupling="embedded")` inline forwarding to
  `g.reinforce`; mixed coupling; `on_conformal_infeasible`.
- **P3** — `ACI318` / `ACI318_seismic` standards + hook factories +
  `true_arc=True` fillet geometry + `make_conformal` welding.
- **P4** — standardized `column`/`beam` generators with seismic
  hinge-zone densification + L3 fluent `BarBuilder`.
- **P5** — composed-Part library path (gated on Open Item 1); `element="beam"`
  + twist stabilization (gated on Open Items 2–3).

## References

- `src/apeGmsh/core/ReinforcementsComposite.py` — `g.reinforce`
  (host/bars, bond-xor-perfect, resolve at `get_fem_data`).
- `src/apeGmsh/_kernel/defs/constraints.py` — `ReinforceDef`
  (`bar_diameter`/`bar_area`, `.diameter` derivation, Option-B layering).
- `src/apeGmsh/_kernel/records/_constraints.py` — `ReinforceTieRecord`
  + `tag_rewrite_spec` (compose-aware).
- `src/apeGmsh/opensees/element/embedded_rebar.py` — `LadrunoEmbeddedRebar`
  (ELE 33005), `embedded_rebar_args`.
- `src/apeGmsh/opensees/_internal/build.py:3160` — `emit_reinforce_ties`
  (name→tag via `name_to_tag`).
- `src/apeGmsh/mesh/_mesh_editing.py:52` — `embed(tags, in_tag, *, dim=0,
  in_dim=3)`.
- `src/apeGmsh/core/_model_geometry.py` — `add_point`/`add_line`/`add_arc`
  (point refs, equidistant invariant), `add_wire` (no `label=`).
- `src/apeGmsh/core/_model_queries.py:158` — `make_conformal` (global,
  renumbers).
- `src/apeGmsh/emitter/h5.py:117,900` + `mesh/FEMData.py:1795` — the
  reinforce-tie H5 deferral (R2).
- `src/apeGmsh/opensees/transform.py:166` — ADR-0010 Phase-4 orientation
  `NotImplementedError`.
- ADR 0022 (MP-constraint emission), 0035/0036 (`ASDEmbeddedNodeElement`),
  0038 (`g.compose`), 0041 (chain-phase routing).
- `OpenSees/Ladruno_scripts/ladruno_rc.py` — `RCColumnSpec` (the
  grid-conformal predecessor this generalises).
- Refined by design workflow `wf_7e15f74c-6e2` (14 agents; 7 dimensions
  designed + adversarially verified against `src/apeGmsh`).
