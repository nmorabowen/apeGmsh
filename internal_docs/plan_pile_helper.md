# Plan — a pile helper (`g.piles`) for pile–soil interaction

**Status: PLAN ONLY (2026-09-29). No code.** Written against `origin/main`
@ `c38eac97`. The inputs are the piles-validation ladder: the R1 verdict
(`ladder/R1_plumbing/VERDICT.md`, gaps G-1..G-9, numerics N-1..N-4, plan
changes P-1..P-4), its workaround builder `r1_model.py`, the R2 spec
(`ladder/R2_elastic/SPEC.md`, candidates A–E), the ladder plan, and the three
research reports of 2026-09-29 (`research_apegmsh.md`, `research_opensees_src.md`,
`research_literature.md`). Every path below is relative to `src/apeGmsh/`
unless it says otherwise.

## 1. The problem

A pile in a 3-D soil continuum takes about 250 lines of fragile hand geometry
in apeGmsh today. R1 needed seven workarounds to build one elastic pile with a
rigid skin and mortar contact, and none of them is something a user should have
to rediscover: the skin must be extruded one beam level at a time so every
ring is a geometric curve (G-5), the soil must be split into quadrant boxes so
the hole face can be meshed transfinite (G-6), the contact must be declared per
sector with its own radial `outward` (G-1, N-1, N-4), and the element-less skin
nodes need their rotations fixed by hand (G-2).

The literature says why the perimeter matters. PISA (Byrne, Burd et al. 2020)
shows that a large-diameter pile resists through four components: a
distributed lateral load p, a distributed moment m from shaft shear at radius
R, a base shear and a base moment. A beam on a line of soil nodes cannot carry
m. Every candidate the ladder compares, except E, therefore puts the pile's
real perimeter into the soil mesh, and each does it differently (R2 SPEC):

| ID | Pile | Coupling to soil | Soil mesh |
|---|---|---|---|
| A | beam | `kinematic_coupling` rings onto the hole perimeter at every beam level | conforming, one soil layer per pile segment (P-1) |
| B | beam | rigid skin (RBE2 rings) + `contact(mortar, tie=True)` per sector | free |
| C | beam | rigid skin + frictional mortar contact per sector, interference fit | free |
| D | solid (hex) | shared nodes, later tie or contact; RBE2 head | conforming or free |
| E | beam | `g.embed` beam nodes in a continuous soil, no hole | free |

The helper's job is to turn "a pile of diameter D and length L at (x, y), coupled
to `Soil` like B" into one call, and to name everything it makes so that the
results side can find the rings, the sectors and the tip without the user
keeping a tag list.

## 2. Scope: what the helper absorbs, what ships as standalone fixes

Most of R1's gaps are bugs or missing refusals in general-purpose verbs. They
bite anyone using `contact()` on a curved surface or `ops.fix` on two PGs, pile
or not. Burying their fix inside a pile helper would leave the general verbs
broken and hide the fix behind a pile-shaped API. So the rule is: a gap that is
wrong for every caller becomes its own small slice and issue; a gap that is
only "hard to author" is absorbed.

| Gap | What it is | Home | Slice |
|---|---|---|---|
| G-1 | `contact(tie=True)` takes one global `outward`; on a cylinder a single tie is silently 3.65× too stiff | **standalone**: refuse a single `outward` when the master's facet normals span more than 90°. The helper builds sectors anyway (P-2) | F4 |
| G-2 | element-less skin nodes inherit ndf 6; rotations singular | **standalone**: infer ndf = ndm for nodes referenced only by translational couplings and contact surfaces | F3 |
| G-3 | `geomTransf` under an ndf-3 builder raises a raw `OpenSeesError` | **standalone**: emit the transform under an ndf-6 builder, or refuse at build with a named message | F2 |
| G-4 | two `ops.fix(pg=)` sharing nodes emit two `fix` lines on one DOF | **standalone**: fold masks per node, as `fix_from_model` does | F1 |
| G-5 | skin rings must be geometric curves | **absorbed**: the helper extrudes per level | P2 |
| G-6 | an OCC cylinder face "has 1 hole" and refuses transfinite | **absorbed**: the helper cuts the hole in sectors | P2 |
| G-7 | `eps_n="auto"` on an element-less master aborts at run time | **standalone**: refuse `"auto"` at resolve when master facets have no solid owner | F5 |
| G-8 | NTS slave seam nodes land in two contacts, tractions added | **standalone**: de-duplicate slave nodes across contacts on one slave label | F6 |
| G-9 | no initial-gap adjust / interference | **absorbed now** (skin radius R + δ, geometric), then the fork flag (§3.7) | P2, P6 |
| N-1 | mortar pairs antipodal facets on a closed surface | absorbed by sectors now; fork pairing guard later | P2, P6 |
| N-2 | mortar Uzawa update at every commit, no opt-out | fork augmentation control, exposed when it lands | P6 |
| N-3 | frictional mortar needs εT = εN/10 | documented default in the helper's frictional spec (owner decision D3) | P3 |
| N-4 | NTS orientation on a coincident curved interface | sectors + half-facet phase offset, both helper parameters | P2 |

Two stale docstrings found by the research ride along with F4 (they sit in the
same files): `interface()` still says a 3-D interface "does not emit yet"
(`core/ConstraintsComposite.py:1149`), and `emit_mp_constraints` says kinematic
couplings emit `equalDOF` (`opensees/_internal/build.py:6028`).

Out of scope: the typed `PySimple1/TzSimple1/QzSimple1` family (research gap 1).
It is the BNWF baseline for R4, a plain bridge-primitive slice through the
bridge-feature guide, and it should be filed on its own. The helper does not
need it.

## 3. API design

### 3.1 Where it lives

Four homes were considered.

A **Part factory** (`PilePart`, stamped into the assembly) is ruled out by two
facts in the source. Part node maps are per-instance bounding boxes
(`core/_parts_registry.py:1231-1245`), so a pile Part nested in a soil Part
grabs every soil node inside the pile's box. And composed or `from_h5` sessions
refuse `contact()`, `interface()` and `embed()` (`raise_if_from_h5_session`,
`core/ConstraintsComposite.py:779`), because their frames need live geometry.
A Part could not carry the coupling that is the whole point.

A **`g.parts.add_*` builder** in the style of `add_absorbing_shell(box=…)` fits
"bring your own soil" but not the declaration of couplings; `g.parts` builds
geometry and returns a result record, and it does not call `g.constraints`.

A **function in a `geotech`/`ssi` module** (`place_pile(g, …)`) has no home for
session state (the per-pile name registry, the group's spacing check) and
breaks the composite convention every other session verb follows.

The fit is the **`g.rebar` precedent (ADR 0067)**: an L1 spec layer that is pure
data and never touches gmsh, and an L2 session composite that places the spec
into a host and *delegates* coupling to the existing verbs. A cage is authored
once and placed into a column; a pile is authored once and placed into a soil.
So:

- `apeGmsh.piles` (new package, L1): `Pile`, `PileGroup`, and the coupling specs.
  Frozen dataclasses, validated in `__post_init__`, JSON-serialisable, no gmsh
  import. They satisfy charter P11 (standalone instances) and P12 (typed, no
  `**kwargs`).
- `g.piles` (new `_COMPOSITES` row, `core/PilesComposite.py`, L2): `place()`
  and `place_group()`. It builds geometry through `g.model`, mesh controls
  through `g.mesh.structured`, names through `g.labels`/`g.physical`, and
  couplings through `g.constraints.kinematic_coupling`, `g.constraints.contact`
  and `g.embed`. It adds **no new record kind, no new FEMData stream and no
  bridge code**, so the `compose-streams` rule and the H5 schema are untouched
  (charter P3/P9: the bridge still sees only FEMData).

### 3.2 Signatures

```python
from apeGmsh.piles import Pile, PileGroup, Rings, Skin, Solid, Embedded, Tie, Frictional

pile = Pile(
    name="P1",
    head=(0.0, 0.0, 0.0),        # vertical pile, head point; axis is -z
    length=20.0, diameter=1.0,
    segment=0.25,                # beam node spacing; or levels=[z0, z1, ...]
    n_around=24,                 # skin quads / ring nodes around
    n_sectors=4,                 # contact sectors (>= 4 for a closed skin, P-2)
    phase=0.0,                   # sector/skin rotation; half a facet for NTS (N-4)
    tip="closed",                # "closed" = blind hole + tip disk; "open" = no tip coupling
)

placement = g.piles.place(
    pile, into="Soil",
    coupling=Skin(interface=Tie(eps=1e8), coupling_k=1e10, interference=0.0),
    hole_n_around=28,            # soil side; need not match the skin (B/C)
    hole_nz=None,                # None: soil vertical spacing left to the user's sizing
)

group = PileGroup.grid(pile, nx=2, ny=2, spacing=3.0, name="G1")   # names G1.P00 .. G1.P11
placements = g.piles.place_group(group, into="Soil", coupling=Skin(...))
```

`place()` returns a frozen `PilePlacement`: the PG names of §3.4, the level z
list, the `ContactDef`s and `KinematicCouplingDef`s it declared (so a stage can
claim them by name), and the skin/ring node-set labels a partitioner must keep
together (§6). The cap is a later `g.piles.cap(group, thickness=…, coupling=…)`
(slice P9); nothing in the first slices depends on it.

### 3.3 Coupling modes

| Spec | Candidate | What `place()` builds |
|---|---|---|
| `Rings(k=None, enforce="penalty")` | A | void cut as a stack of per-level sector tools, so the hole face has a curve at every beam level; hole patches transfinite with one element per segment (P-1 canonical mesh); one `kinematic_coupling(dofs=[1,2,3])` per level onto its hole ring; the tip disk to the tip node |
| `Skin(interface=Tie(eps))` | B | a free-standing skin (in no volume) extruded per level and per sector, welded at the seams so a ring has `n_around` nodes; one `kinematic_coupling` per ring; one `contact(formulation="mortar", tie=True)` per sector with its radial `outward`; the tip disk as its own contact with `outward=(0,0,-1)` |
| `Skin(interface=Frictional(mu, cohesion, tau_max, eps_n, eps_t, formulation="mortar"), interference=δ)` | C | as B; the skin at radius R + δ while the fork has no adjust flag (G-9) |
| `Solid(interface=None \| Tie \| Frictional, head="rbe2")` | D | a solid cylinder fragmented with the soil (shared nodes, `interface=None`) or kept separate with sector contacts; an RBE2 of the head face to a 6-DOF head node; a label per beam-level cross-section for section cuts (P8) |
| `Embedded(k=None, enforce="penalty")` | E | beam line only, no void; `g.embed(into, beam nodes)` |

The beam element, its section and transform, the soil material, loads and
fixities stay with the user on the bridge side (`ops.element.…(pg=placement.axis)`).
The helper is a model-authoring verb and has no business choosing an element.

### 3.4 Names it creates

Solver-facing PGs, the ones a user writes into `ops.*` calls, are few and flat.
Per-level and per-sector bookkeeping goes into labels (`g.labels`, the `_label:`
tier), which survive booleans, reach FEMData and are written to `model.h5`
under `/labels` (`mesh/_femdata_h5_io.py:787`). That is what lets the results
side (§4) find a ring without the helper writing a new H5 zone.

| Name | Tier | Dim | Modes | Holds |
|---|---|---|---|---|
| `P1` | PG | 1 | all but D | the beam centreline |
| `P1.head`, `P1.tip` | PG | 0 | all | head and tip nodes |
| `P1.skin` | PG | 2 | B, C | every skin facet |
| `P1.hole` | PG | 2 | A, B, C | the soil faces of the hole (shaft and bottom) |
| `P1.solid` | PG | 3 | D | the pile volume |
| `P1.level.{k}` | label | 0 | all | beam node k, k = 0 at the head |
| `P1.ring.{k}` | label | 1 | A, B, C | the ring coupled to level k |
| `P1.skin.sector.{s}`, `P1.hole.sector.{s}` | label | 2 | B, C, D | the sector pairs one contact binds |
| `P1.tip_disk` | label | 2 | A, B, C | the tip disk (closed tip) |
| `P1.section.{k}` | label | 2 | D | the cross-section at level k |

`k` is zero-padded to the level count so labels sort by depth. Contacts are
named `P1.contact.{s}` and `P1.contact.tip`, couplings `P1.kc.{k}`. A
`PileGroup` prefixes the group name (`G1.P01.ring.07`). The separator must be
checked against the label `safe_name` rule in the H5 writer before the ADR
fixes it (open question Q5).

### 3.5 Composing with the soil

The soil is the user's: a box, a layered stack, a DRM box or an absorbing
shell, built and labelled before `place()`. The helper never splits the soil
into layers for B, C or E, which is the point of those candidates: R1 ran B
with 13 soil layers against 8 pile segments.

What it does to the soil is limited to the hole. For A, B, C it cuts a void of
radius R (a blind hole to −L for a closed tip) as `n_sectors` wedge tools, so
the hole face arrives as `n_sectors` four-sided patches that mesh transfinite
(G-6). This replaces R1's quadrant boxes, which forced a fragmented soil on the
user. For A the tools are also stacked per level, so every level has a hole
curve for its ring. The helper sets transfinite counts on the hole patches
only; far-field sizing stays with the user's fields. Two things need care:

- `boolean.apply_voids(host)` subtracts *every* unapplied void tool from the
  host (`core/_model_boolean.py:296`), including the user's own. The helper
  must cut with its own tools only, either through `boolean.cut` or a small
  `tools=` filter on `apply_voids` (decided in P2).
- A spike in P2 must confirm that a cut by N sector tools yields N hole
  patches without splitting the soil volume. The fallback is R1's approach,
  fragmenting the soil with N half-planes bounded to a box around the pile.

The skin (B, C) and the beam line (A, B, C, E) are never fragmented with the
soil. A skin at radius R coincides with the hole face; a later
`g.parts.fragment_all` or boolean over the session would weld them and silently
turn B into A with shared nodes. `place()` records the skin entities, and its
resolve step fails loud if any skin node is shared with a soil element (§6,
risk 2). The same rule protects the beam line: a centreline fragmented into
the soil shares nodes with it (research G, "Gotchas").

### 3.6 Defaults the helper does not guess

The helper takes penalties explicitly. R1 found no value that is safe by
default: `LadrunoKinematicCoupling` stalls at its default k = 1e12 and works at
1e10; the mortar tie wants about 250 × E/h and frictional contact about 25 × E/h
with εT = εN/10 (VERDICT "Penalties that worked"). A default that is right for
one soil stiffness and mesh is wrong for the next, and `"auto"` is refused on
an element-less master (G-7). Whether `Frictional` should *default* εT to εN/10
is decision D3.

### 3.7 The fork's pile-contact flags (branch `wp/pile-contact-r05`)

The fork slice adds three opt-in mortar flags: an augmentation control (pure
penalty, which answers N-2), a pairing guard against antipodal facets (N-1), and
an initial-gap adjust / offset (G-9). The first helper slices do not depend on
them: sectors avoid antipodal pairing, a geometric interference replaces the
adjust, and B's step-count drift is small at ε = 1e8.

They reach users in two steps, and the first is not the helper. Slice P6 adds
them to `contact()` itself (`augment=`, `pairing_guard=`, `adjust=`), gated
exactly like `opensees/_rc_c2_flags.py`: a `…_MIN_BUILD = None` floor until the
fork PR merges into `ladruno`, the live route refusing and the Tcl/py emitters
warning, because a fork parser that ignores unknown tokens would drop them
silently (bridge guide, "Consume a fork flag only after…"). Then the helper's
`Tie`/`Frictional` specs grow the same fields and `Skin(interference=δ)` can
switch from geometry to `adjust=δ` without moving the skin.

## 4. Results side

The results↔viewer boundary (ADR 0014; viewer-results guide) puts extraction in
`results/`, reading only what `model.h5` and the results file carry, and keeps
viewers as pure consumers. The helper (`core/`) holds no results code. The link
between them is the naming contract of §3.4, persisted through `/labels`. No
H5 schema bump is needed for the first results slice.

**Pile profile (slice P4).** A function in `results/` (`results/_piles.py`,
exposed as `results.piles.profile("P1")`) returns a `PileProfile` of arrays
over the levels: z, u and θ at the beam nodes, N, V and M from beam
`localForce`, and the PISA reactions from **force jumps at the beam nodes**.
Because every coupling acts at a beam node, the jump in beam shear between the
elements on either side of node k is the lateral reaction at that level,
p_k = (V⁻ − V⁺)/ℓ_k with ℓ_k the tributary length, and the jump in beam moment
is the distributed moment, m_k = (M⁻ − M⁺)/ℓ_k. The tip node's end forces are
the base shear and base moment. So all four PISA components come from beam
forces that are recordable today (`results.elements.get("localForce")`), with no
coupling-force recorder. The lateral displacement y is the beam node
displacement minus the mean displacement of the level's ring (A) or of the
hole nodes nearest the level (B, C: the hole mesh does not match the levels, so
the mean is taken over hole nodes within half a segment of z). Hole-side means
are an explicit choice (`reference="ring" | "hole" | "far_field"`), because a
perfectly bonded ring has y ≈ 0 relative to itself (research_opensees_src §9.5).

**Coupling and contact forces (slice P7).** Per-node coupling forces and
`ladrunoContactForce` are needed to split the tip jump into shaft ring and tip
disk, and to map the shaft traction around the circumference. Today the
constraint-generated `LadrunoKinematicCoupling` elements get bridge tags outside
FEMData, so the typed recorders cannot target them (research F). P7 lets a
recorder target a constraint by name (`recorder.Element(constraint="P1.kc.*")`).
That touches the response catalog, a shared literal (AGENTS.md "How work
lands"), so it is its own bridge slice.

**Solid pile resultants (slice P8).** D needs N, V, M at each level from Gauss
stresses integrated over `P1.section.{k}`. There is no in-house integration
(`cuts/` is a spec producer for STKO_to_python). P8 adds
`results.section_resultants(label)` in `results/`; it is useful beyond piles.

**Views.** First a static `results.plot.pile_profile(profile)` (matplotlib, in
`results/plot/`), which is enough for the ladder's verdicts. A viewer diagram
over the same labels (the `IsochroneProfileDiagram` is the nearest relative) is
a later viewer slice under the viewers-change guide and is not planned here.

## 5. Slices in dependency order

Effort: S = one PR, under a day; M = a few days; L = a week or more. "Unit"
means gmsh + FEMData + deck text, no OpenSees. "Live" means the `ladruno_fork`
marker (auto-skipped off-fork), or `live-stock` where the path is stock.

### Standalone fixes (independent of each other and of the helper)

| Slice | Gap | Effort | Oracle |
|---|---|---|---|
| F1 ([#1259](https://github.com/nmorabowen/apeGmsh/issues/1259)) | G-4 fold `ops.fix(pg=)` masks per node | S | unit: two PGs sharing an edge emit one `fix` per node with the OR-ed mask; live-stock: the R1 boundary set runs |
| F2 ([#1260](https://github.com/nmorabowen/apeGmsh/issues/1260)) | G-3 `geomTransf` under ndf < 6 | S | unit: deck shows the transform inside an ndf-6 builder, or a named refusal at `build()`; live-stock: an ndf-3 envelope beam-in-brick model runs |
| F3 ([#1261](https://github.com/nmorabowen/apeGmsh/issues/1261)) | G-2 infer ndf = ndm for element-less contact/coupling nodes | M | unit: inferred ndf of skin nodes is 3; live fork: R1's B deck without the rotation fix is non-singular and matches 0.6955 mm |
| F4 ([#1262](https://github.com/nmorabowen/apeGmsh/issues/1262)) | G-1 refuse one `outward` on a master whose normals span > 90°; the two stale docstrings | S | unit: a cylinder master with `tie=True` and a global outward raises, naming the sector remedy |
| F5 ([#1263](https://github.com/nmorabowen/apeGmsh/issues/1263)) | G-7 refuse `"auto"` penalty on an element-less master | S | unit: raise at resolve, not at run time |
| F6 ([#1264](https://github.com/nmorabowen/apeGmsh/issues/1264)) | G-8 de-duplicate NTS slave nodes across contacts sharing a slave label | M | unit: resolved slave node sets are disjoint and their union is the label's node set |

F1, F2, F4 and F5 are one-file changes and can be dispatched in parallel.
F3 changes ndf inference, so it goes through the bridge-feature guide and is
checked field by field on a partitioned emit.

### The helper

| Slice | Content | Needs | Effort | Oracle |
|---|---|---|---|---|
| P0 | ADR (§6, Q1) fixing the L1/L2 split, the coupling vocabulary and the naming contract | — | S | review |
| P1 | `apeGmsh.piles` L1: `Pile`, `PileGroup.grid`, coupling specs, level computation | P0 | S | unit: levels, validation (n_sectors ≥ 4 for a closed skin, segment divides or `levels` monotone), JSON round-trip |
| P2 | `g.piles.place` for `Skin` (B, C): sector void, welded per-level skin, rings, couplings, sector and tip contacts, `interference`, labels | P1; F3 soft | M–L | unit: `n_sectors` shaft contacts plus one tip contact, each `outward` radial within 1e-12, no master facet outside its sector's cone, ring k has `n_around` nodes at z_k, no skin node shared with a soil element; live fork: R1's geometry through `place()` reproduces B u_head 0.6955 mm within 0.1 % and equilibrium ≤ 1e-8 |
| P3 | `Frictional` defaults and C documentation; `Rings` (A) with the stacked void | P2 | M | unit: every hole-perimeter node at level k is in ring k and in no other; live fork: R1's rigid-body check (translation and rotation) gives zero coupling force to 1e-9 |
| P4 | `results.piles.profile`: shear- and moment-jump p, m, base H and M, y by reference | P1 (names only) | S–M | unit on synthetic arrays: a beam with known point springs gives exact p; live fork: Σ p·ℓ + H_base = H to 1e-8 on B |
| P5 | `Embedded` (E) and `Solid` (D, shared nodes + RBE2 head + section labels) | P2 | M | unit: D volume shares nodes with soil only on the hole; live fork: D and B head stiffness within R2's band on the R1 block |
| P6 | Fork flags on `contact()` then on the specs (§3.7) | fork r05 merged into `ladruno` | S | unit: deck flags; live refused below the floor; live fork: pure-penalty B has no step-count drift (N-2) |
| P7 | Recorder targeting of constraint-generated coupling elements by name | — | M | unit: recorder resolves `P1.kc.*` to the right ops tags; live fork: summed ring forces equal the beam shear jump |
| P8 | `results.section_resultants(label)` for solid piles | P5 | M | unit: a uniform-stress block integrates to σ·A; live fork: D's M(z) against the beam candidates |
| P9 | `g.piles.place_group` spacing checks, then `g.piles.cap` | P2 | M, then L | unit: overlapping hole sectors refused; group names unique; live fork: a 2×2 group runs |

**What unblocks the ladder soonest.** R2 is already running on R1's workarounds
and should not wait for the helper; F1–F5 remove most of its hand-fixes and
cost a day each. R3 is blocked on the fork (N-1, N-2, G-9), so its apeGmsh path
is P6, which waits for the r05 merge. The helper pays off first at R4, where
layered nonlinear soil and refined Esmeralda meshes make hand-built skins
impractical: P1 → P2 → P4 is the critical path, about two weeks. R5 (u-p soil)
needs nothing new from the helper beyond `dofs=[1,2,3]` on the couplings, which
`place()` always passes (#1100). R6 needs P9.

Suggested order: F1, F2, F4, F5 in parallel → P0 → P1 → P2 (with F3) → P4 →
P3 → P5 → P6 when the fork lands → P7, P8 → P9.

## 6. Risks and open questions

1. **Partitioned emit and rings.** A kinematic coupling whose slave set spans
   two ranks fails loud (`opensees/_internal/build.py:10387`). For A the ring is
   soil nodes, so a partitioner that cuts through the pile breaks it. For B and C
   the ring is skin nodes with no elements, and it is not known which rank owns
   an element-less node. `PilePlacement` will expose the node sets that must
   stay together, and P2 must test a 2-rank `partition_explicit` build of B.
   Whether contact across ranks (ADR 0092) lifts the problem for B and C is open.
2. **The skin must never be welded to the soil.** A session-wide boolean after
   `place()` would merge the coincident skin and hole faces. The resolve-time
   check in §3.5 catches it; whether `place()` should instead refuse any later
   boolean that touches its entities is open.
3. **Sector tools.** P2's spike may show that OCC splits the soil volume when it
   cuts with N wedge tools. The fallback (bounded half-plane fragments) changes
   the user's soil topology, which the plan otherwise avoids.
4. **PG count in groups.** A 3×3 group with 80 levels and one label per ring and
   level is about 1,500 labels. Label resolution is per call, so `place_group`
   may need a batched coupling declaration. Measure in P9 before optimising.
5. **Label separator.** `.` is proposed; check the H5 writer's `safe_name` and
   the `g.labels` rules in P0.
6. **Results without a helper.** `results.piles.profile` depends only on the
   naming contract, so a hand-built model that follows it works too. The ADR
   should state that the contract, not the helper, is the interface.
7. **An ADR is warranted.** The slice adds a session composite, a cross-module
   naming contract that `results/` reads, and a coupling vocabulary. The next
   number on `origin/main` today is **0111** (`git ls-tree --name-only
   origin/main architecture/decisions/` ends at 0110, and no open PR adds a
   decision file). Re-list before writing it: numbers collide.

## 7. Decisions (owner, 2026-09-30)

- **D1. Accepted.** The `g.piles` composite plus the `apeGmsh.piles` L1 layer
  (the ADR 0067 pattern).
- **D2. Accepted.** F1–F6 are filed as issues now. R2 keeps its workarounds in
  the meantime.
- **D3. Both penalties explicit.** `Frictional` requires `eps_n` and `eps_t`.
  The docstring cites the R1 evidence that εT = εN/10 is the only ratio that
  converged there, and that the fork's own default εT = εN diverged. One
  problem is not enough to justify a default.
- **D4. Skin first.** P2 ships B/C; A follows in P3.
- **D5. Deferred to R6.** P9 is specified so that the cap can move to a shared
  foundation composite without breaking the `g.piles` API.
