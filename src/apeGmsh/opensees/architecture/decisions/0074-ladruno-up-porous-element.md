# ADR 0074 — `LadrunoUP` Biot u-p porous element emission (heterogeneous intra-element ndf)

**Status:** Proposed (2026-07-13). Companion runway to OpenSees-fork ADR-71
(LadrunoUP family v1 SHIPPED: fork PRs #551/#557/#559/#563/#566, ELE_TAG
33017). Design doc for the apeGmsh side — nothing here is built yet.

## Context

The Ladruno fork now ships one unified saturated-porous continuum element:

```
element LadrunoUP $tag $n1 … $nk $matTag
    <-thick $t>                              ;# 2D only, default 1.0
    -Kf $Kf -poro $n -rhoF $rhof             ;# REQUIRED (raw Kf — NOT quadUP's pre-combined bulk)
    -perm $k1 $k2 <$k3>                      ;# REQUIRED, k_hydraulic/γw per axis
    <-permH $k1 $k2 <$k3> -gammaW $gw>       ;# sugar pair (both or neither)
    <-alpha $biot> <-Ks $Ks> <-body …> <-fluidBody …>
    <-formulation std|bbar> <-pOrder equal|linear> <-lumped>
    <-stab auto <$a0> | off | $alpha>        ;# equal-order default: auto(0.25)
    <-dynSeepage on|off>                     ;# default OFF (fork P4 adjudication)
    <-geom linear>
```

`(ndm, nodeCount)` selects the shape: (2,3) T3 · (2,4) Q4 · (2,6) Bézier T6 ·
(3,8) H8 · (3,10) Bézier Tet10. The quadratic Bézier shapes are
**Taylor–Hood only** (`-pOrder linear` mandatory: quadratic u, vertex-linear
p) and require **straight sides** (mid-edge nodes at edge midpoints,
1e-6·edge-length guard, loud element deactivation otherwise).

The modelling surface has four properties no existing apeGmsh element has,
and they are exactly why hand-writing LadrunoUP models is painful — i.e. why
this emitter is worth building:

1. **The pressure DOF is honest**: nodal DOF `ndm+1` IS p (disp channel, not
   upstream's ∫p·dt vel-trick). Drained BC = fix that slot; prescribed head =
   `sp` on it; recorders read `disp`.
2. **Heterogeneous ndf *inside one element*** (Taylor–Hood): vertex/carrier
   nodes need `ndf = ndm+1`, mid-edge nodes need `ndf = ndm` — STRICTLY
   validated by the element (loud error either way). The hand-written idiom
   is an ugly two-step `model(..., ndf=ndm+1)` / `model(..., ndf=ndm)` dance.
3. **A mandatory general solver**: the honest-p tangent is unsymmetric; the
   no-`system`-command default (ProfileSPD) **silently drops a coupling
   block** and returns plausible garbage (measured p ~1e88 with rc=0). The
   fork guide's #1 danger box.
4. **Sharp flag legality** (unknown flags are parser-FATAL; `-stab` on TH is
   fatal; `-permH` requires `-gammaW`; quadratic shapes reject `-pOrder
   equal`), plus physics-critical defaults the user shouldn't have to
   remember (`-dynSeepage off`, `-stab off` for wave runs).

Authoritative fork-side references (read before implementing):
`Ladruno_implementation/LadrunoUP_guide.md` (user contract, all warnings),
`Ladruno_implementation/71_ladruno_up_family_adr.md` §3–§4 + §12 log,
`Ladruno_implementation/LEDGER_quirks.md` (five LadrunoUP rows), and the
executable spec: `tests/test_ladruno_up_element_th.py` (the TH modeling
dance + equalDOF mixed-ndf example), `tests/test_ladruno_up_init_recorders.py`
(init recipes), `tests/test_ladruno_up_element_analytic.py` (BC idioms).
The apeGmsh row to amend at ship time lives in the fork repo:
`Ladruno_implementation/ladruno_apegmsh_contract.md`.

## Decision

### D1 — one typed element class, shape from the mesh (LadrunoBrick/Quad precedent)

`ops.element.LadrunoUP(pg=..., material=..., Kf=..., poro=..., rhoF=...,
perm=(k1, k2, k3?), thick=None, permH=None, gammaW=None, alpha=None, Ks=None,
body=None, fluidBody=None, formulation="std", lumped=False, stab=None,
dynSeepage=None, geom=None)` in `element/solid.py` — one class, fan-out
driven, exactly like `LadrunoBrick`/`LadrunoQuad`/`BezierTet10`. The shape
comes from each fanned-out mesh element's type:

| mesh type | LadrunoUP shape | pOrder emission |
|---|---|---|
| tri3 | T3 | omitted (equal) |
| quad4 | Q4 | omitted (equal) |
| hexa8 | H8 | omitted (equal) |
| tri6 | Bézier T6 (TH) | `-pOrder linear` emitted automatically |
| tet10 | Bézier Tet10 (TH) | `-pOrder linear` emitted automatically |

The user never types `-pOrder`: it has exactly one legal value per shape at
v1, so the emitter owns it. Node ordering for tri6/tet10 follows the
`BezierTri6`/`BezierTet10` precedent already in `solid.py` (same Bernstein
node maps — verified in the fork at ADR-71 P3; the vertex-first,
edge-(0,1),(1,2),(2,0) / tet edge order (0,1),(1,2),(0,2),(0,3),(2,3),(1,3)
convention is shared).

**Kwarg policy: pass-through, no re-defaulting.** Every optional kwarg that
is `None` is simply not emitted — the fork parser's defaults (α=1, Ks≤0,
stab auto-on-equal-order, dynSeepage off) stay the single source of truth.
The typed class only *validates*; it never invents a value the parser would
not have picked. (This is why `formulation="std"`/`lumped=False` are still
emitted-if-non-default only.)

Construction-time validation (fail fast in Python, mirroring parser-fatal):
`permH`⇔`gammaW` come together and exclude `perm`; `thick` is 2D-only;
`stab` on a TH-shape pg is an error naming the fork rule ("TH is inf-sup
stable; `-stab` is parser-fatal on quadratic shapes"); `geom` accepts only
`"linear"`; `dynSeepage` accepts `"on"`/`"off"`/bool. `Kf`/`poro`/`rhoF`/
`perm` are REQUIRED (keyword-only, no silent physics).

### D2 — heterogeneous intra-element ndf (the new machinery)

ADR 0048/0049 inference currently lets an element class contribute **one**
ndf floor for all its nodes. LadrunoUP-TH needs a **per-node-slot** vector:

- Equal-order shapes: all slots `ndm+1` — works with today's machinery
  (`ndf_floor = ndm+1`, `ndf_ok = {ndm+1}`), nothing new.
- TH shapes: slots 0..2 (tri6) / 0..3 (tet10) contribute `ndm+1`; the
  mid-edge slots contribute `ndm`.

Extension: `_element_capabilities.py` grows an optional
`ndf_floor_per_slot: tuple[int, ...]` (absent ⇒ today's scalar behaviour —
fully additive, zero change for every existing element). The per-node
resolver takes the max over incident (element, slot) contributions, and
`ndf_ok` validation likewise consults the slot. Mid-edge nodes are never
shared with a vertex slot of a *conforming* neighbour tri6/tet10 (mid-edge ↔
mid-edge only), so the per-node result is well-defined on conforming meshes;
a mid-edge node also touched by, e.g., a beam would resolve through the
existing ∩ gate and fail loud — correct, that mesh is illegal for the fork
element too.

Deck emission then produces the `-ndf` tokens automatically (elided when
equal to the `ops.model(..., ndf=)` envelope, today's rule). Either envelope
choice works; the class docstring recommends `ndf=ndm+1` for saturated-only
models (vertex-heavy elision) and notes TH meshes will emit `-ndf ndm`
tokens on mid-edge nodes.

**Mixed dry/saturated regions** ride ADR 0069 (`equal_dof` mixed-ndf):
duplicated interface nodes + explicit-DOF-list u-ties — the fork guide §6
example (`tests/test_ladruno_up_element_th.py::test_iv_*` is the executable
version). No new mechanism; the ADR-0069 emission already handles it.

### D3 — straight-side pre-validation (fail before the deck does)

For tri6/tet10 pgs, the emitter checks every mid-edge node against its edge
midpoint with the fork's own tolerance (`‖x_m − (x_a+x_b)/2‖ ≤ 1e-6·‖x_b−x_a‖`)
and raises `BridgeError` naming element/node/distance. Rationale: the fork
guard fires per-element at `setDomain` with a loud message and a deactivated
element — but from openseespy that surfaces only as a cryptic `analyze()`
failure. Catching it at emission with mesh context (and the hint: "generate
order-2 meshes on straight geometry, or set the Gmsh high-order optimization
off") converts a runtime mystery into a build error. The check reuses the
`FEMData` connectivity — cheap, one pass.

### D4 — solver guard (the #1 footgun, promoted to a build gate)

At `build()`/emit time, if any `LadrunoUP` element is present and the deck's
`system` is missing **or** in the symmetric-storage set
{`ProfileSPD`, `SparseSYM`, `Mumps` without SYM=0 args}, raise `BridgeError`
citing the fork guide §2 (allowed serial: UmfPack / SuperLU / FullGeneral /
BandGeneral; MPI: Mumps SYM=0). The fork can only print a notice — apeGmsh
can actually stop the silent-garbage run. An `unsafe_solver_ok=True` escape
hatch on the element class is deliberately NOT provided; the divergence test
measured 89 orders of magnitude, there is no legitimate use.

### D5 — fork gating, capabilities, persistence

- `LadrunoUP` joins `_FORK_ONLY_ELEMENTS` (deck emission works anywhere;
  `ops.run()` on stock openseespy fails loud), and the capabilities probe
  learns the element name (fork banner lists "LadrunoUP" since P4).
- Persistence rides the standard typed-element bridge-zone payload (kwargs
  round-trip like `LadrunoBrick`'s); no neutral-schema change expected —
  confirm at implementation, bump bridge-zone only if a new dtype column is
  genuinely needed.
- `ops.capabilities()` gains a `ladruno_up` feature key so scripts can
  branch.

### D6 — BC/recorder ergonomics (thin, v1)

No new composites. The class docstring + guide crosslink document the
idioms, because they are one-liners once ndf is right:

- drained set: `ops.fix(pg="Top", dofs=(0, 0, 1))` — dofs length = that
  node's resolved ndf (the existing per-node fix machinery already sizes
  this; verify it consults resolved ndf, not the envelope, and fix if not —
  in scope for this runway).
- prescribed head: standard `sp` inside a pattern; docstring warns
  `constraints('Penalty')` for *staged* heads (fork quirks row:
  Transformation converges to a wrong steady state) and `NormDispIncr` (not
  `NormUnbalance`) under Penalty.
- p read-back: `recorder Node -dof <ndm+1> disp` / `nodeDisp` — plus the
  fork's `.ladruno` recorder PRESSURE channel is already contract-aware
  (disp-slot for 33017 nodes), so `recorder ladruno -N pressure` works
  unchanged through the existing ADR-0064 translate path.

## Scope / deferred

- **Init-recipe staged verbs** (`ops.stage` sugar for the gravity /
  steady-seepage / setNodeDisp-commit recipes, incl. the `ops.reset()`
  sequencing trap) — deferred; the recipes are documented and testable via
  plain verbs. Revisit after first real project use.
- **Results-side pore-pressure convenience** (a `results.pressure(...)`
  accessor unifying disp-slot LadrunoUP p with vel-slot upstream-UP p) —
  deferred to the Results runway.
- P5 shapes (Q9/H20), hybrid mode 2, explicit u-p / ADR-73 overlay — track
  the fork; the fan-out table grows a row per shape when they land, nothing
  structural.
- `-stab`/wave-run advisories beyond docstrings (e.g. warning when a
  LadrunoUP model has a `UniformExcitation` and stab unset) — nice-to-have,
  not v1.

## Testing

- **Unit (emission)**: kwarg→token mapping incl. pass-through elision;
  legality errors (permH/gammaW pairing, stab-on-TH, thick-in-3D); tri6/tet10
  auto `-pOrder linear`; per-slot ndf resolution + `-ndf` token
  emission/elision both envelope choices; straight-side BridgeError on a
  curved tri6; solver guard on ProfileSPD/absent system.
- **Integration (runnable deck, fork-gated via `OpenSeesTarget`)**: a mini
  Terzaghi column (Q4 lane) emitted → run → p decay sane vs the series
  (loose gate — the tight gates live in the fork repo); a 2-element BT6-TH
  patch emitted with the two-ndf dance → runs rc=0 and matches the
  fork-side hand-written twin byte-for-byte on the element/node lines
  (deck-diff test, the strongest cheap gate).
- Update `Ladruno_implementation/ladruno_apegmsh_contract.md` (fork repo)
  with the LadrunoUP row in the same PR that ships this.

## Consequences

Consolidation / staged-construction / liquefaction models enter the normal
apeGmsh mesh→snapshot→deck pipeline: pg-driven fan-out replaces hand-written
node loops, the TH ndf dance disappears into inference, and the two
silent-failure landmines (wrong solver, curved TH mesh) become build errors.
The per-slot ndf extension is the one piece of genuinely new machinery, and
it is additive — every existing element keeps the scalar path.
