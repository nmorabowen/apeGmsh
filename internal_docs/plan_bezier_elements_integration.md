# Plan — Bézier elements integration (`ops.element.BezierTri6` + `ops.element.BezierTet10`)

**Status:** proposed (2026-06-01) · **Owner:** nmora · **Scope:** apeGmsh-side
typed-primitive + result-read support for the two Ladruno-fork Bézier continuum
elements.

This is the apeGmsh half of contract item **#3 (BezierTri6)** and **#4
(BezierTet10)** from the fork's apeGmsh-facing contract
(`nmorabowen/OpenSees@origin/docs/apegmsh-feature-recs:Ladruno_implementation/ladruno_apegmsh_contract.md`).
The fork-side elements are **shipped + validated** (`ladruno` PR #6 for Tri6; the
Tet10 sibling element + recorder wiring merged on `ladruno`). This plan is what
apeGmsh builds to make them **first-class typed primitives** and to read their
results. ADRs: `ladruno:Ladruno_implementation/04_bezier_elements.md` (Tri6) and
`…/06_bezier_tet10.md` (Tet10, carries **O11**).

> **Why a separate plan.** These elements ride the **same self-describing
> `B(ξ; FAMILY, ORDER)` basis seam** as the Ladruno recorder
> (`plan_ladruno_recorder_integration.md`, L2). That plan's basis lib is the
> dependency; this plan does **not** duplicate it. The element-registration work
> here is otherwise a near-mechanical mirror of the existing `SixNodeTri` /
> `TenNodeTetrahedron` typed primitives.

---

## Direct-drive already works — this is the ergonomic upgrade only

**The two elements run on apeGmsh meshes today** via *direct-drive* (fork
`bezier_apegmsh_integration.md`, validated 2026-05-30):

1. py3.11 + apeGmsh meshes a straight-sided domain to T6 / T10, extracts `fem`,
   dumps `nodes` + `fem.elements.<group>.connectivity` to JSON;
2. py3.12 + the fork build reads JSON and calls
   `ops.element('BezierTri6', eid, *conn6, thick, type, matTag)` (resp. `'BezierTet10'`).

Gmsh's native `tri6`/`tet10` node order is **byte-identical** to the element's
control-point order (Tri6 verified to `1.78e-15` on a real mesh; Tet10 D2′ was
*constructed* to match Gmsh — see O11). Connectivity rows are usable **verbatim**.

**Consequence for phasing:** direct-drive is the documented fallback and needs
**zero** apeGmsh change. Every phase below is opt-in ergonomics — none of it
blocks any user who direct-drives today. The plan keeps direct-drive documented
through the final phase (B5).

---

## Governing constraints (non-negotiable)

1. **Fork is opt-in; vanilla never breaks.** apeGmsh keeps running on stock
   `openseespy`. `BezierTri6`/`BezierTet10` exist only in the fork build.
   - *Emit:* `ops.element.BezierTri6(...)` / `ops.tcl` / `ops.py` produce deck text
     on **any** build — the line is just `element BezierTri6 …`. Gate the fork
     requirement **at `ops.run()`**, with a clear *"requires the Ladruno fork
     build"* error. Never force the fork; never fail at import.
   - *Read:* result parsing needs only `h5py` (no fork at read time); it keys on
     the element's self-declared `FAMILY=bernstein` / class name in the file.
2. **Class tags read LIVE, never hardcoded.** The fork moved tags to a private
   **≥33000 band**; the old sub-300 values (272/273) are **dead**. Confirmed live
   from `ladruno:SRC/classTags.h`:
   - `ELE_TAG_BezierTri6  = 33000`
   - `ELE_TAG_BezierTet10 = 33001`
   The apeGmsh `_response_catalog` must carry **these** values (still hand-entered,
   as every other `ELE_TAG_*` constant in that file is — but sourced from the live
   header, with the dead 272/273 explicitly rejected in a comment). The bezier
   docs (`04`/`06`/`bezier_apegmsh_integration`) still cite 272/273 in prose — do
   **not** trust those; trust `classTags.h`.
3. **Canonicity — read-direct, never transcode.** Result reads interpret the
   fork's own descriptors (the `FAMILY=bernstein` basis + `QUADRATURE` the element
   self-declares); apeGmsh re-encodes nothing to a derived/cached on-disk form.
   Geometry reconstruction `x(ξ)=Σ Bᵢ(ξ)·Xᵢ` uses the file's BASIS — the file
   dictates the map.
4. **Simplicity first.** Each typed primitive is a ~50-line frozen dataclass
   mirroring `SixNodeTri` / `TenNodeTetrahedron`. No new abstraction: B-bar is a
   `bbar=` flag (mirrors the fork's `-bbar`, not a second class), cMass a
   `consistent_mass=` flag. No speculative curved-geometry / NURBS surface (the
   element is v1 straight-sided only).

---

## What's *different* from `SixNodeTri` / `TenNodeTetrahedron` (read before mirroring)

The existing `SixNodeTri`/`TenNodeTetrahedron` typed classes are the templates, but
the bezier wrappers diverge in three small ways:

- **The optional tail is the fork grammar, not the quad/tri31 tail.** Fork factory
  (`ladruno:SRC/element/bezierTriangle/OPS_BezierTri6.cpp`):
  `element BezierTri6 tag n1..n6 thick type matTag [-bbar] [-cMass] [-pressure p] [-rho r] [-bodyForce b1 b2]`.
  These are **flag-prefixed** (`-bbar`, `-cMass`, …), **not** the positional
  `<pressure rho b1 b2>` tail of `quad`/`tri31`. So do **not** reuse
  `_quad_tri_optional_tail`; emit flags explicitly (each independently optional —
  no all-or-none ordering rule).
- **B-bar validity differs by element (mirror the fork guards D5 / D5′).** Tri6:
  `-bbar` valid PlaneStrain/3D only — fork *warns+disables* on PlaneStress.
  Mirror with an apeGmsh-side construction-time guard (raise or warn). Tet10:
  `-bbar` always valid (pure 3D, no plane-stress degeneracy).
- **Tri6 `plane_type` is 2-value, not `SixNodeTri`'s 4-value set.** The fork factory
  `OPS_BezierTri6.cpp:97-101` validates **only** `{PlaneStress, PlaneStrain}` and
  errors otherwise; `SixNodeTri` also accepts `PlaneStress2D`/`PlaneStrain2D`. Give
  `BezierTri6` its **own** 2-value validator (don't inherit the 4-value one) so a
  `*2D` string fails at construction, not late at run. This validator is also the
  natural host for the D5 PlaneStress-+-B-bar guard.
- **Type token == class name** (unlike `quad`/`tri31`/`tri6n`). The fork factory
  token is literally `BezierTri6` / `BezierTet10`, so **no** `_CLASS_TOKEN_ALIASES`
  entry and **no** `cpp_class_name` override are needed (the registry key *is* the
  token *is* the C++ class name).

Everything else mirrors the templates: `current_element_nodes` fan-out,
`resolve_tag` for the material, `_ElemSpec` capability flags, the `_continuum_layout`
response-catalog rows.

---

## Seam map (apeGmsh files to touch)

Grounded against current `src/`. Both elements touch the **same six seams**; the
table lists the Tri6 anchor and the Tet10 delta.

| Seam | File:line | Mirror of | Change |
|---|---|---|---|
| Element dataclass | `src/apeGmsh/opensees/element/solid.py` (`SixNodeTri` 374–458; `TenNodeTetrahedron` 117–154) | `SixNodeTri` / `TenNodeTetrahedron` | New frozen `BezierTri6` (6 nodes, `thickness`, `material`, `plane_type`, `bbar=False`, `consistent_mass=False`, `pressure`, `rho`, `body_force`) emitting token `"BezierTri6"` with **flag-prefixed** tail; new `BezierTet10` (10 nodes, `material`, `bbar=False`, `consistent_mass=False`, `body_force`) emitting `"BezierTet10"`. **No** `_quad_tri_optional_tail` reuse. |
| `__init__` export | `src/apeGmsh/opensees/element/__init__.py` (`from .solid import …` 48–55; `__all__` 81–86) | the solid-family export block | Add `BezierTri6`, `BezierTet10` to the `.solid` import and `__all__`. |
| Namespace factory | `src/apeGmsh/opensees/_internal/ns/element.py` (`SixNodeTri` 414–…; `TenNodeTetrahedron` 338–350; solid import block 26–33) | the two `_ElementNS` methods | New `_ElementNS.BezierTri6(...)` / `.BezierTet10(...)` → `_resolve(material, base=NDMaterial)` + `_register`. Exposes `ops.element.BezierTri6(pg=…, thickness=…, type=…, material=…, bbar=…)`. |
| `_ElemSpec` capability | `src/apeGmsh/opensees/_element_capabilities.py` (`tri6n` 243–251; `TenNodeTetrahedron` 186–192) | the `tri6n` / `TenNodeTetrahedron` registry entries | New `"BezierTri6"`: `mat_family="nd"`, `needs_transf=False`, `ndm_ok={2}`, `ndf_ok={2}`, `gmsh_etypes={9}`, `node_reorder={9:(0,1,2,3,4,5)}` (identity), `slots=("nodes","thick","eleType","matTag")`, `has_gauss=True`. New `"BezierTet10"`: `ndm_ok={3}`, `ndf_ok={3}`, `gmsh_etypes={11}`, `node_reorder={11:(0,1,2,3,4,5,6,7,8,9)}` (identity), `slots=("nodes","matTag")`, `has_gauss=True`. **No** `cpp_class_name` (token == class). **`slots` drives positional fan-out only** — `body_force` is **flag-prefixed** (`-bodyForce`), not a positional tail, so it is **not** a slot (drop the positional `bodyForce` copied from `TenNodeTetrahedron`); emit it from the dataclass field. Same rule for Tri6. |
| Response catalog | `src/apeGmsh/opensees/_response_catalog.py` (`ELE_TAG_*` consts 102–115; `SixNodeTri` rows 818–841; `TenNodeTetrahedron` rows 697–708) | `SixNodeTri` (Triangle_GL_2) / `TenNodeTetrahedron` (Tet_GL_2) rows | Add `ELE_TAG_BezierTri6 = 33000`, `ELE_TAG_BezierTet10 = 33001` (live from `classTags.h`; comment the dead 272/273 **and** note these values are *reused across class families* — `INTEGRATOR_TAGS_ExplicitBathe`/`RECORDER_TAGS_EnergyBalance` are also 33000 — so they disambiguate only within their own `*_TAG_*` namespace; don't "correct" them to a unique global value). The catalog rows supply **component layout / `n_gp` / `coord_system` metadata ONLY** — see the **Tri6 GP-order caveat** below. Register both under their real `IntRule` (`Triangle_GL_2` / `Tet_GL_2`) **and** `IntRule.Custom` (1000) — the Ladruno recorder serves these via the `basisInfo` self-declaration / `CustomIntegrationRule` path, exactly like the existing `SixNodeTri` Custom mirror rows (818–841). |
| Flags (`-bbar` / `-cMass`) | (inside the dataclass `_emit`, `solid.py`) | — (new) | Emit `-bbar` / `-cMass` from the `bbar` / `consistent_mass` fields. Tri6 `__post_init__` mirrors fork **D5**: `bbar=True` + `plane_type` PlaneStress → warn + drop the flag (or raise). Tet10: no guard (**D5′**). |
| Docs/skill | **canonical** `skills/apegmsh/references/` (`ladruno.md` + `api-cheatsheet.md` element-type table ~`:544`) — **NOT** the `.claude/skills/apegmsh-helper/` mirror (it's derived via `sync_skill.py` + CI `--check`; edits there get overwritten) | the bezier contract rows | Document `ops.element.BezierTri6/BezierTet10`, the fork-build run requirement, and that **direct-drive remains the fallback**. Let `sync_skill.py` regenerate the mirror. |

> **No `node_reorder` consumer beyond storage.** apeGmsh stores `getElementsByType`
> connectivity **verbatim** in `ElementGroup` (`_element_types.py`); `node_reorder`
> for etypes 9/11 is identity, so the fan-out passes Gmsh order straight through.
> This is the whole basis of O11 being clean (below).

> [!warning] **Tri6 GP-index-order caveat — the file is the sole source of per-GP
> coordinates.** The fork `BezierTri6.cpp` integrates its 3 GPs in a **different
> index order** than `SixNodeTri`'s `_TRI_GL_2_COORDS`: fork GP order is
> `{1/6,1/6},{2/3,1/6},{1/6,2/3}` (GP0 near corner 3) vs the catalog's
> `{2/3,1/6},{1/6,2/3},{1/6,1/6}` (GP0 near corner 1) — *same three points, permuted
> indices*. So **B4 must take per-GP coordinates from the file's `QUADRATURE/GP_PARAM`
> (the canonical read-direct path), never from a GP-index→catalog map** — the latter
> would silently permute each Tri6 GP's reported `(x,y)`. The catalog row is
> layout/`n_gp`/`coord_system` metadata only. **Tet10 is clean** —
> `BezierTet10.cpp`'s GP4_L order `(a,b,b)/(b,a,b)/(b,b,a)/(b,b,b)` matches
> `_TET_GL_2_COORDS` exactly — so this is Tri6-only. If a catalog-coords fallback is
> ever kept for files lacking `GP_PARAM`, the Tri6 row needs **its own** coords array
> in the *fork's* GP order, not a borrow of `_TRI_GL_2_COORDS`.

---

## O11 — tet10 node-order reconciliation (the load-bearing risk) — **argued identity, LOCKED BY B2 TEST (not yet live-run-confirmed)**

The fork ADR flags O11 as the #1 correctness risk: a wrong mid-edge order silently
yields a valid-looking but **wrong** stiffness. Three orderings, laid out as
vertex-pair edge labels (`a-b` = the edge whose endpoints are corner nodes a,b),
corners 1-4 first in all three:

**(a) Gmsh native `tet10` (MSH element type 11).** The Gmsh second-order tetrahedron
convention: 4 corner vertices, then 6 mid-edge nodes in edge order
**(1-2), (2-3), (1-3), (1-4), (3-4), (2-4)** → `N5=(1-2) N6=(2-3) N7=(1-3) N8=(1-4)
N9=(3-4) N10=(2-4)`.

**(b) Element D2′ (`06_bezier_tet10.md`).** 4 vertices, then
`N5=(1-2) N6=(2-3) N7=(1-3) N8=(1-4) N9=(3-4) N10=(2-4)`. The ADR states D2′ is the
`TenNodeTetrahedron::shp3d` order **"deliberately N9↔N10-swapped to match Gmsh."**

**(c) What apeGmsh's `get_fem_data` emits for tet10.** `_element_types.py:34` maps
gmsh code 11 → `'tet10'`; connectivity is stored **verbatim** from gmsh
`getElementsByType` (no permutation in storage). The bridge fan-out applies
`_ElemSpec.node_reorder[11]`, which for `TenNodeTetrahedron` is the **identity**
`(0,1,2,3,4,5,6,7,8,9)` (`_element_capabilities.py:190`). So apeGmsh emits the **raw
Gmsh tet10 order = (a)**, unpermuted.

### Verdict

**All three MATCH — the permutation is the identity.** (a) = (b) = (c).
`BezierTet10` shares `TenNodeTetrahedron`'s node order byte-for-byte (the fork ADR's
D2′ is defined as exactly that), apeGmsh already trusts that identity for the
shipped, validated `TenNodeTetrahedron`, and the registry encodes it as identity.
**Therefore `BezierTet10` reuses `node_reorder={11:(0,...,9)}` unchanged** — no
permutation map, no remap of `fem.elements.connectivity`.

> [!warning] **This is read off a triple-source argument, not a live gmsh run.**
> The identity verdict rests on: the Gmsh MSH-type-11 convention (a), the fork ADR's
> explicit "constructed to match Gmsh" claim (b), and the existing identity entry for
> `TenNodeTetrahedron` (c). The Tri6 sibling was *live-confirmed* to `1.78e-15`;
> Tet10 has not been on an apeGmsh mesh yet. The fork itself still lists O11 as
> *unconfirmed against a live run*. **Lock it with the test below — keep B2 a HARD,
> non-optional gate; do not downgrade it on the strength of the triple-source
> argument alone.**

> [!note] **Vertex winding / Jacobian sign is a non-issue by construction (not by
> the B2 test).** The midpoint-only B2 test proves edge-node *labelling* +
> straight-sidedness but is blind to corner-vertex winding (a flipped tet has every
> mid-edge still at its midpoint yet a negative `detJ`). That half is covered not by
> a test but by the **element itself**: `BezierTet10.cpp` integrates `K`/`M`/`F` with
> `fabs(detJ)` (orientation-robust, per the ADR's O11 hedge) — `BᵀDB` / `NᵀN` /
> `J⁻¹` are orientation-independent — so winding cannot flip the stiffness.
> **Therefore midpoint-only B2 is sufficient; do NOT add a redundant signed-volume
> assertion.**

### The locking test (B2 gate — round-trip mid-edge-at-midpoint)

A `tests/` test that, for a straight-sided T10 tet mesh:

1. mesh a single straight-sided tet (or a small block) to `tet10`, extract `fem`;
2. for each element, take the verbatim 10-node connectivity apeGmsh emits, fetch the
   10 node coordinates;
3. assert **each of the 6 mid-edge nodes (positions 5-10) sits at the midpoint of
   the corresponding corner-pair edge** under the **(a)/(b)/(c) order**
   `N5=mid(1,2) N6=mid(2,3) N7=mid(1,3) N8=mid(1,4) N9=mid(3,4) N10=mid(2,4)` —
   `max‖X_mid − ½(X_a+X_b)‖ < 1e-12·edge_len`.

A pass confirms ordering **and** straight-sidedness simultaneously (exactly how the
Tri6 integration test caught it at `1.78e-15`). A fail on any specific mid-edge
position points straight at the offending edge in the permutation.

> [!note] **One-line live check (optional, not a blocker).** If a definitive
> machine-precision confirmation is wanted before B2, run in a gmsh session:
> mesh one straight tet to order 2, `g.mesh.queries.get_fem_data()`, then the
> midpoint assertion above. This is the same one-element check the Tri6 path used;
> it does not require the fork build (pure gmsh + apeGmsh on py3.11).

---

## Basis library dependency (do NOT duplicate)

Result reconstruction for the bezier elements needs `B(ξ; bernstein, 2)` on the
triangle (tri6-bernstein) and tetrahedron (tet10), to evaluate `x(ξ)=Σ Bᵢ(ξ)·Xᵢ`
at the GPs the file declares. **This is the identical `B(ξ; FAMILY, ORDER)` library
the Ladruno recorder plan defines** (`plan_ladruno_recorder_integration.md`, seam
"Family basis lib", L2): its per-family list **already includes `tri6-bernstein`
and `tet10`** precisely to de-risk #3/#4.

**Recommendation: the basis lib lives NEUTRAL, and the bezier work imports it — no
second copy.** The recorder plan's open question ("reader-local
`results/readers/_ladruno_basis.py` vs neutral `src/apeGmsh/_basis.py`") should be
**decided neutral** *because* of this dependency: two independent consumers (the
`.ladruno` reader and any bezier-result read path) need the same `B(ξ)`. A
reader-local module would force the bezier path to either import across the reader
package boundary or duplicate the math. Concretely:

- **Flag to the recorder plan (L2/L5):** resolve its basis-lib OQ in favor of a
  neutral `src/apeGmsh/_basis.py` and have `_ladruno*.py` import it. This plan then
  **imports the same module** for bezier GP reconstruction. No duplication.
- If the recorder plan ships first with a reader-local lib, this plan's read phase
  (B4) **promotes** it to neutral rather than copying it.

apeGmsh's read of bezier results is otherwise **canonical / read-direct**: the
element self-declares `FAMILY=bernstein` + `QUADRATURE` into the file (fork
recorder, already wired), so the reader needs **no per-class shape-function table** —
just the neutral `B(ξ; bernstein, 2)` and the file's own descriptors.

---

## Phased delivery

Each phase is independently shippable (its own PR), verifiable, ordered so the
ergonomic value (typed primitive + deck emission, fork-free) lands before the
fork-dependent read work. **Direct-drive stays the documented fallback throughout.**

### B1 — `ops.element.BezierTri6` typed primitive + deck emission  *(no fork, no fixture)* — ✅ DONE
- `BezierTri6` dataclass (`solid.py`, + `BezierBBarPlaneStressWarning`) +
  `_ElementNS.BezierTri6` (`ns/element.py`) + `element/__init__` export +
  `_ElemSpec "BezierTri6"` (etype 9, identity reorder, no `cpp_class_name` —
  token==class) + `_response_catalog` rows (`ELE_TAG_BezierTri6 = 33000`,
  `Triangle_GL_2` + `Custom`; new `_TRI_GL_2_COORDS_BEZIER` in the fork's GP order).
  Flag-prefixed tail `-bbar`/`-cMass`/`-pressure`/`-rho`/`-bodyForce`; own 2-value
  `_PLANE_TYPES_BEZIER_TRI6` validator (rejects the `*2D` spellings); D5
  PlaneStress-B-bar guard = **warn + drop** (decided OQ).
- Fork requirement bites at `ops.run()`, **not** at emit — emission is just a
  `element BezierTri6 …` line on any build (the friendly fork-build error is B3).
- **Shipped & verified:** 13 new unit tests (`test_elements_solid.py` —
  construction, 2-value plane_type rejects `*2D`, flag-prefixed emit order, D5
  warn+drop / PlaneStrain-keeps, wrong-node-count, namespace) + `test_catalog_coverage_v1`
  widened for the 4 BezierTri6 rows. 1478 primitives/contract/emitter/response
  regression green; `solid.py` mypy-clean (the 3 trivial seam files add one
  instance of the pre-existing `_resolve(base=NDMaterial)` pattern shared by all
  element ns methods).

### B2 — `ops.element.BezierTet10` typed primitive + O11 lock  *(no fork; mesh-only)* — ✅ DONE
- `BezierTet10` dataclass (`solid.py`) + `_ElementNS.BezierTet10` + `element/__init__`
  export + `_ElemSpec "BezierTet10"` (etype 11, identity reorder, `ndm_ok={3}`,
  `ndf_ok={3}`, `slots=("nodes","matTag")`, `has_gauss`; no `cpp_class_name`) +
  `_response_catalog` rows (`ELE_TAG_BezierTet10 = 33001`, `Tet_GL_2` + `Custom`,
  reusing the clean `_TET_GL_2_COORDS` — Tet10 GP order matches, no permutation).
  Flag tail `-bbar`/`-cMass`/`-rho`/`-bodyForce`(3-comp)/`-pressure` (the fork
  factory `OPS_BezierTet10.cpp` carries `-rho`/`-pressure` too — the plan's earlier
  brevity omitted them); **no** D5 guard (B-bar always valid in 3D, D5′).
- **O11 LOCKED — LIVE-RUN CONFIRMED (not just argued):** the mid-edge-at-midpoint
  round-trip (`tests/opensees/integration/test_bezier_tet10_o11.py`) meshes a
  straight-sided box to `tet10` and asserts every mid-edge node sits within
  **2.2e-16 (rel)** of its corner-pair midpoint under the order `N5=mid(1,2)
  N6=mid(2,3) N7=mid(1,3) N8=mid(1,4) N9=mid(3,4) N10=mid(2,4)`. The triple-source
  identity argument is now an empirical fact (the Tet10 analogue of the Tri6
  1.78e-15 confirmation). Mesh-only, no fork.
- **Shipped & verified:** 9 unit tests (`test_elements_solid.py` — construction,
  full flag-prefixed emit, B-bar always-kept, wrong-node-count, namespace) + the O11
  test + widened `test_catalog_coverage_v1` (4 Tet10 rows); 1496 primitives/contract/
  emitter/response regression green; `solid.py` mypy-clean.

### B3 — Run gating + clear fork-build error  *(no fork)*
- At `ops.run()`, when the deck contains a `BezierTri6`/`BezierTet10` and the live
  `openseespy` build lacks the class, raise a clear
  *"element BezierTri6 requires the Ladruno fork build (≥33000 tag band); meshing
  + deck emission work on any build — see direct-drive fallback."*
- **Verify:** a stock-build run with a bezier element in the deck raises the exact
  message (not a raw openseespy KeyError); a tcl/py export still succeeds.

### B4 — Result read (GP stress/strain) via the neutral basis lib  *(fixtures; depends on recorder-plan basis lib)*
- Read bezier GP stress/strain from a `.ladruno` (or `.mpco`) fixture: map the
  `ELE_TAG_BezierTri6=33000` / `33001` + `Custom`/real-rule keys through
  `_response_catalog` **for component layout / `n_gp` only**. Per-GP **coordinates**
  come from the file's `QUADRATURE/GP_PARAM` (canonical), reconstructed to physical
  space via the **neutral** `B(ξ; bernstein, 2)` (imported, not copied — promote the
  recorder lib to neutral if it shipped reader-local). **Never** map GP-index →
  catalog coords (the Tri6 GP-order caveat — the fork's index order differs).
- **Verify:** read a committed bezier fixture (tri6 + tet10), assert GP stress
  components match the catalog layout; **assert per-GP physical coords come from
  `B(ξ)·X` with `ξ` read from the file and land near the fork's GP barycentric
  positions** (this assertion catches the Tri6 index permutation). Fixture committed
  under `tests/fixtures/…`, generated once from the fork build (no fork at test time).

### B5 — Docs / parity / fallback note
- Document `ops.element.BezierTri6/BezierTet10` in the **canonical** skill references
  (`skills/apegmsh/references/`, not the `apegmsh-helper` mirror) + the contract rows
  (flip "typed primitive deferred" → shipped). **Keep direct-drive documented as the
  supported fallback** (`bezier_apegmsh_integration.md` cross-ref). CHANGELOG. Confirm
  B4 imports the *same* basis module the recorder plan ships.

---

## Open questions

> [!decided] **`B(ξ)` lives NEUTRAL at `src/apeGmsh/_basis.py`** — settled jointly
> with the recorder plan (its basis-lib OQ is now marked decided-neutral *because*
> this plan is the second consumer). B4 **imports** that module for `tri6-bernstein`
> + `tet10`; no second copy. If the recorder somehow ships L2 reader-local first, B4
> promotes it to neutral rather than duplicating.

> [!question] **B-bar guard: warn-and-drop vs raise (Tri6 PlaneStress).** Fork D5
> *warns + silently disables*. apeGmsh convention leans fail-loud. Mirror the fork
> (warn + drop the `-bbar` flag, keep the run) for behavioral parity, or raise at
> construction? Lean **warn + drop** to match the fork's documented behavior; revisit
> if users want the stricter guard.

> [!question] **Live O11 confirmation before B2 ships?** The identity verdict is a
> triple-source argument (Gmsh convention + fork "matches Gmsh" + existing
> `TenNodeTetrahedron` identity), locked by the B2 round-trip test. The Tri6 sibling
> was live-confirmed to `1.78e-15`; Tet10 has not been on an apeGmsh mesh. The B2
> test *is* the live confirmation — no separate gate needed, but flagged so a wrong
> Gmsh-convention assumption surfaces as a test failure, not a silent wrong stiffness.

> [!question] **`Custom` vs real `IntRule` rows — keep both?** `SixNodeTri` registers
> both `Triangle_GL_2` and `Custom` because upstream MPCO doesn't dispatch its tag
> and falls through to the `basisInfo`/`CustomIntegrationRule` path. The Ladruno
> recorder serves bezier via the same self-declaration. **Register both** (mirror the
> `SixNodeTri` 818–841 pattern); drop the real-rule mirror only if a reader proves it
> unused.

---

## Fork-side asks (request from the Ladruno team, not work around)

1. **Confirm the `basisInfo` self-declaration is emitted on every bezier result
   group** in the current `Ladruno` recorder build (the fork log says Tet10 wiring
   is "one block" reusing the generic `basisInfo` capture). The apeGmsh read (B4)
   relies on `FAMILY=bernstein` + `QUADRATURE/{GP_PARAM,GP_WEIGHT}` being present so
   it needs **zero** per-class decode. (If any bezier result still ships a flat
   class-tag-only group, ask to structure it.)
2. **A committed bezier fixture recipe** — a stable export (mirroring the recorder
   plan's `make_synthetic.py`) producing a small `tri6`/`tet10` `.ladruno`
   (and/or `.mpco`) so apeGmsh's B4 fixtures track the writer across schema bumps.
3. **Fork-side doc sweep.** (a) Stop citing the dead tags in prose —
   `04`/`06`/`bezier_apegmsh_integration` still say `ELE_TAG 272`/`273`; the live
   header is `33000`/`33001` (avoids the next reader copying the dead value into
   `_response_catalog`). (b) Fix the stale pointer in `bezier_apegmsh_integration.md`
   (L17-19) — it cites a canonical apeGmsh spec at `docs/plans/bezier-tri6-element.md`
   that does **not** exist; the real spec is this file
   (`internal_docs/plan_bezier_elements_integration.md`).

## Out of scope (this plan)

- **Curved Bézier edges/faces** + the Eq.14 Dirichlet control-point↔DOF mapping
  (fork D9/D9′ — element warns on curved meshes; v1 is straight-sided only).
- **Real edge/surface traction** (the `-pressure` volume hack), selectable
  quadrature, rational/NURBS, higher order (fork O5–O7, deferred there).
- **The explicit integrator** (Chung–Lee β=13/12 — a global `TransientIntegrator`,
  separate fork track / its own apeGmsh surface).
- **The `.ladruno` reader core / energy / orientation** — that is
  `plan_ladruno_recorder_integration.md`; this plan only *imports* its basis lib.

## References
- Fork ADRs: `ladruno:Ladruno_implementation/04_bezier_elements.md` (Tri6),
  `…/06_bezier_tet10.md` (Tet10 — O11/O12).
- Live tags: `ladruno:SRC/classTags.h` → `ELE_TAG_BezierTri6=33000`,
  `ELE_TAG_BezierTet10=33001`.
- Fork factories: `ladruno:SRC/element/bezierTriangle/OPS_BezierTri6.cpp`,
  `…/bezierTetrahedron/OPS_BezierTet10.cpp`.
- Contract: `origin/docs/apegmsh-feature-recs:Ladruno_implementation/ladruno_apegmsh_contract.md`
  (BezierTri6 row + reader notes), `…/bezier_apegmsh_integration.md` (direct-drive).
- apeGmsh seams: `solid.py` (`SixNodeTri`/`TenNodeTetrahedron`),
  `_internal/ns/element.py`, `_element_capabilities.py`, `_response_catalog.py`,
  `element/__init__.py`, `mesh/_element_types.py`.
- Basis-lib dependency: `internal_docs/plan_ladruno_recorder_integration.md`
  (L2 "Family basis lib", + its basis-location OQ).
