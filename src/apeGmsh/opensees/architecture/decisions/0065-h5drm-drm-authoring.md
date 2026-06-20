# ADR 0065 — H5DRM DRM-load authoring: typed pattern emit + DRMBox-from-`.h5drm` builder + buffer/boundary contract

**Status:** Proposed — REQUIREMENT (not yet implemented). Grounded in a validated
reference implementation in the OpenSees-fork side
(`OpenSees/Ladruno_implementation/drm_study/`, shipped with fork PR
`nmorabowen/OpenSees#296`, branch `ladruno`) and direct reads of the fixed
`SRC/domain/pattern/drm/H5DRMLoadPattern.cpp`. Pairs with **ADR 0054** (ASD
absorbing boundary — the ghost-layer emit) and the existing parametric `DRMBox`
part (`parts/drm_box.py`). This ADR owns the **H5DRM load pattern** and the
**dataset-keyed box builder**; ADR 0054 owns the absorbing layer.

## Context

The Domain Reduction Method (DRM) drives a soil box with a regional incident
wavefield read from an `.h5drm` HDF5 dataset (e.g. ShakerMaker synthetics) via
OpenSees' `H5DRM` load pattern. Today apeGmsh has **no** H5DRM support: a user
must hand-author the box, the `pattern H5DRM ...` line, the coordinate-frame
handshake, and the exterior/absorbing boundary. Doing this by hand is error-prone
— the reference study hit several silent, non-obvious failure modes (below) that
cost a full debugging loop. This ADR encodes that hard-won contract so apeGmsh
makes them impossible to get wrong.

### Source-verified facts the generator MUST honor (from the fixed C++)

1. **H5DRM coordinate transform** (`H5DRMLoadPattern::do_intitialization`):
   `xyz_model = T · ((xyz_station − drmbox_x0) · crd_scale) + x0`, where
   `drmbox_x0` is the **box centre read from the file** (`DRM_Metadata/drmbox_x0`),
   `crd_scale`/`T`/`x0` come from the pattern command. Station coords are
   typically **km**; FE models are **m** ⇒ `crd_scale = 1000`. Build the model
   **centred at the lateral origin** and `T = I`, `x0 = 0`, `crd_scale = 1000`
   reproduces the node coords exactly (verified: 98/98 match, d_err ~1e-12).
2. **Frame:** ShakerMaker is `z = Down (+down)`. With the model built in that
   native z-down frame and `T = I`, the OpenSees fix **removes** the historical
   `d[2]/a[2]` Z-flip (PR #296) — validated correct (interior `uz` tracks the
   prescribed `Dn`, corr +0.98). The generator should build z-down (surface at
   `z=0`, depth increasing +z) to match, OR supply the matching `T`.
3. **DRM-element rule** (`H5DRMLoadPattern.cpp:580`): an element is a DRM element
   **iff ALL its nodes are matched dataset nodes**; effective forces are computed
   only for elements having BOTH boundary (`b`) and interior (`e`) dataset nodes.
   ⇒ any buffer/absorbing element that touches a **non-dataset** node is excluded
   from the DRM force computation. This is the lever the buffer design uses.
4. **DOF/element limit:** H5DRM works only for **3-DOF, ≤8-node** elements
   (`stdBrick`/`SSPbrick`). Header-stated; no runtime guard.
5. **Past-tend:** the fix now **holds the final displacement** with zero
   acceleration past `tend` (no more `exit(-1)`); a transient may run
   indefinitely past the record. (`crd_scale` is Tcl- and now openseespy-exposed.)

### Failure modes the generator must prevent (all hit in the reference study)

- **Garbage transform** if `crd_scale`/`T` are wrong or defaulted → ~1 of N nodes
  match, DRM silently does nothing. (The C++ parser bug is fixed in #296, but the
  *frame handshake* is still the user's to get right — own it in the generator.)
- **Free box diverges:** a DRM box with NO exterior/absorbing boundary blows up
  (rigid-body modes excited by the residual; interior → 100s of m, diverging
  before wave arrival). A DRM box **requires** an exterior buffer + boundary.
- **Absorbing-on-`b`-nodes is wrong:** placing Lysmer/ASD elements directly on the
  DRM `b` nodes makes H5DRM sweep them into the force set (rule #3) — they must
  sit on a **separate exterior layer**.

## Decision (what apeGmsh must implement)

### R1 — Typed `ops.pattern.H5DRM(...)` on the apeSees bridge

```python
with ops.pattern.H5DRM(
        h5drm = "motions.h5drm",
        factor = 1.0,
        crd_scale = 1000.0,                 # km dataset -> m model
        distance_tolerance = 1.0,           # in MODEL units (m)
        transform = None,                   # None => identity (built-in centred frame)
        x0 = (0.0, 0.0, 0.0),
) as p:
    pass                                    # H5DRM carries its own field; no p.load
```

- Emits `pattern H5DRM $tag $file $factor $crd_scale $dist_tol $do_transform
  T00..T22 x00..x02` (Tcl) and the equivalent `ops.pattern('H5DRM', ...)`
  (openseespy — now registered, fork #296). `do_transform = 1` always (the
  transform is how km→m + centring happens); `transform=None` ⇒ identity.
- It is a **field-carrying** pattern: no `p.load` / `p.from_model`. Mutually
  exclusive with a base-input absorbing drive (ADR 0054 §7.2): **DRM ring OR
  `-fx/-fy/-fz` base input, never both.**
- Validate at build: every element the pattern can reach is 3-DOF ≤8-node (fact
  #4) — else `BridgeError`.

### R2 — `g.parts.add_DRM_box_from_h5drm(...)` (dataset-keyed box builder)

A builder that reads an `.h5drm` and produces the **inner DRM region** whose nodes
coincide with the dataset stations:

```python
drm = g.parts.add_DRM_box_from_h5drm(
        "motions.h5drm",
        crd_scale = 1000.0,            # station units -> model units
        centred = True,                # node = (xyz_station - drmbox_x0) * crd_scale
        soil = (E, nu, rho),           # or a material handle (per-layer/callable for graded soil)
)
# returns: PGs for the interior + the b/e split, the grid descriptor, the
# centre/transform so R1 + R3 stay consistent.
```

- Reads `DRM_Data/{xyz,internal}` + `DRM_Metadata/{drmbox_x0,h,...}`; derives the
  structured grid (origin, spacing `h`, per-axis counts) and builds a transfinite
  hex mesh landing nodes EXACTLY on the (transformed) stations. The fork study did
  this with `set_transfinite_box(size=h)`; reuse that.
- Distinct from the existing parametric `DRMBox` part (which is an SSI
  inner/transition/outer layout, NOT keyed to a dataset). This builder is
  dataset-first; the two should share the structured-hex machinery.
- Surfaces the **frame contract** (R1) so the matching `crd_scale`/`T`/`x0` for the
  pattern are returned, not re-derived by the user.

### R3 — Exterior buffer + boundary (pairs with ADR 0054)

```python
g.drm_buffer(
        drm,                            # the R2 result
        layers = 2,                     # 50 m soil-brick layers OUTWARD
        faces  = "sides+bottom",        # NOT the top free surface
        boundary = "fixed",             # "fixed" (validated) | "lysmer" | "asd" (ADR 0054)
)
```

- Extends the mesh outward on the 4 sides + bottom (not the top free surface) with
  buffer soil bricks, then applies the boundary on the **outermost** faces.
- **INVARIANT (rule #3):** every buffer/boundary node must be a **non-dataset**
  node (outside the DRM `b` shell), so H5DRM excludes the buffer/boundary elements
  from the effective-force set. The generator must guarantee this (the reference
  study's `add_lysmer.py`-on-`b`-nodes was WRONG: 168 vs 40 DRM elements).
- `boundary="fixed"` — `fix` the outermost nodes; simplest, validated (interior
  reproduces the free-field, corr 0.91–0.98, ratio ~1.0). `boundary="lysmer"` —
  bare `LysmerTriangle` on the outer faces (DRM exterior = scattered field, so bare
  dashpots are correct; but pure dashpots can't restrain the rigid null-space →
  needs a pin, and tracking was poor in the study — document the caveat).
  `boundary="asd"` — defer to ADR 0054's staged `ASDAbsorbingBoundary3D` ghost
  layer (the production path; requires the staged gravity→absorbing flip).

### R4 — Convention/pitfall encoding (the value-add)

The builders must **own** these so the user never reasons about them:
z-down frame; `crd_scale` km→m; centred `(xyz−drmbox_x0)·crd_scale` transform;
the all-nodes-DRM exclusion rule (R3 invariant); 3-DOF ≤8-node only; free-box
divergence ⇒ buffer+boundary mandatory; past-tend now holds final displacement.

## Consequences / acceptance

- **Reference implementation = the spec by example:** `drm_study/build_drm_model.py`
  (R2 inner box), `build_drm_buffered.py` (R3 buffer+fixed), `setup_tcl_run.py`
  + `add_lysmer.py` (R1 emit + lysmer), `post_drm_buffered.py` (validation). An
  apeGmsh implementation must reproduce the study's result: **98/98 nodes**,
  interior reproduces the free-field (post-arrival corr 0.91–0.98, ratio ~1.0),
  bounded past tend.
- **Acceptance test (port of the fork regression test):** a self-contained
  synthetic-`.h5drm` test that builds R2+R3 via the API, emits R1, runs a
  transient past tend, and asserts (a) the pattern is accepted with `crd_scale`,
  (b) the interior moves and stays bounded, (c) no crash past tend. (Cf.
  `OpenSees/tests/test_h5drm_drm_loadpattern.py`.)
- **Slices:** **D-1** R1 typed pattern (unblocks Tcl/py emit immediately) → **D-2**
  R2 dataset-keyed builder → **D-3** R3 buffer + `fixed` boundary (validated path)
  → **D-4** R3 `asd` boundary via ADR 0054 + staged flip (production SSI).
- **Non-goals:** generating the `.h5drm` itself (that is ShakerMaker's job);
  nonlinear far-field (the boundary stays linear-elastic — ADR 0054 §9.6).

## Related

- ADR 0054 — ASD absorbing boundary (the ghost-layer emit; R3 `asd` path).
- `parts/drm_box.py` — the parametric (non-dataset) DRM box; share hex machinery.
- Fork: `Ladruno_implementation/drm_study/SCOPE.md` (validation write-up),
  `lysmer_asd_absorbing_boundaries_guide.md` §7.2/§9, OpenSees PR #296.
