# Plan — DRM / H5DRM authoring (ADR 0066)

Implements [ADR 0066](../src/apeGmsh/opensees/architecture/decisions/0066-h5drm-drm-authoring.md).
Grounded in the copied fork reference study `internal_docs/drm_study/`
(the "spec by example") and the fixed C++ `H5DRMLoadPattern.cpp` / fork PR #296.

**Goal:** a user drives a soil box with a regional incident wavefield from an
`.h5drm` (e.g. ShakerMaker), authored entirely through apeGmsh — no hand-written
`pattern H5DRM` line, no manual frame handshake, no foot-guns. Reproduce the
study result: 98/98 nodes matched, interior tracks the free-field (post-arrival
corr 0.91–0.98, ratio ~1.0), bounded past `tend`.

## The 4 hard facts the generators must own (from the fixed C++)

1. **Transform:** `xyz_model = T·((xyz_station − drmbox_x0)·crd_scale) + x0`,
   where `drmbox_x0` = box centre read from the file. Build the model **centred
   at the lateral origin, z-down, in metres** ⇒ `crd_scale=1000`, `T=I`, `x0=0`
   reproduces node coords exactly (98/98, d_err ~1e-12).
2. **Frame:** ShakerMaker is z-Down. The fix **removes** the historical Z-flip;
   a z-down model with `T=I` is correct (interior `uz` tracks `Dn`, corr +0.98).
3. **DRM-element rule** (`H5DRMLoadPattern.cpp:580`): an element is a DRM element
   **iff ALL its nodes are matched dataset nodes**. ⇒ buffer/boundary elements
   must touch **non-dataset** nodes so H5DRM excludes them. (b-node Lysmer was the
   study's headline bug: 168 vs 40 elements.)
4. **DOF/element limit:** 3-DOF, ≤8-node only (`stdBrick`/`SSPbrick`). No runtime
   guard — apeGmsh must validate at build (`BridgeError`).

Plus: a **free DRM box diverges** (rigid-body null-space) → buffer + boundary is
mandatory, not optional.

## Existing apeGmsh surfaces to plug into (verified)

| Need | Lives at |
|------|----------|
| Pattern primitives (`Plain`, `UniformExcitation`) | `src/apeGmsh/opensees/pattern/pattern.py` |
| Pattern namespace `ops.pattern.*` | `src/apeGmsh/opensees/_internal/ns/pattern.py` (`_PatternNS`) |
| Parts builders (`add_DRM_box`, `add_plane_wave_box`) | `src/apeGmsh/core/_parts_registry.py` + `src/apeGmsh/parts/*.py` |
| Structured hex mesh | `src/apeGmsh/mesh/_mesh_structured.py::set_transfinite_box(vol, size=, recombine=)` ✅ exists |
| ASD absorbing boundary (R3 `asd` path) | `src/apeGmsh/opensees/element/absorbing.py` + `_internal/ns/element.py::absorbing_boundary(skin=…)` |
| `AbsorbingSkinResult` / skin builders | `src/apeGmsh/parts/plane_wave_box.py` |

No `H5DRM` / `drm_buffer` exists in `src/` yet — all net-new.

---

## Slices

### D-1 — R1: typed `ops.pattern.H5DRM(...)`  ✅ SHIPPED *(unblocks emit immediately)*

> Shipped: `H5DRM` primitive + `_PatternNS.H5DRM` + Tcl/py/H5 emit (all generic —
> single-line pattern, no emitter changes needed). Unit + contract + emitter +
> H5-round-trip tests green. The 3-DOF≤8-node + no-base-input-mixing guards are
> deferred to the bridge build pipeline (need the FEM snapshot), tracked below.


Add an `H5DRM` pattern primitive mirroring `UniformExcitation` (field-carrying,
no `p.load`).

- **`pattern/pattern.py`**: new `@dataclass(frozen, kw_only, slots)` `H5DRM(Pattern)`
  with fields `h5drm: str`, `factor: float = 1.0`, `crd_scale: float = 1000.0`,
  `distance_tolerance: float = 1.0`, `transform: tuple[...]|None = None` (None ⇒
  identity), `x0: tuple[float,float,float] = (0,0,0)`. `__enter__/__exit__`
  no-op body (like `UniformExcitation`). Add to `__all__`.
- **`_internal/ns/pattern.py`**: `def H5DRM(self, *, h5drm, factor=1.0,
  crd_scale=1000.0, distance_tolerance=1.0, transform=None, x0=(0,0,0), name=None)`.
- **Emit** (Tcl + py) — canonical 18-arg form (fork-verified, see drm_study README):
  `pattern H5DRM $tag $file $factor $crd_scale $dist_tol 1  T00..T22  x00..x02`
  and `ops.pattern('H5DRM', tag, file, factor, crd_scale, dist_tol, 1, *T, *x0)`.
  `do_transform = 1` always; `transform=None` ⇒ row-major identity.
- **Build-time validation** (fact #4): every element the pattern can reach is
  3-DOF ≤8-node, else `BridgeError`. Mutually exclusive with a base-input
  absorbing drive (ADR 0054 §7.2) — DRM ring **OR** `-fx/-fy/-fz`, never both.
- **Verify:** unit test asserts the emitted Tcl/py line matches the canonical
  form for identity + km→m; `BridgeError` on a 6-DOF element pg.

### D-2 — R2: `g.parts.add_DRM_box_from_h5drm(...)`  ✅ SHIPPED *(dataset-keyed inner box)*

> Shipped: `parts/h5drm_box.py` (`build_drm_box_from_h5drm` + `DRMBoxFromH5Result`)
> + `_parts_registry.add_DRM_box_from_h5drm`. Reads the `.h5drm`, validates a
> complete uniform regular grid, builds the centred z-down box landing nodes on
> the stations, tags the soil volume + six outer-face boundary PGs (the b/e split,
> with a sanity check vs the dataset `internal` flag), and returns the frame
> contract (identity transform / x0=0 / center=drmbox_x0). Materials stay on the
> bridge (no `soil=` — parts build geometry, not physics). 8 tests incl. the
> node-coincidence acceptance; ruff + mypy clean.

A parts builder that reads an `.h5drm` and produces the inner DRM region whose
nodes land exactly on the (transformed) stations. Port `build_drm_model.py`.

- **`parts/h5drm_box.py`** (new, parallel to `plane_wave_box.py`):
  `build_drm_box_from_h5drm(session, *, h5drm, crd_scale=1000.0, centred=True,
  soil=(E,nu,rho)|material, name=None)`. Reads `DRM_Data/{xyz,internal}` +
  `DRM_Metadata/{drmbox_x0,h}`; derives origin/spacing/per-axis counts; builds an
  `add_box` + `set_transfinite_box(size=h_m)` landing nodes on the stations;
  splits PGs into interior (`internal==1`) vs boundary (`b`/`e`).
- Returns a frozen `DRMBoxFromH5Result`: interior PG, b/e split PGs, grid
  descriptor, and the **frame contract** (`crd_scale`, `T`, `x0`, centre) so D-1
  + D-3 stay consistent without the user re-deriving it.
- **`_parts_registry.py`**: `def add_DRM_box_from_h5drm(self, *, …)` wrapper.
- Share the structured-hex machinery with `drm_box.py`; this is dataset-first,
  distinct from the parametric `DRMBox`.
- **Verify:** build against the synthetic `.h5drm`; assert node count + that every
  station coord has a coincident mesh node (d_err < tol).

### D-3 — R3: `g.drm_buffer(...)` + `fixed` boundary  *(the validated stable path)*

Extend the inner box outward + apply the boundary on non-dataset nodes. Port
`build_drm_buffered.py` (the structured-numpy logic → apeGmsh geometry).

- **`g.drm_buffer(drm_result, *, layers=2, faces="sides+bottom",
  boundary="fixed")`** — extends the mesh by `layers` soil-brick layers on the 4
  sides + bottom (**never** the top free surface), then applies the boundary on
  the **outermost** faces.
- **INVARIANT (fact #3):** every buffer/boundary node is a non-dataset node.
  The builder must guarantee it (this is the whole point — the study's
  Lysmer-on-b-nodes was wrong).
- `boundary="fixed"`: `fix` outermost nodes — simplest, **validated** (interior
  reproduces free-field, corr 0.91–0.98). Ship this.
- `boundary="lysmer"`: bare `LysmerTriangle` on outer faces — document the caveat
  (pure dashpots can't restrain the rigid null-space; needs a pin; tracking poor).
- **Verify:** the acceptance test below (port of the fork regression test).

### D-4 — R3 `asd` boundary via ADR 0054 + staged flip  *(production SSI)*

Wire `boundary="asd"` to the existing `ASDAbsorbingBoundary3D` ghost layer
(`ops.element.absorbing_boundary`) + the staged gravity→absorbing flip. Defer
until D-1..D-3 land; this is the production path but depends on the staged-flip
plumbing already shipped for ADR 0054.

---

## Acceptance test (port of `drm_study/test_h5drm_drm_loadpattern.py`)

Self-contained: synthesize a tiny 3×3×3 `.h5drm` with `h5py` (no ShakerMaker),
build R2+R3 via the API, emit R1, run a transient past `tend`, assert:
1. the pattern is accepted with `crd_scale` (no "unknown pattern type"),
2. node-matching drives the model (a free node moves > 1e-9 — guards garbage-T),
3. past `tend`, `analyze()` returns 0 and stays bounded (hold-past-end).

Run under `C:\Users\nmora\venv\opensees_venv\Scripts\python.exe`. **Live-skip** if
the fork binary lacking openseespy-H5DRM is in play (the registration + CMake
`_H5DRM`-scope fixes shipped in fork #296 — gate on import availability, like the
`g.reinforce` LadrunoEmbeddedRebar live-skip pattern).

## Non-goals

Generating the `.h5drm` (ShakerMaker's job); nonlinear far-field (boundary stays
linear-elastic, ADR 0054 §9.6); 2D DRM (3-DOF 3D only per fact #4).

## Open questions for the user

1. **Scope now:** D-1 only (typed pattern, unblocks hand-built boxes), or
   D-1→D-3 (full validated authoring path) in this runway?
2. **`g.drm_buffer` home:** free function `g.drm_buffer(...)` (ADR's spelling) vs
   `g.parts.add_drm_buffer(...)` (consistent with the other parts builders)?
3. **Fork binary:** is a worktree build with openseespy-H5DRM available on this
   machine for live tests, or do we ship D-1..D-3 emit-only + live-skip?
