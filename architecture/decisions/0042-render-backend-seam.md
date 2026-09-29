# ADR 0042 — Rendering-backend seam: `SceneLayer` IR + `RenderBackend` Protocol

**Status:** Accepted (2026-05-28, head-engineer review; three open
questions resolved in §Resolved decisions). Mirror of
ADR 0026 on the *render* side: where ADR 0026 made the viewer's
**read** path a structural Protocol so foreign-format model readers
drop in, this ADR makes the viewer's **render** path a structural
Protocol + a declarative scene description so alternate render
backends (web/Jupyter via `trame`, and later a ParaView export
adapter) drop in without touching `diagrams/`, `overlays/`, or
`core/`. Successor to ADR 0014 (viewers pure-h5 consumer). Adoption
ships as Phases R-A through R-C (sequenced under §Decision); R-D is
a separate future ADR.

## Context

### The read side got disciplined; the render side never did

ADR 0014 made `apeGmsh.viewers` a pure `model.h5` consumer and
ADR 0026 formalised the in-process read contract as the
`H5ModelReader` Protocol. The result is that the *input* to the
viewer is now backend-agnostic: any object satisfying the Protocol
feeds `ViewerData.from_reader()`, and an AST guard
(`tests/viewers/test_viewers_pure_h5_consumer.py`) keeps the import
surface honest.

The *output* — how the viewer puts pixels on a screen — has no such
seam. A subsystem audit (May 2026) found **~39 direct `pyvista`
imports** scattered across `viewers/core/`, `viewers/diagrams/`,
`viewers/overlays/`, and `viewers/scene/`. Every one of the 11
diagram types (`_contour`, `_deformed_shape`, `_fiber_section`,
`_gauss_marker`, `_layer_stack`, `_line_force`, `_spring_force`,
`_reactions`, `_vector_glyph`, `_section_cut`, `_loads`) calls
`plotter.add_mesh(...)` / `plotter.add_scalar_bar(...)` directly.
Visibility is expressed three different ways tied to VTK mechanics —
`VisibilityManager` via `extract_cells`, `ElementVisibility` via the
`vtkGhostType` bitmask, `OverlayVisibilityModel` via per-actor
toggles. Picking is raw VTK ray-casting through
`plotter.renderer.pick`.

There is **no abstraction layer**. The render backend is locked to
`pyvistaqt.QtInteractor`.

### What that costs, repeatedly

The lack of a seam is the root cause of the recurring viewer pain:

1. **No headless / no-GPU rendering.** This sandbox cannot get an
   OpenGL context (`vtkWin32OpenGLRenderWindow: failed to get valid
   pixel format`); every viewer PR re-discovers that a full `show()`
   cannot be exercised. Verification falls back to parity + smoke +
   the user's eyeball.
2. **VTK+Qt blocking-kernel crashes** in Jupyter — the default
   blocking viewer kills the ipykernel; `viewer(blocking=False)` /
   subprocess is the workaround.
3. **Every new diagram re-couples to PyVista.** There is no contract
   that says "a diagram emits *this*"; it says "a diagram calls the
   plotter." Growth compounds the coupling.

### The delivery target widened

The decision driving this ADR (head-engineer review, May 2026) is to
support **both** a native Qt desktop viewer **and** a web/Jupyter
viewer. The desktop experience (dock layout, outline trees,
preferences, session persistence, the `OverlayVisibilityModel`
cross-widget sync pattern) is a large, heavily-Qt investment worth
keeping. The web/Jupyter target — the natural home for `trame` — is
what closes the headless and kernel-crash pain at the source:
server-side or browser-side rendering, no local OpenGL context, no
Qt event loop fighting the ipykernel.

One render path cannot serve both. A seam can.

### ParaView and Blender were assessed as foundations and rejected as wholesale replacements

- **Blender** — GPL (incompatible with the project's permissive
  posture), artist-centric, no native scalar colormaps / scalar bars
  / clip / contour / glyph filters, degrades past ~100k vertices.
  Wrong tool. Rejected outright (see §Rejected A).
- **ParaView** — *same VTK family as the current stack* (PyVista is
  VTK; ParaView is VTK + server/client + Qt + a filter catalog),
  BSD-licensed, with a clean `paraview.simple` Python automation API
  and native readers for VTU/VTM/Exodus/XDMF and even `.msh`. But it
  has **zero** concept of the domain logic that is this subsystem's
  actual value: OpenSees fiber sections, shell layer stacks,
  gauss-point markers, partition-rank colouring (ADR 0027), the
  stage/step `ResultsDirector`. A wholesale pivot would mean
  *rebuilding* all of that as ParaView filters/plugins — strictly
  more work, not less (see §Rejected B). ParaView's right role is a
  **future export/automation adapter behind this seam**, not a
  dependency taken on now.

### Why this is an ADR, not a refactor

The same reasoning ADR 0026 used applies in mirror image. An
*implicit* render contract that 11 diagrams + every overlay already
depend on is one `plotter`-method change away from a silent break,
and it structurally forecloses the web/Jupyter target. Codifying the
seam **before** migrating the diagrams is what makes the migration a
controlled, parity-gated sequence instead of a brittle big-bang
rewrite. Introducing or widening a Protocol that multiple call sites
depend on is "an architecture event" (precedent: ADR 0024/0025).

## Decision

Introduce a two-part seam between the viewer's domain logic and its
renderer:

1. A **declarative scene description** — `SceneLayer` value types
   carrying plain arrays + style specs, no VTK/pyvista.
2. A **`RenderBackend` Protocol** — the structural contract a backend
   implements to consume `SceneLayer`s and produce pixels.

Domain logic (`diagrams/`, `overlays/`, `core/` colour & visibility)
*emits* `SceneLayer`s and calls only `RenderBackend` methods. It
imports neither `pyvista` nor `vtk`.

### Part 1 — `SceneLayer` IR (`apeGmsh.viewers.scene_ir`)

A small, additive vocabulary of frozen value types. Array data is
carried in **light typed bundles** (`PointSet`, `CellBlocks`,
`ScalarField`) — thin wrappers over `numpy` that pin dtype +
contiguity and validate shape at construction — **never VTK
objects**. The bundles keep the IR constructible and assertable with
no GPU and no render context (INV-1), and the pinned dtype lets the
backend hand arrays to VTK zero-copy, which protects step-animation
from per-frame casts.

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Mapping, Optional, Sequence
import numpy as np


@dataclass(frozen=True)
class PointSet:
    """(n, 3) coordinates, pinned C-contiguous float32 at construction."""
    coords: np.ndarray

    def __post_init__(self) -> None:
        assert self.coords.ndim == 2 and self.coords.shape[1] == 3


@dataclass(frozen=True)
class CellBlocks:
    """vtk-cell-type token -> (n_cells, n_nodes) connectivity, int64."""
    blocks: Mapping[str, np.ndarray]


@dataclass(frozen=True)
class ScalarField:
    """A named scalar/vector field bound to a domain location.

    Carries its own dtype guarantee so the backend never re-casts
    mid-animation. ``location`` removes the point-vs-cell ambiguity
    that raw dict-of-arrays leaves implicit."""
    name: str
    values: np.ndarray
    location: Literal["point", "cell"]


@dataclass(frozen=True)
class ColorSpec:
    """How a layer is coloured. Exactly one mode is active."""
    mode: Literal["solid", "by_array", "per_entity_rgb"]
    solid_rgb: Optional[tuple[float, float, float]] = None
    array_name: Optional[str] = None          # for "by_array"
    lut: Optional["LutSpec"] = None            # for "by_array"
    entity_rgb: Optional[np.ndarray] = None    # (n_cells, 3) for "per_entity_rgb"


@dataclass(frozen=True)
class LutSpec:
    name: str                                  # "viridis", ...
    vmin: float
    vmax: float
    n_colors: int = 256


@dataclass(frozen=True)
class VisibilityMask:
    """Per-cell visibility. The IR-level expression of
    VisibilityManager / ElementVisibility / OverlayVisibilityModel.
    Backends apply it however they like (vtkGhostType, extract_cells,
    actor toggles) — the *semantics* live here, not in the backend."""
    hidden_cells: frozenset[int] = frozenset()


@dataclass(frozen=True)
class MeshLayer:
    layer_id: str
    points: PointSet
    cells: CellBlocks
    fields: Sequence[ScalarField] = ()
    color: ColorSpec = ColorSpec(mode="solid", solid_rgb=(1, 1, 1))
    visibility: VisibilityMask = VisibilityMask()
    opacity: float = 1.0
    show_edges: bool = False
    silhouette: bool = False


@dataclass(frozen=True)
class GlyphLayer:
    """Loads / masses / constraints / reactions / vector fields."""
    layer_id: str
    positions: np.ndarray                      # (n, 3)
    kind: Literal["arrow", "sphere", "cone", "axes"]
    orientations: Optional[np.ndarray] = None  # (n, 3) for vectors
    scales: Optional[np.ndarray] = None        # (n,) or scalar
    color: ColorSpec = ColorSpec(mode="solid", solid_rgb=(1, 1, 1))
    visibility: VisibilityMask = VisibilityMask()


@dataclass(frozen=True)
class ScalarBarSpec:
    layer_id: str
    title: str
    lut: LutSpec


@dataclass(frozen=True)
class LabelLayer:
    layer_id: str
    positions: np.ndarray
    texts: Sequence[str]
```

The vocabulary is **additive by design** — new diagram needs widen
the IR the way ADR 0024/0025 widened the `Emitter` Protocol, not by
leaking VTK back across the seam.

### Part 2 — `RenderBackend` Protocol (`apeGmsh.viewers.scene_ir`)

```python
from typing import Protocol
from pathlib import Path


class LayerHandle(Protocol):
    """Opaque backend handle to an added layer."""


class RenderBackend(Protocol):
    """The viewer-side render contract. Reference implementer:
    PyVistaQtBackend (desktop). Alternate: TrameBackend (web/Jupyter,
    headless-capable). Future: ParaViewExportBackend.

    A backend owns ALL VTK / pyvista / trame construction. The domain
    layer never imports those — it speaks SceneLayer + this Protocol.
    """

    def add_layer(self, layer: "SceneLayer") -> LayerHandle: ...
    def update_layer(self, handle: LayerHandle, layer: "SceneLayer") -> None: ...
    def remove_layer(self, handle: LayerHandle) -> None: ...
    def set_visibility(self, handle: LayerHandle, mask: "VisibilityMask") -> None: ...
    def add_scalar_bar(self, spec: "ScalarBarSpec") -> None: ...
    def remove_scalar_bar(self, layer_id: str) -> None: ...
    def reset_camera(self) -> None: ...
    def render(self) -> None: ...
    def screenshot(self, path: Path) -> None: ...

    # Capability probe — picking is optional (see INV-3).
    def supports_picking(self) -> bool: ...
```

Picking is **not** on the base Protocol. Ray-casting is the most
deeply VTK-bound surface and the least essential for the first
web/Jupyter target ("look at my model" / "animate my steps" does not
require pick). It gets its own `PickBackend` Protocol in Phase R-D
(future ADR). A backend that returns `supports_picking() == False` is
legal and renders view-only.

### Backends

| Backend | Target | Status |
|---|---|---|
| `PyVistaQtBackend` | Native Qt desktop | **Reference** — wraps the existing `QtInteractor`/`Plotter`; translates `SceneLayer` → `add_mesh`/`add_glyph`; applies `VisibilityMask` via the current `vtkGhostType`/`extract_cells` mechanics (now backend-internal). |
| `TrameBackend` | Web / Jupyter / headless | New (Phase R-C). VTK + `trame-vtk` in **hybrid** mode: client-side WebGL (vtk.js) for smooth interaction, server-side render (software/OSMesa fallback) for large meshes and high-fidelity stills. No Qt event loop → closes the kernel-crash pain; the remote half closes the no-GPU pain. View-only initially (`supports_picking() == False`; picking is R-D). |
| `ParaViewExportBackend` | Power-user export / automation | Future (R-D+, separate ADR). Translates `SceneLayer`s to `paraview.simple` calls or a VTM/Exodus export. Behind the seam; never a hard dependency. |

### Phased adoption (parity-gated, mirrors ADR 0026's staging)

| Phase | Scope | Effect |
|---|---|---|
| **R-A** | Define `scene_ir` (Part 1 + Part 2). Extract `PyVistaQtBackend` and route **one** surface through it — the scene build (`scene/mesh_scene.py`, `scene/fem_scene.py`) + `core/color_mode_controller`. | No behaviour change. New AST guard asserts `scene_ir` imports neither vtk nor pyvista. Parity test: backend-rendered scene == current scene for a fixture model. |
| **R-B** | Migrate the 11 diagrams + overlays to **emit** `SceneLayer`s and call only `RenderBackend`. Done diagram-by-diagram behind parity tests. | The big chunk. **Side benefit:** each diagram becomes headlessly testable by asserting on the emitted IR — no GPU, no pixels. Directly retires the "viewers can't be verified here" debt for the migrated surfaces. |
| **R-C** | Implement `TrameBackend`; add a web/Jupyter entry (`results.viewer(backend="web")` / a notebook handle). Picking stays desktop-only. | Web/Jupyter target delivered. Headless rendering possible. |
| **R-D** *(future, separate ADR)* | `PickBackend` Protocol + trame server-side picking. `ParaViewExportBackend`. | Validates the seam under a third backend and restores picking on web. |

R-A and R-B ship behind the existing `tests/viewers/` suite. Each
diagram migration in R-B is independently revertible.

## Invariants

- **INV-1** — The `scene_ir` module imports **neither `vtk` nor
  `pyvista`**. `SceneLayer`s carry plain `numpy` arrays + value-type
  specs. This is what makes the IR GPU-free, assertable in CI, and
  backend-agnostic. Enforced by an AST guard mirroring
  `test_viewers_pure_h5_consumer.py`.
- **INV-2** — `viewers/diagrams/`, `viewers/overlays/`, and the
  colour/visibility logic in `viewers/core/` import **no `pyvista`**.
  They emit `SceneLayer`s and call `RenderBackend` methods only.
  Backends own all VTK/pyvista/trame construction. (Migrated
  per-module across R-A/R-B; the guard tightens as each lands.)
- **INV-3** — Picking is an **optional** capability, not on the base
  `RenderBackend`. A view-only backend (`supports_picking() ==
  False`) is legal. The desktop backend supports it; the first trame
  backend does not.
- **INV-4** — `PyVistaQtBackend` is the **reference** backend. Parity
  tests assert any second backend handles the shared IR equivalently
  (same layers added/removed/updated for the same diagram step). The
  IR — not pixel output — is the equivalence oracle.
- **INV-5** — Visibility **semantics** live in the IR
  (`VisibilityMask`), not in backend tricks. The current
  `vtkGhostType` bitmask, `extract_cells`, and per-actor toggles
  become *implementation details* of `PyVistaQtBackend.set_visibility`.
  Both backends honour the same mask identically. The
  `OverlayVisibilityModel` cross-widget sync pattern (model-side,
  pure-Python) is unaffected — it already sits above the render layer.
- **INV-6** — The IR widens **additively**. A new diagram need that
  the vocabulary cannot express is a deliberate IR-widening event
  (new `SceneLayer` subtype or field), never a VTK object smuggled
  across the seam. Mirrors the `Emitter` Protocol widening discipline
  (ADR 0024/0025).

## Rejected alternatives

### A — Adopt Blender as the render engine

GPL (incompatible with the project's permissive licensing posture),
artist-centric, no native scalar colormaps / scalar bars / clip /
contour / glyph filters, degrades past ~100k vertices, path-traced
rendering is overkill and slow for interactive engineering data.
Retrofitting scientific viz costs more than any other option.
Rejected outright.

### B — Adopt ParaView wholesale as the engine

ParaView is the same VTK family, BSD-licensed, and has a clean
Python automation API — attractive on paper. Rejected as a *wholesale
replacement*: ParaView has no concept of the OpenSees-specific domain
logic that is this subsystem's value (fiber sections, layer stacks,
gauss markers, partition colouring, the stage/step director).
Replacing the viewer with ParaView means rebuilding all of that as
ParaView filters/plugins — more work, and it discards the asset. The
seam keeps ParaView as a **future export/automation adapter behind
`RenderBackend`** (Phase R-D+), which is where it belongs.

### C — Harden `pyvistaqt` in place, no seam

Stay single-backend; fix headless/crash pain tactically (offscreen
rendering, subprocess isolation). Cheapest now. Rejected: it does not
deliver the web/Jupyter target the review committed to, and it
perpetuates the 39-import scatter — every new diagram keeps
re-coupling to PyVista, and viewer verification stays pixel-bound
(unrunnable here).

### D — Imperative backend Protocol only (no declarative IR)

Abstract the `plotter` behind a Protocol but keep diagrams calling
`backend.add_mesh(...)` imperatively, with no `SceneLayer` value
types. Lower migration cost. Rejected: it loses the headless-
testability win (there is no value object to assert on) and it leaks
VTK semantics across the seam (cell types, ghost masks) — a second
backend would have to re-implement VTK's mental model. The hybrid
(declarative IR + thin imperative backend) costs more up front and
pays back in testability and backend independence.

### E — Big-bang rewrite to trame, drop Qt desktop

Make trame the only viewer. Rejected: discards the mature Qt desktop
investment (docks, outline trees, preferences, sessions, the
`OverlayVisibilityModel` pattern) and the review explicitly wants
**both** desktop and web. A seam serves both from one domain layer.

### F — Swap PyVista for raw VTK, keep one backend

Replace the PyVista dependency with direct VTK calls. Rejected: it
changes the coupling *target* without introducing a seam — same
single-backend lock-in, same no-web outcome, plus a large mechanical
rewrite with zero strategic payoff.

## Consequences

**Positive:**

- Web/Jupyter rendering becomes possible (Phase R-C) — the headless
  and kernel-crash pain is closed at the source, not worked around.
- Each migrated diagram becomes **headlessly testable** by asserting
  on the `SceneLayer` IR it emits. This is the single biggest payoff
  given that full `show()` cannot be exercised in CI / this sandbox.
- The domain logic — the 11 diagrams, the director, partition
  colouring, overlay visibility — is preserved intact and made
  backend-portable. None of it is thrown away.
- A ParaView export/automation adapter (or any future engine) becomes
  a drop-in behind `RenderBackend`, never a hard dependency.
- The render seam is the structural mirror of the read seam
  (ADR 0026), giving the viewer a symmetric, documented contract on
  both its input and output boundaries.

**Negative:**

- Large migration. R-B touches all 11 diagrams + every overlay.
  Staged and parity-gated, so it spreads across many small PRs rather
  than one risky rewrite, but it is the dominant cost.
- Two render paths to maintain once `TrameBackend` lands. Mitigated
  by INV-4 (one reference backend, IR-level parity oracle).
- Picking is desktop-only until Phase R-D — the first web/Jupyter
  viewer is view-only. Acceptable for the primary notebook use case;
  a known gap, not a silent one (INV-3 makes it explicit).
- The IR vocabulary must be expressive enough for 11 diagram types;
  early churn is likely. Mitigated by INV-6 (additive widening) — the
  same discipline that kept the `Emitter` Protocol stable.
- **Hybrid trame (resolved decision 2) is the most complex render
  mode to build and tune** — local/remote handoff, geometry-size
  thresholds, and two rendering code paths in the first web release.
  Accepted deliberately to get the best interaction UX without a
  later local-rendering follow-up; the seam itself is indifferent, so
  the complexity is contained inside `TrameBackend`.
- **The hard `trame` dependency (resolved decision 3) makes the base
  install heavier** for desktop-only and headless-batch users who
  never open the web viewer — larger images, more version-conflict
  surface. Accepted for install/import simplicity; reversible to an
  optional `[web]` extra later without a breaking change.

## Resolved decisions (head-engineer review, 2026-05-28)

1. **IR array shape — typed bundles.** The IR carries `PointSet` /
   `CellBlocks` / `ScalarField` bundles, not raw dict-of-`numpy`.
   They pin dtype + contiguity at construction (protecting animation
   from per-frame casts) and validate shape at emit time (malformed
   diagrams fail loud, not as a cryptic backend error). The bundles
   still import no vtk/pyvista, so INV-1 holds. Chosen over raw numpy
   because retrofitting type discipline across 11 diagrams *after*
   R-B is expensive.
2. **`trame` mode — hybrid from the start.** `TrameBackend` ships
   local (client WebGL) + remote (server render, software fallback)
   together in R-C, not remote-first. Buys the best interaction UX
   immediately *and* the headless-safe path, at the cost of more
   build/tuning effort in the first web release (see §Consequences).
3. **Web dependency — hard, not optional.** `trame` / `trame-vtk`
   are base dependencies; there is no `[web]` extra and no guarded
   import. Chosen for a single, simple install + import story; the
   accepted cost is a heavier base install for desktop-only and
   batch users (see §Consequences). The decision is revisitable if
   the dependency surface proves to conflict in practice — flipping
   to an optional extra later is additive, not a breaking change.

   **REVISED at R-C implementation (2026-05-29).** This rested on a
   false premise: it assumed the render stack lived in base
   `dependencies`. It does not — `pyvista` / `vtk` / `PySide6` are all
   in the optional `[viewer]` extra, so making `trame` a *base* dep
   would have pulled web libraries for users who installed no renderer
   at all (and still lacked pyvista). `trame` / `trame-vtk` therefore
   joined the existing `[viewer]` extra, not base `dependencies` and
   not a separate `[web]` extra — one render-install story, consistent
   with the actual packaging. Exactly the additive, non-breaking flip
   this decision anticipated.

## References

- [ADR 0014](0014-viewer-is-pure-h5-consumer.md) — viewers as a
  pure-h5 consumer. This ADR extends the same discipline from the
  read boundary to the render boundary.
- [ADR 0026](0026-h5modelreader-protocol-contract.md) — the
  `H5ModelReader` read-side Protocol. This ADR is its render-side
  mirror; the staging (zero-cost wrap, then incremental refactor) and
  the structural-`Protocol` + AST-guard pattern are lifted directly.
- [ADR 0024](0024-emitter-protocol-widen-region.md) /
  [ADR 0025](0025-emitter-protocol-widen-eigen.md) — Protocol
  widening as an architecture event; the precedent for INV-6's
  additive-widening rule for the IR.
- [ADR 0027](0027-cross-partition-mp-constraints.md) — cross-partition
  partition colouring; a `SceneLayer` colour mode (`per_entity_rgb`)
  the IR must preserve for the shipped `mesh.viewer` Partition mode.
- `tests/viewers/test_viewers_pure_h5_consumer.py` — the AST-guard
  pattern this ADR re-uses to enforce INV-1 and INV-2.
- Viewer subsystem assessment (May 2026 head-engineer review) — the
  PyVista/ParaView/Blender three-way evaluation that motivated the
  seam over a wholesale engine swap.
