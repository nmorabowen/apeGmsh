# ADR 0078 — In-process cross-section property analyzer (`SectionProperties`) (summary)

**Status:** Accepted (2026-07-18) — shipped in six slices, all `--base main`: S1 geometric #802
· S2 warping #803 […] · S3 plastic #804 · S4 stress #805 · S5 bridge binding + flat-face builders
#808 […] · S6 Qt inspector #810 · close-out (this flip + docs + PyPI oracle CI lane). […]

## Decision

A standalone **`SectionProperties(fem, materials=..., disconnected=...)`** analyzer, a
FEMData-consuming broker like `apeSees(fem)` / `Results`, computes geometric, warping, plastic
and stress results for any meshed 2-D cross-section natively in-process (`apeGmsh.fem` shape
functions + quadrature + `scipy.sparse`). No new runtime dependency; the PyPI
`sectionproperties` package is a dev-only test oracle.

- **Placement.** `src/apeGmsh/sections/`, exported as `from apeGmsh import SectionProperties`;
  session-independent after construction; each `*Properties` is a cached frozen dataclass.
- **Input contract (fail-loud).** Any meshed face qualifies (OCC, CAD import or builders).
  2-D elements only, one plane (`SectionMeshError` otherwise). `materials=` maps PG names to
  `SectionMaterial(E, nu, G=None, fy=None, density=None)`, and every element is covered by
  exactly one PG; omitting it runs geometric-only mode (unit `E`). `disconnected="raise"`
  (default) makes `warping()` require one connected domain; `"sum"` solves per component
  (`GJ = ΣGJᵢ`, `GJᵢ`-weighted shear centre, no inter-part shear transfer). Partial shear
  transfer is authored (a connector strip with a calibrated `G`), never a knob. `plastic()`
  needs `fy`; `tri3`/`quad4` warn `SectionAccuracyWarning` on `warping()`/`stress()`.
- **Axes.** Results are in the gmsh (x, y) authoring plane (`Ixx = ∫y²dA`). The bridge maps
  *authoring x ≡ local z, authoring y ≡ local y*: `Ixx_c → Iz`, `Iyy_c → Iy`, `J → J`,
  `As_y/A → alphaY`, `As_x/A → alphaZ`, in one shared lowering. Composites need an explicit
  reference `E`.
- **Declarative binding.** `p.section.ComputedSection(analysis=sec, E=None, G=None)`
  (`opensees/section/computed.py`) subclasses `Section`, so every consumer takes it
  unchanged; it resolves at emit into a plain `section Elastic` line (no `Emitter` widening,
  no schema change), memoized so N references cost one solve. Emit-time failure is loud.
  Eager escape hatch: `sec.to_elastic_section()`.
- **Analyses.** Geometric: modulus-weighted quadrature. Warping: Saint-Venant, ω regularized
  by a Lagrange row `∫ω dA = 0`, one `splu`, three back-substitutions. Plastic: neutral-axis
  bisection (invalid for softening materials). Stress: a blend of six unit-load fields.
- **Inspector.** `sec.viewer(*, blocking=True)`: a standalone Qt + matplotlib panel outside
  the ADR 0014/0042/0056 viewer family; all of it is also headless (`summary()`,
  `plot_section()`, `stress().plot()`, `_repr_html_`).
- **Flat-face builders** (convenience only): `g.sections.W_face`, `rect_face`,
  `rect_hollow_face`, `pipe_face`, `pipe_hollow_face`, `angle_face`, `channel_face`, `tee_face`.

## Amendments

- A1 (2026-07-18, ratified + implemented 2026-07-19): H5 persistence as a provenance
  sidecar, `/opensees/computed_sections` (tag, analyzer name, JSON payload), written only
  when a record exists, hash-excluded, `opensees_schema_version` 2.19.0 → 2.20.0, read via
  `OpenSeesModel.computed_sections()`. The analyzer mesh is not persisted. Later extended by
  ADR 0080 B3 (`bars=` overlay on the fiber kind, gate G-E).
- A2 (2026-07-18, ratified + implemented same day): `ComputedSection(kind="fiber",
  fibers={pg: UniaxialMaterial}, GJ=None)`. `fibers=` must exactly cover the material PGs;
  argument families are validated per kind at construction. `lower_to_fiber` returns the
  existing `Fiber` primitive with one `FiberPoint` per Gauss point about the elastic centroid,
  `FiberPoint(y=ȳ_auth, z=x̄_auth)`; `GJ` defaults to `warp.GJ` and `-GJ` is always emitted.
  Gate G-D (handedness) PASSED 2026-07-18.
- Follow-up shipped: stress recovery under `disconnected="sum"`.

Full text: [../0078-section-properties-analyzer.md](../0078-section-properties-analyzer.md).
Precedence (AGENTS.md): the code wins over the ADR, and the ADR wins over this summary.
