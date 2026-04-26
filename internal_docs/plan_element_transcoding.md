# Element Transcoding — Plan

> [!note] Status
> Scoped April 2026 after Phases 0–8 landed. The Results module's
> read path (Phase 3) and all three write strategies (Phases 5/6/7/8)
> are complete for **nodal** records. Element-level records (`gauss`,
> `fibers`, `layers`, `line_stations`, `elements`) are stubbed in
> three places that all need the same missing piece: an **element-class
> response metadata table**.
>
> This plan ties the three stubs together with a single shared
> implementation, so element data flows through every read/write path
> at once.

## What's stubbed today

| Surface | Behavior on element-level records |
|---|---|
| `MPCOReader.read_gauss/fibers/layers/...` | Returns empty slabs |
| `RecorderTranscoder` (Phase 6, txt) | Skips element records silently |
| `DomainCapture.step()` (Phase 7) | Raises `NotImplementedError` |
| `Recorders.gauss/fibers/...` (Phase 4 declaration) | **Works** — declared records make it into the spec |
| Phase 5 emission, Phase 8 emission | **Works** — emits the recorder commands |

Recording works end-to-end through OpenSees and into MPCO/.out files.
The gap is on the **read/decode** side.

## What's missing — the metadata table

For each `(opensees_class_name, integration_rule, response_token)` tuple
we need:

| Field | Example for `FourNodeTetrahedron` + `stress` |
|---|---|
| `n_gauss_points` | 1 |
| `gauss_natural_coords` | `[(0.25, 0.25, 0.25)]` (parent space, normalized to `[-1, +1]` per MPCO conventions) |
| `n_components_per_gp` | 6 |
| `component_layout` | `("stress_xx", "stress_yy", "stress_zz", "stress_xy", "stress_yz", "stress_xz")` |
| `flat_size_per_element` | `n_gp × n_components_per_gp = 6` |

For `globalForce` on a 3-node beam with 6 DOFs/node: `flat_size_per_element = 18`,
component layout `(force_x, force_y, force_z, moment_x, moment_y, moment_z)` × 3 nodes.

For section/fiber responses, the table grows: layout depends on the
section assignment, not just the element class.

The catalog spans ~60 element classes with ~5 integration rules and ~10
common response tokens. Roughly **~3000 distinct entries**, but most
are derivable from a small set of primitives (Gauss-Legendre point
sets, standard component orderings).

## Scope

### Phase 11a — Element response metadata + unflattening core

Three deliverables, sequenced:

#### Step 1 — Build the metadata table

**New file:** `src/apeGmsh/solvers/_element_response.py`

```python
@dataclass(frozen=True)
class ResponseLayout:
    """How a single ``ops.eleResponse(eid, token)`` flat array unflattens."""
    n_gauss_points: int
    natural_coords: ndarray         # (n_GP, dim) in [-1, +1]
    n_components_per_gp: int
    component_layout: tuple[str, ...]   # canonical apeGmsh names

# Catalog: (opensees_class, int_rule, response_token) → ResponseLayout
RESPONSE_CATALOG: dict[tuple[str, int, str], ResponseLayout] = {
    ("FourNodeTetrahedron", 1, "stress"): ResponseLayout(
        n_gauss_points=1,
        natural_coords=np.array([[0.25, 0.25, 0.25]]),
        n_components_per_gp=6,
        component_layout=STRESS,    # from _vocabulary.py
    ),
    ("stdBrick", 402, "stress"): ResponseLayout(
        n_gauss_points=8,
        natural_coords=_GL_2x2x2_HEX,
        ...
    ),
    # ... ~60 entries
}

def lookup(class_name: str, int_rule: int, token: str) -> ResponseLayout:
    """Catalog access with helpful error messages."""

def unflatten(flat: ndarray, layout: ResponseLayout) -> dict[str, ndarray]:
    """Convert a (T, E_g, flat_size) array into per-component (T, E_g, n_GP)."""
```

The catalog is the bulk of the work. Cross-reference:
- `mpco-recorder` skill's `references/element-compatibility.md`
- `opensees-expert` skill's class-tag table in `classTags.h`
- Native-coords for Gauss-Legendre rules (from
  `references/integration-rules-and-gauss.md`)

**Coverage target for v1**: every element class currently in
apeGmsh's `_ELEM_REGISTRY` (~16 classes) plus the common solid/shell
types most users hit with MPCO. Other classes raise a clear "not yet
catalogued" error pointing to this file.

#### Step 2 — Wire into the three stubs

For each existing stub site, replace the empty/raise behavior with:
1. Look up the response layout for the record's element class
2. Read the flat data (file column, MPCO META unflatten, or
   `ops.eleResponse` return)
3. Reshape to `(T, E_g, n_GP_g)` per component
4. Write into the native HDF5's `gauss_points/group_<n>/<component>`
   datasets (or `fibers/`, `layers/`, etc.)

**Modified files:**
- `src/apeGmsh/results/readers/_mpco.py` — `read_gauss/fibers/layers/...`
  parse MPCO META, look up layout, extract canonical components
- `src/apeGmsh/results/transcoders/_recorder.py` — replicate the same
  flow against `.out` columns (no META, layout looked up by class name)
- `src/apeGmsh/results/capture/_domain.py` — replace
  `NotImplementedError` with per-element loop calling
  `ops.eleResponse(eid, token)` and unflatten

#### Step 3 — Tests

- Per-class unit tests for `lookup` and `unflatten` (synthetic
  flat arrays)
- Real-file MPCO tests: extend `test_results_mpco_real_file.py` to
  read `stress_xx` through the composite (currently asserts empty)
- TXT transcoder element test: synthetic .out with stress columns
- Domain capture element test: mocked ops.eleResponse + verify the
  unflatten round-trip

## What's NOT in this plan (out of scope)

- **Custom integration rules** beyond standard Gauss-Legendre
  (e.g. `forceBeamColumn` Lobatto with user-specified IPs). These
  store their own `GP_X` in MPCO; the catalog handles fixed rules
  only. Custom rules can be a follow-up.
- **MVLEM-family 2-D Gauss rules** (per the mpco skill, `int_rule`
  values like 200/201/202 with `CUSTOM_INTEGRATION_RULE_DIMENSION=2`).
- **State variables** (`state_variable_<n>`) — these are
  material-specific and require an additional per-material table.
- **Fiber section responses** with VARIABLE fiber count per section
  (heterogeneous patches). The MPCO `SECTION_ASSIGNMENTS` group has
  the fiber data; we'd consume it and merge with the response layout.

These are real follow-on phases (11b, 11c) but each is a small,
independent extension of the catalog approach.

## Effort estimate

| Step | LOC (src) | LOC (tests) | Time |
|---|---:|---:|---:|
| 1. Metadata catalog (16 classes × ~3 tokens) | ~600 | ~200 | 1 day |
| 2. Wire into MPCOReader (META unflatten) | ~200 | ~100 | 0.5 day |
| 3. Wire into RecorderTranscoder | ~150 | ~100 | 0.5 day |
| 4. Wire into DomainCapture | ~150 | ~150 | 0.5 day |
| 5. Real-file integration tests (extend MPCO fixture) | — | ~150 | 0.5 day |
| **Total** | **~1100** | **~700** | **~3 days** |

Most of the time is in the catalog. Once that's compiled, the three
wiring sites are straightforward — they all consume the same
`unflatten()` function with the same `(T, E_g, flat) → dict[name,
(T, E_g, n_GP)]` shape contract.

## Why this unblocks element data everywhere at once

The `unflatten()` function is the keystone. By splitting it out from
any specific path, all three reader/writer surfaces share the
implementation:

```
┌─────────────────────────────────────────────────────────────┐
│         Catalog: ResponseLayout per (class, rule, token)     │
└──────────────────┬──────────────────────────────────────────┘
                   │
       ┌───────────┴───────────┐
       │                       │
       ▼                       ▼
   unflatten()             validate_layout()
       │                       │
       ▼                       ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ MPCOReader   │  │ RecorderTrans│  │ DomainCapture│
│ (META input) │  │ (TXT cols)   │  │ (ops return) │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │
       └────────┬────────┴────────┬────────┘
                ▼                 ▼
       NativeWriter.write_gauss_group(...) etc.
                │
                ▼
       Results.elements.gauss/fibers/layers.get(...)
```

After Phase 11a, a user who runs an analysis (any of the three
strategies) and has fiber/layer/gauss recorders declared in their
spec will get the data back through the same composite API
they already use for nodes.

## Decision points before starting

1. **Catalog format**: hand-coded Python dict (proposed) vs. external
   YAML/JSON file? Dict is type-safe and IDE-friendly; external file
   is easier for non-Python contributors to extend. Lean: dict.
2. **Catalog scope**: just `_ELEM_REGISTRY`'s 16 classes, or the full
   ~60 from the mpco-recorder catalog? Latter takes ~3× longer but
   covers more user models. Lean: start with `_ELEM_REGISTRY`,
   extend on demand.
3. **Naming convention** for catalog keys: `(class_name, int_rule)`
   tuple vs. nested dict vs. parsed from MPCO's bracketed name
   format `<class_tag>-<class_name>[<int_rule>:0]`? The bracketed
   form is nice because MPCO datasets are already keyed that way;
   we'd parse on lookup. Lean: parsed bracketed form for MPCO
   reads; tuple for emit/capture (where we don't have the bracket).

## See also

- [[Results_architecture]] — architecture reference (Part I)
- `references/element-compatibility.md` (mpco-recorder skill) — the
  canonical class tag → geometry/integration-rule catalog
- `references/integration-rules-and-gauss.md` (mpco-recorder skill) —
  Gauss-point coordinate conventions
- `src/apeGmsh/solvers/_element_specs.py` — current `_ElemSpec` with
  `has_gauss/fibers/layers/line_stations` flags (Phase 4)
