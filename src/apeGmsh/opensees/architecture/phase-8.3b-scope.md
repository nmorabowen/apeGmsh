# Phase 8.3b — Recorder cluster relocation and `results/` rewire

**Status:** Scoping (May 2026). Drafted at the close of PR γ (#130) of the
Phase 8 bridge teardown.

This document scopes the work that finishes the `apeGmsh.solvers`
retirement. After PR γ, the only content left in `apeGmsh.solvers`
that isn't a one-line deprecation shim is the **recorder cluster**:

```
src/apeGmsh/solvers/Recorders.py            1208 lines
src/apeGmsh/solvers/_recorder_specs.py       473
src/apeGmsh/solvers/_recorder_emit.py        886
src/apeGmsh/solvers/_element_specs.py        353
                                            ────
                                            2920 lines
```

(`_element_response.py` is already a shim from PR #123; the canonical
home is `opensees/_response_catalog.py`.)

## 1. The cluster and its consumers

### 1a. What's in the cluster

| File | Role | Public surface |
|---|---|---|
| `Recorders.py` | **Declaration helper.** User instantiates `Recorders()`, calls `.nodes(...)`, `.gauss(...)`, `.fibers(...)`, `.layers(...)`, `.line_stations(...)`, `.modal(...)`, then `.resolve(fem, ndm, ndf)` → `ResolvedRecorderSpec`. | `Recorders` class. |
| `_recorder_specs.py` | **Resolved containers + capture infrastructure.** Output of `Recorders.resolve(...)`. Owns the `.capture(...)`, `.emit_recorders(...)`, `.emit_mpco(...)`, `.to_mpco_tcl_command()` methods. | `ResolvedRecorderSpec`, `ResolvedRecorderRecord`. |
| `_recorder_emit.py` | **Emit helpers.** Translates resolved records into OpenSees `recorder` arguments (logical → ops_args). Used by both the live driver and the Tcl/Py emit path. | `emit_logical`, `to_ops_args`, `mpco_ops_args`, `LogicalRecorder`, `line_station_gpx_path`, `_DEFERRED_CATEGORIES`, `PER_MATERIAL_STRAIN_CLASSES`. |
| `_element_specs.py` | **Element-capability map.** Per-element-class flags (`has_gauss`, `has_fibers`, `has_layers`, `has_line_stations`, `has_nodal_forces`). Drives resolve-time validation. | `_ELEM_REGISTRY`. |

### 1b. Consumers in `src/`

| File | What it imports / how it uses the cluster |
|---|---|
| `results/capture/_domain.py` | `ResolvedRecorderSpec`, `ResolvedRecorderRecord`, emit helpers. Implements `DomainCapture` (the `spec.capture(...)` context). |
| `results/live/_recorders.py` | `ResolvedRecorderSpec` + `emit_logical` / `to_ops_args`. Implements `spec.emit_recorders(out_dir)`. |
| `results/live/_mpco.py` | `ResolvedRecorderSpec` + `mpco_ops_args`. Implements `spec.emit_mpco(path)`. |
| `results/transcoders/_recorder.py` | `ResolvedRecorderSpec` + emit helpers + `_element_response.lookup`. Parses `.out` files into native HDF5. |
| `results/transcoders/_txt.py` | `LogicalRecorder` (the per-record meta carrier). |
| `results/writers/_cache.py` | `ResolvedRecorderSpec` + `_DEFERRED_CATEGORIES`. Cache-key hashing. |
| `opensees/_response_catalog.py` | `_ELEM_REGISTRY` (read-only — pure metadata lookup). |

Seven source-side consumers. All in `results/` except the response
catalog itself, which is bridge-side.

### 1c. Consumers in `tests/`

~30 tests still import `ResolvedRecorderSpec` / `ResolvedRecorderRecord`
/ `_recorder_emit` helpers / `_ELEM_REGISTRY` directly:

```
test_results_catalog_{planes,shells,solids,trusses}_real.py
test_results_domain_capture{,_fibers,_fibers_real,_gauss,_layers,_layers_real,_line_stations,_line_stations_real,_nodal_forces,_nodal_forces_real,_real}.py
test_results_live_{recorders,mpco}.py
test_results_mpco_{element_real,emit_real}.py
test_results_recorder_{emit,live_args,transcoder_gauss,transcoder_gauss_real,transcoder_line_stations,transcoder_line_stations_real,transcoder_nodal_forces,transcoder_nodal_forces_real}.py
test_results_transcoder.py
test_results_txt_parser.py
test_results_element_capabilities.py     # _ELEM_REGISTRY
```

That is the test surface that has to follow whatever migration
shape we pick.

## 2. The design tension

The legacy `Recorders` class is a **declaration helper**. Its output
`ResolvedRecorderSpec` is a **container that also owns the capture
infrastructure** — the same spec drives:

1. `.capture(path, fem, ndm, ndf)` — in-memory HDF5 capture via
   `DomainCapture` context manager. The capture story.
2. `.emit_recorders(out_dir)` — classic OpenSees `.out` files.
3. `.emit_mpco(path)` — STKO-format HDF5.
4. `.to_mpco_tcl_command()` — emit a `recorder mpco ...` Tcl line.

The **new typed primitives** at `apeGmsh.opensees.recorder` (Phase 3
of the typed-primitives rebuild) are **emission-only**:

- `Node(file=..., response=..., pg=..., dofs=...)` → emits
  `recorder Node` lines through the `Emitter` Protocol.
- `Element(file=..., response=..., pg=..., elements=...)`.
- `MPCO(file=..., ...)`.

They were designed for the typed-bridge emit pipeline (Tcl / Py / H5
emit through the same Protocol the materials, sections, elements,
etc. use). **They have no capture story** — they're leaves in the
dependency graph that produce one `recorder ...` line each.

This is the crux of Phase 8.3b. Two abstractions, both about
"recorders", both used in different places, with no overlap. The
phase has to either pick one (and reshape the other) or relocate
the legacy cluster without touching the architecture mismatch.

## 3. Two flavors of Phase 8.3b

### Flavor 1 — Pure relocation (light touch)

**Move** the four cluster files out of `solvers/` to a canonical
home. Update consumers to the new namespace. **Don't unify** the
two abstractions.

Proposed homes:

```
solvers/Recorders.py          -> results/spec/declaration.py
solvers/_recorder_specs.py    -> results/spec/_resolved.py
solvers/_recorder_emit.py     -> results/spec/_emit.py
solvers/_element_specs.py     -> opensees/_element_capabilities.py
```

(The element-capabilities map is OpenSees-class metadata; it belongs
next to the response catalog. Moving it under `opensees/` gives the
results layer a one-way import dependency, which matches what
already happens for the response catalog.)

Public re-export surface:

```python
# results/spec/__init__.py
from .declaration import Recorders
from ._resolved import ResolvedRecorderSpec, ResolvedRecorderRecord

# canonical user import:
from apeGmsh.results.spec import Recorders
```

Migration work:

1. `git mv` the four files; fix internal relative imports.
2. Update ~7 source files in `results/`. Most changes are namespace
   updates (`from ...solvers._recorder_specs import ...` →
   `from ..spec._resolved import ...`).
3. Update ~30 test files (mechanical sed-able for the most part).
4. Update the EOS notebook imports (`from apeGmsh.solvers.Recorders
   import Recorders` → `from apeGmsh.results.spec import
   Recorders`). 9 notebooks, one-line edit each — exactly the kind
   of follow-up the EOS notebooks already flagged with a TODO
   comment.
5. Leave deprecation shims at the old paths for one release cycle,
   same envelope pattern as PRs #119 / #121 / #123.

**Risk:** Low. Mechanical relocation. No behavior change.

**Cost:** ~50 file edits but most are find-replace.

**End state:** `apeGmsh.solvers` is just a deprecation envelope.
The recorder declaration API and capture flow are unchanged from
the user's perspective; they just import from a new location.

### Flavor 2 — Architectural unification (heavier)

Reshape the recorder layer so there's **one** API for both
declaration and emission. The two abstractions become one.

This is more invasive. Two sub-options:

**2a. Promote the typed primitives to declaration.** The user
declares with the typed primitives (`Node(file=..., ...)` etc.); the
capture infrastructure is rebuilt on top so that calling
`bm.capture(path)` on a `BuiltModel` records the same data the
typed primitives would otherwise emit to a file.

- Pros: One API. Charter-aligned (typed-primitive-first).
- Cons: The capture flow has to be re-implemented. The current
  capture introspection (`getNodeResponse`, etc.) is keyed off the
  resolved-spec records — rewriting that against typed primitives
  is non-trivial. The user-facing declaration shape changes.

**2b. Keep the declaration helper, retire only the spec container.**
The user keeps `Recorders().nodes(...).resolve(...)`, but the
resolver returns a **list of typed primitives** instead of a
`ResolvedRecorderSpec`. The capture / emit methods move onto a new
`RecorderSession` or similar that takes a list of primitives.

- Pros: User-facing declaration unchanged.
- Cons: The internal data shape changes substantially. `results/`
  has to be rewired to consume primitives instead of records.
  Two-step migration — the deprecation period would be awkward.

Both 2a and 2b require **redesigning the capture API**, which is a
substantial architectural decision that should land on its own.
This is what I would defer to a follow-up scoping conversation
after Flavor 1 ships.

### Recommendation

**Ship Flavor 1 in Phase 8.3b**. It empties `apeGmsh.solvers` and
unblocks Phase 8.4 (model.h5 zone reshuffle), Phase 8.5 (broker
h5 writers), and Phase 8.7 (viewer migration). The architectural
unification of the recorder layer can be its own project that lands
later — there is no hard dependency on it from any other Phase-8
sub-phase.

## 4. Phase 8.3b sub-commits (Flavor 1)

If the maintainer picks Flavor 1, the commit shape:

### Commit 1 — relocate `_element_specs.py` → `opensees/_element_capabilities.py`

Smallest of the four. Mechanical move + shim. ~2 consumers
(`Recorders.py` which we move next, and `opensees/_response_catalog.py`
which is already in the right place).

### Commit 2 — relocate `_recorder_emit.py` → `results/spec/_emit.py`

~5 source consumers in `results/`. Update their imports.

### Commit 3 — relocate `_recorder_specs.py` → `results/spec/_resolved.py`

The big container types. Drives the capture / emit / mpco flows.
Touch every results-layer consumer.

### Commit 4 — relocate `Recorders.py` → `results/spec/declaration.py`

The user-facing declaration API.

### Commit 5 — `results/spec/__init__.py` umbrella + deprecation shims

Build the public umbrella so users can `from apeGmsh.results.spec
import Recorders`. Drop deprecation shims into the old
`solvers/Recorders.py`, `_recorder_specs.py`, `_recorder_emit.py`,
`_element_specs.py` paths — same envelope pattern as PRs #119 /
#121 / #123.

### Commit 6 — migrate the 9 EOS notebooks' `Recorders` import

One-line edit per notebook (the canonical-home-moves-in-Phase-8.3b
comment we left in PR #125 now resolves).

### Commit 7 — migrate the ~30 test files

Bulk find-replace across the test surface. Most files just need
their `from apeGmsh.solvers.X import Y` line rewritten.

### Final state

```
src/apeGmsh/solvers/
├── __init__.py        # deprecation envelope (__getattr__ for relocated names)
├── _kinds.py          # shim → mesh.records._kinds
├── _constraint_records.py  # shim
├── _constraint_defs.py     # shim
├── _constraint_geom.py     # shim
├── _constraint_resolver.py # shim
├── _consistent_quadrature.py  # shim
├── _opensees_csys.py  # shim
├── _element_response.py  # shim
├── Constraints.py     # shim
├── Loads.py           # shim
├── Masses.py          # shim
├── Numberer.py        # shim
├── Recorders.py       # shim (new in 8.3b)
├── _recorder_specs.py # shim (new in 8.3b)
├── _recorder_emit.py  # shim (new in 8.3b)
└── _element_specs.py  # shim (new in 8.3b)
```

After one release cycle of deprecation, Phase 8.8 deletes the whole
`solvers/` directory.

## 5. Verification gates (per commit)

Same as previous Phase-8 PRs:

- `mypy --strict src/apeGmsh/`
- `ruff check src/apeGmsh/ tests/`
- `pytest -m "not live and not subprocess"`

Each commit's verification: no new errors / regressions relative to
the post-PR-γ baseline (`1424 mypy / 219 ruff / 4456 passed`).

Special check: `import apeGmsh` and `import apeGmsh.solvers` continue
to emit zero DeprecationWarnings — every internal apeGmsh consumer
should already be migrated to the new path by the same commit that
introduces the shim.

## 6. Open questions for the implementing session

1. **`results.spec` vs `opensees.recorder` for the canonical home.**
   `results.spec` is the natural fit since the spec is consumed by
   the results layer. `opensees.recorder` would parallel the typed
   primitives. Flavor 1 picks `results.spec` because Flavor 2 (later)
   may want to merge with `opensees.recorder` and we shouldn't claim
   that namespace prematurely.

2. **Should `_element_specs` move under `opensees/` or `results/`?**
   Recommended: `opensees/_element_capabilities.py`. The map is
   OpenSees-class metadata. `Recorders` consults it during resolve;
   `opensees/_response_catalog.py` already consumes element-class
   tags. Putting both under `opensees/` keeps the results layer's
   dependencies one-way.

3. **`_DEFERRED_CATEGORIES` placement.** This is a string-set
   constant used by `results/writers/_cache.py` and the typed
   recorder primitives. Should it live with the spec or in a shared
   "recorder vocabulary" module? Probably with the spec for now —
   the typed primitives don't currently import it.

4. **Migration window for the 9 EOS notebooks.** The notebooks
   imported from `apeGmsh.solvers.Recorders` with a TODO comment
   pointing at this phase. The Phase 8.3b PR should land the
   notebook updates in the same commit-stream so the curriculum
   doesn't see a half-migrated state on `main`.

## 7. Out of scope (defer to later phases)

- **Architectural unification of `Recorders` with the typed primitives**
  (Flavor 2 above). Worth its own scoping conversation after 8.3b.
- **model.h5 zone reshuffle (Phase 8.4)** — independent of the
  recorder cluster move. Can land before or after 8.3b.
- **Viewer migration off `FEMData` / `solvers` (Phase 8.7)** —
  consumes the post-8.4 schema. Lands after both 8.3b and 8.4.

## References

- [phase-8-untangle.md](phase-8-untangle.md) — master plan
- [decisions/0013-records-in-mesh-not-solvers.md](decisions/0013-records-in-mesh-not-solvers.md) — Phase 8.1 ADR
- PR #119 / #120 / #121 / #123 / #124 / #125 / #130 — landed
  Phase-8 work the cluster relocation will mirror.
