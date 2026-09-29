# ADR 0020 — `Results` carries `OpenSeesModel` via the Composed-file pattern; viewer stays file-mediated

**Status:** Accepted (Phase 4 of the major architectural refactor,
May 2026). Builds on [ADR 0019](0019-opensees-model-read-side-broker.md);
preserves [ADR 0014](0014-viewer-is-pure-h5-consumer.md); preserves
[ADR 0011](0011-h5-as-fourth-emit-target.md).

## Context

`Results` today carries `(reader, fem, stage_id, path)`. The
`model_h5=` kwarg on `Results.viewer(...)` is pure plumbing — it is
passed through to `ResultsViewer.__init__` and is **not** stored on
`Results`. The viewer side already has the seam:

- Auto-resolve at `viewers/results_viewer.py:256-292` peeks at
  `results._path` for `has_opensees_orientation` — but only works when
  the results file happens to also be a model.h5 (i.e. the bridge ran
  with `out_dir=`/`save_to=` collapsing the two). When the model and
  results are separate files, the user must pass `model_h5=` by hand.
- Cuts auto-load (`viewers/results_viewer.py:1421-1473`) is gated on
  the **explicit** `model_h5=` kwarg, never on `results._path`.
  Asymmetric with the orientation auto-resolve and easy to forget.

The user wants the chain to flow forward, naming it explicitly:

```
Results.model ──→ OpenSeesModel.fem ──→ FEMData
```

Reading `results.model` should answer every structural question the
viewer needs, without a separate `model_h5=` kwarg threaded through
every constructor and every subprocess hop.

This must be reconciled with two prior ADRs that constrain the
solution:

1. **ADR 0011** — *"the bridge does not read its own H5 back."* The
   read happens on `OpenSeesModel`, never `apeSees`. (Per ADR 0019.)
2. **ADR 0014's AST guard** — `apeGmsh.viewers.*` imports nothing from
   `apeGmsh.opensees.*` or `apeGmsh.mesh.*` internals, only
   `apeGmsh.results` and `apeGmsh.opensees.emitter.h5_reader`. The
   viewer cannot hold a Python `OpenSeesModel` handle directly; it
   consumes through h5_reader against a file path.

The two together force a *file-mediated* design: `Results` carries the
`OpenSeesModel` handle in-process; the viewer reads the same file the
`OpenSeesModel` was rehydrated from, through `h5_reader`. Symmetric
with how `ViewerData.from_h5` already works (per ADR 0014).

## Decision

### `Results._model` — additive field, then required (phased migration)

`Results` gains an optional field:

```python
@dataclass(frozen=True)
class Results:
    _reader: ...
    _fem: FEMData
    _stage_id: int | None
    _path: pathlib.Path
    _model: OpenSeesModel | None = None      # NEW in Phase 4
```

`Phase 4` makes this **additive** — every existing construction path
keeps working with `_model=None`. `Phase 8` (prune) flips it to
**required** — `_model` cannot be `None` for any constructor that owns
the chain (`from_native`, `from_mpco`, `from_recorders`). The
intermediate phase emits no `DeprecationWarning`; once `_model` is
threaded through the three constructors, the kwarg ergonomics is the
public surface.

### Three constructors accept `model=`

| Constructor | Signature gain | Auto-resolve rule |
|---|---|---|
| `Results.from_native(path, *, model=None)` | `model=` optional | If `model` not supplied AND the results file carries an embedded `/opensees/` zone (probe via `has_opensees_orientation`, per the precedent at `viewers/data/_h5_probe.py`), load via `OpenSeesModel.from_h5(path)`. |
| `Results.from_mpco(path, *, fem=None, model_h5=None, merge_partitions=True)` | `model_h5=` path kwarg | Closes the deferred MPCO vecxz seam (skill §7.2). When `model_h5=` is supplied, load `OpenSeesModel.from_h5(model_h5)`; MPCO data merges with it. No auto-resolve from the MPCO file itself (MPCO has no `/opensees/` zone — `project_mpco_no_vecxz`). |
| `Results.from_recorders(spec, output_dir, *, fem, model=None)` | `model=` optional | `spec` already carries enough for recorder-resolution; `model=` is the optional add-on for downstream viewer auto-resolve. |

The asymmetry (path string for MPCO, object for native/recorders) is
deliberate: MPCO is the third-party-file case (the user did not write
this file from apeGmsh), and the convenient action is *"point me at a
sibling model.h5."* For native/recorders, the user already has a
Python `OpenSeesModel` handle (or will, post-Phase 8) — passing it
directly avoids a useless re-walk.

### Composed file — `NativeWriter` extended to embed `/opensees/`

`NativeWriter` (`results/writers/_native.py`) is extended to write the
full `/opensees/` zone into `results.h5` during initial open. One
`results.h5` carries:

```
results.h5
├── /meta            schema + lineage attrs
├── /<run data>      results time-series, slabs, components
└── /opensees/       full bridge zone — the OpenSeesModel
    ├── /opensees/materials
    ├── /opensees/sections
    ├── /opensees/transforms
    ├── /opensees/element_meta
    ├── /opensees/patterns
    └── ...
```

This is the **Composed-file pattern**: one self-contained artifact
that answers both *"what did you measure?"* and *"on what model did
you measure it?"* The redundancy with a sibling `model.h5` is
intentional — peer review, archival, sharing all use one file.

The composition happens at `NativeWriter.open(...)`, not at `close`,
because h5py file fragmentation is materially worse if the bulk of the
file lands during the close fsync. Documented for the Phase 4
implementation.

### Viewer — drop `model_h5=` kwarg, file-mediated

`Results.viewer(...)` drops the `model_h5=` kwarg:

- **Phase 5** — emit `DeprecationWarning` from `Results.viewer` when
  `model_h5=` is passed; suggest *"`Results` now carries the model
  natively; remove `model_h5=`."*
- **Phase 8** — remove the kwarg entirely. `ResultsViewer` reads
  `results.model` (the `OpenSeesModel` handle).

The subprocess hop (when `blocking=False`) sends the **results path
only**. The spawned viewer process re-opens via
`Results.from_native(path)`, which re-rehydrates the embedded
`OpenSeesModel` via the `/opensees/` zone in the same file. The
viewer never holds an in-process `OpenSeesModel` handle directly — it
consumes through `h5_reader` and `viewers/data/_viewer_data.py`'s
`from_h5(path)` builder, preserving ADR 0014's AST guard verbatim.

### Invariants

**INV-1.** `Results._model is None` is legal in Phase 4 and Phase 5.
It is **illegal** post-Phase 8: every Results-side constructor owns
the chain. The Phase 8 prune deletes any code path that branches on
`_model is None`.

**INV-2.** The viewer never imports `OpenSeesModel` directly. ADR
0014's AST guard is unchanged; the viewer's only structural read
surface remains `viewers/data/ViewerData` (via
`apeGmsh.opensees.emitter.h5_reader`). The Python `OpenSeesModel`
handle lives only on `Results` and is consumed in-process by code that
is **not** under `apeGmsh.viewers`.

**INV-3.** `Results.from_mpco(path, model_h5=path)` does **not** copy
the model zone into a derived `results.h5`. MPCO is the third-party
case; the `OpenSeesModel` is loaded from the sibling file the user
pointed at, in memory only. The Composed-file pattern applies to
native results writing, not to MPCO loading.

**INV-4.** Subprocess transport carries the **results path**, not the
`OpenSeesModel` object. Pickling `OpenSeesModel` across the
subprocess boundary is forbidden — the spawned viewer rehydrates from
file. This is symmetric with how the cuts subprocess hop already
works.

**INV-5.** Cuts auto-load and orientation auto-resolve become
symmetric: both gated on `results._model is not None`. The asymmetric
"explicit kwarg for cuts, auto-resolve for orientation" split is
removed.

## Alternatives considered

| Alternative | Why rejected |
|---|---|
| **HDF5 `ExternalLink` from `results.h5` to a sibling `model.h5`** | Fragile if the files separate (the collaborator-emails-me-the-results case). Adds a path-resolution failure mode at every viewer open. The Composed-file pattern's one-file-rules-all property is precisely the value. |
| **Path string + `bind(fem)`-style call** | This is the current pairing problem dressed up. The whole point of *"Results carries Model"* is to make pairing **structural**, not procedural. The user memory `project_bind_contract` already locks the May 2026 rejection of re-introducing `BindError`-style enforcement; replacing it with a different bind shim regresses to the same problem. |
| **Mandatory `model=` at construction in one PR (skip the additive phase)** | Breaks 5864 existing tests in one PR. The phased migration was explicitly chosen so that the Phase 4 work is isolated from the test rewrites, and the Phase 8 prune is a separate reviewable PR. |
| **Carry the path string only; lazy-load `OpenSeesModel` on `results.model` access** | Surface ambiguity: *is `results.model` a path, an object, or None?* Eager load (when the zone is present) keeps the property type stable (`OpenSeesModel | None`). The cost of eager rehydration is acceptable — `OpenSeesModel.from_h5` is fast on typical models, and the handle is immutable so it caches trivially. |
| **`ResultsViewer` reads the `OpenSeesModel` Python handle directly across the subprocess boundary** | Breaks ADR 0014's AST guard. The viewer is a *pure h5 consumer*; the `OpenSeesModel` handle stays on the `apeGmsh.results` side. Subprocess transport is by file path. |
| **Two-file design: `results.h5` + `model.h5` as paired siblings, no embedding** | Loses the *"hand someone one file and they have everything"* property. The HDF5 file is fundamentally an archive format; treating it as such means a results file should be self-sufficient. |

## Consequences

**Positive:**

- The chain forward — `Results → OpenSeesModel → FEMData` — is named
  in the type system. Every structural question the viewer asks has
  one answer source.
- The `model_h5=` plumbing kwarg disappears from the viewer public
  API. The viewer's surface shrinks; the user holds one object
  (`Results`) instead of three.
- ADR 0014 preserved without amendment. Viewer remains a pure
  `h5_reader` consumer; subprocess transport is file-mediated.
- ADR 0011 preserved without amendment. The read goes through
  `OpenSeesModel`, not `apeSees`.
- One self-contained `results.h5` artifact (Composed-file pattern).
  Peer review, archival, sharing all use one file. Lineage (ADR
  0021) annotates the same single file.
- MPCO vecxz gap (skill §7.2) closes for free via the new
  `model_h5=` kwarg on `from_mpco`. The deferred MPCO orientation
  workflow is no longer deferred.
- Cuts auto-load and orientation auto-resolve become symmetric (INV-5).
  One gate (`results._model is not None`), one mental model.

**Negative:**

- `results.h5` file size grows. The full `/opensees/` zone is
  duplicated alongside results. Typical models are KB-to-MB; the
  redundancy buys reproducibility. Accepted. Users who explicitly
  want the two-file split can still call `apeSees.h5(model_path)`
  separately and use `from_mpco(results_path, model_h5=model_path)`.
- Composed-file write must happen during `NativeWriter.open` (h5py
  file-fragmentation perf). Documented in Phase 4 implementation;
  the alternative (writing at close) was measured to be materially
  worse on large runs.
- Phased migration takes three PRs (Phase 4 additive, Phase 5
  deprecation, Phase 8 prune) instead of one. Accepted: the breakage
  surface for the 5864-test suite is bounded per PR.

## Open questions

- **Q1 — MPCO + sidecar after Phase 8.** Resolved in this ADR: drop
  the convenience desugaring. Users write
  `Results.from_mpco(path, model=OpenSeesModel.from_h5("model.h5"))`
  explicitly. One line, no two-ways-to-do-it.

## References

- [decisions/0011-h5-as-fourth-emit-target.md](0011-h5-as-fourth-emit-target.md)
  — H5 strictly as emit output; preserved unchanged.
- [decisions/0014-viewer-is-pure-h5-consumer.md](0014-viewer-is-pure-h5-consumer.md)
  — the AST guard INV-2 preserves; the file-mediated subprocess hop
  this ADR extends.
- [decisions/0019-opensees-model-read-side-broker.md](0019-opensees-model-read-side-broker.md)
  — the `OpenSeesModel` class this ADR makes `Results` carry.
- [decisions/0021-lineage-chain-replaces-snapshot-id.md](0021-lineage-chain-replaces-snapshot-id.md)
  — the `lineage` annotation written into the Composed file.
- [phase-8-untangle.md](../phase-8-untangle.md) §7 closure — the
  `model_h5=` kwarg fate this ADR resolves.
- User memory `project_bind_contract` — the May 2026 ratification
  that snapshot_id is never enforced; this ADR's structural-pairing
  approach replaces procedural-bind.
- User memory `project_mpco_no_vecxz` — the MPCO seam this ADR closes
  via `from_mpco(model_h5=)`.
