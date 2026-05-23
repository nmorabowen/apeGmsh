# ADR 0026 — `H5ModelReader` Protocol contract for viewer-side model access

**Status:** Accepted (May 2026, head-engineer review on branch
`guppi/suspicious-babbage-5b8b0a` / PR #282). Successor work to
ADR 0014 (viewers pure-h5 consumer), ADR 0019 (`OpenSeesModel`
read-side broker), and ADR 0020 (Results carries `OpenSeesModel`
via Composed file). Codifies the implicit contract Phase 8 left
undocumented and unblocks the elimination of the last
`model_h5`-as-path survivor inside the viewer subpackage. Adoption
ships as PRs 7-a through 7-d (sequenced under §Decision).

## Context

### What the three-broker refactor closed, and what it left open

ADR 0020 collapsed the public `Results.viewer(model_h5=...)` kwarg
and routed the post-solve viewer through the in-process chain
`Results.model.fem`. Phase 8 deleted `Director.set_model_h5()`,
`BindError`, `EXPECTED_SCHEMA_MAJOR`, and the dict-style `h5_reader`
accessors. The viewer's broker-side import surface is now exactly
one allowed module — `apeGmsh.opensees.emitter.h5_reader` — guarded
by an AST sweep at `tests/viewers/test_viewers_pure_h5_consumer.py`.

That sweep enforces *which* module the viewer reads from, not *what
shape* that module exposes. The implicit contract has three live
consumers:

1. `apeGmsh.viewers.data.ViewerData.from_h5(path)` — opens via
   `h5_reader.open(path)` and decodes nodes / elements / loads /
   masses / constraints / mesh_selections / per-element vecxz
   (`_viewer_data.py:144-225`).
2. `apeGmsh.viewers.diagrams._director.ResultsDirector` — holds a
   `_model_h5: Optional[Path]` and lazily builds a `FemToOpsTagMap`
   via `FemToOpsTagMap.from_h5(self._model_h5)` (`_director.py:250-283`).
3. `apeGmsh.viewers.data._h5_probe.has_opensees_orientation(path)` —
   opens with raw h5py to check for `/opensees/transforms` +
   `/opensees/element_meta` (the orientation sidecar gate for the
   results viewer).

All three reach into the file through different APIs (typed
`H5Model`, path-only factory, raw h5py probe). The contract that
unifies them — *what does "the model side of a viewer-readable file"
look like* — exists only as field knowledge spread across three call
sites and the schema document at
[`opensees/architecture/h5-schema.md`](../h5-schema.md).

### The last path-survivor in the chain

`ResultsDirector._bind_model_h5(path)` and the
`director.tag_map` property are the only viewer-side surfaces that
still treat `model.h5` as a file path rather than an opened broker.
The four-agent visualization assessment (May 2026, see post-PR
audit summary) ranked this as debt D5: every time
`ResultsDirector` is bound, the viewer does both
`director.set_model(results.model)` (the chain-forward `OpenSeesModel`
handle) **and** `director._bind_model_h5(results._path)` (the file
path for `FemToOpsTagMap.from_h5`). They always travel together
because the two consumers expect different inputs.

The shape of the duplication (D1 from the same audit) is the same:
`results_viewer.py:256-263` and `:1414-1420` both open the file with
`has_opensees_orientation(...)`, both fall back to `from_fem` on the
no-orientation path, and neither names the resolution as a function.

### The research trajectory

The user's research portfolio (LS-DYNA, OpenSees nonlinear FEM,
seismic SSI, ground motion processing) is going to encounter
non-apeGmsh result formats. The Theory Manual sub-skill for LS-DYNA
is already loaded; `d3plot` consumption is on the realistic horizon.
xDMF / Exodus appear in the broader computational mechanics
ecosystem (Code_Aster, MOOSE, SfePy). The current `H5Model` is
apeGmsh-schema-specific by construction.

A formal contract on what the viewer needs from a model reader makes
those adapters drop-in. Without it, every new reader is a new
implicit duck-type matched against three call sites.

### Why this is an ADR, not a refactor

The widening events for `Emitter.region` (ADR 0024) and
`Emitter.eigen` (ADR 0025) set the precedent: when a Protocol that
multiple call sites already depend on grows a new method, that
growth is "an architecture event" worth recording. The reverse case
applies here. An *implicit* Protocol that multiple call sites
already depend on is one method-rename away from a silent break.
Codifying it as `H5ModelReader` before reorganizing
`ResultsDirector` is what makes PR7 a controlled refactor instead of
a brittle one.

## Decision

### Define the `H5ModelReader` Protocol

Add `apeGmsh.viewers.data._protocol.H5ModelReader` — a typing
`Protocol` (not an ABC) that captures the viewer's read contract
against an opened model file.

```python
from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol


class H5ModelReader(Protocol):
    """The viewer-side read contract for a model archive.

    Adapters: ApeGmshComposedReader, ApeGmshNeutralReader,
    MpcoWithSiblingReader, and (future) ExodusReader / D3PlotReader
    / XdmfReader. Implementing this Protocol is sufficient to feed
    ViewerData.from_reader() and ResultsDirector.bind_results().

    Not a write-side contract. The bridge writes through
    apeSees(fem); ModelData(fem) writes the orientation sidecar.
    See ADR 0011 (H5 as fourth emit target) and ADR 0018
    (ModelData enrichment).
    """

    # -- Identity & metadata --------------------------------------------

    @property
    def path(self) -> Optional[Path]:
        """The on-disk source of the reader.

        ``None`` for synthesised / in-memory adapters (e.g. fixture
        builders that compose a model without ever touching disk).
        """

    @property
    def schema_version(self) -> str:
        """Per-zone schema string (e.g. "2.7.0") of the bridge zone."""

    def meta(self) -> Mapping[str, Any]:
        """Bridge ``meta`` group attributes (ndm, ndf, snapshot_id, ...)."""

    # -- Capability probes ----------------------------------------------

    def has_opensees_orientation(self) -> bool:
        """True iff ``/opensees/transforms`` + ``/opensees/element_meta``
        are both present. The orientation-sidecar gate the results
        viewer uses to decide between ``ViewerData.from_h5_model``
        (oriented path) and ``ViewerData.from_fem`` (degraded path).
        Replaces ``apeGmsh.viewers.data._h5_probe.has_opensees_orientation``.
        """

    def has_neutral_zone(self) -> bool:
        """True iff the file carries the neutral zone (nodes / elements
        / pgs / labels / loads / masses / constraints). Always True
        for apeGmsh-written files; may be False for foreign formats
        until their adapter materialises a neutral view."""

    # -- Neutral-zone access (Schema 2.6.0+) -----------------------------

    def nodes(self) -> Mapping[str, Any]:
        """Returns ``{ids, coords}`` arrays (or empty dict if absent)."""

    def elements(self) -> Mapping[str, Mapping[str, Any]]:
        """Returns ``{type_token: {ids, conn, ...}}`` keyed by element type."""

    def physical_groups(self) -> Mapping[str, Mapping[str, Any]]:
        """Returns ``{name: {dim, tag, node_ids, element_ids, ...}}``."""

    def labels(self) -> Mapping[str, Mapping[str, Any]]: ...
    def mesh_selections(self) -> Mapping[str, Mapping[str, Any]]: ...

    def loads(self) -> Mapping[str, Mapping[str, Any]]: ...
    def masses(self) -> Any: ...
    def constraints(self) -> Mapping[str, Any]: ...

    # -- Bridge-zone access (Schema 2.7.0+) ------------------------------

    def element_meta(self) -> Mapping[str, Mapping[str, Any]]:
        """``/opensees/element_meta/{type_token}/`` per-element bridge
        records. Empty mapping if the reader has no orientation zone."""

    def element_meta_arrays(self, type_token: str) -> Mapping[str, Any]:
        """Per-type-token bridge arrays — the ``ids`` (OpenSees tag),
        ``fem_eids`` (FEM eid) channel that ``FemToOpsTagMap`` consumes."""

    def element_local_axes_vecxz(self) -> Mapping[int, Any]:
        """Per-FEM-eid local-axes vecxz vector. Empty for
        mesh-only / pre-bridge archives — no error."""

    # -- Lifecycle -------------------------------------------------------

    def close(self) -> None:
        """Release any underlying file handle. Idempotent."""

    def __enter__(self) -> "H5ModelReader": ...
    def __exit__(self, *exc: object) -> None: ...
```

### Two-class adoption: zero-cost wrap, then refactor

**Phase A — Wrap existing reader (no behaviour change).**

`apeGmsh.opensees.emitter.h5_reader.H5Model` already satisfies most
of the Protocol. The only structural deltas are:

- Expose `path` as a property (currently `self._path` is private).
- Expose `has_opensees_orientation()` and `has_neutral_zone()` as
  methods (currently the orientation probe lives in
  `viewers/data/_h5_probe.py`).

These additions are additive on `H5Model`; no existing call site
changes. The Protocol is satisfied structurally — no inheritance
relation needed (this is what `typing.Protocol` is *for*).

**Phase B — `ViewerData.from_reader(reader: H5ModelReader)`.**

```python
class ViewerData:
    @classmethod
    def from_reader(cls, reader: H5ModelReader) -> "ViewerData":
        """Build from any H5ModelReader adapter."""
        return cls.from_h5_model(reader)  # one-line shim

    @classmethod
    def from_h5(cls, path: str) -> "ViewerData":
        """Open a file via the reference apeGmsh reader.

        Convenience for the common case. Internally constructs an
        ApeGmshComposedReader and delegates to from_reader.
        """
        from apeGmsh.opensees.emitter import h5_reader
        with h5_reader.open(path) as reader:
            return cls.from_reader(reader)
```

Existing `ViewerData.from_fem` and `ViewerData.from_h5_model` stay —
they are the synchronous / in-memory paths. `from_h5(path)` becomes
the convenience entry point that goes through the Protocol.

**Phase C — Director consumes a reader, not a path.**

```python
class ResultsDirector:
    def bind_results(self, results) -> None:
        """Bind a Results handle, derive the H5ModelReader from
        results._reader / results._model._reader, and pre-build the
        tag_map lazily on first access.

        Replaces the dual ``set_model(results.model)`` +
        ``_bind_model_h5(results._path)`` ceremony with one call.
        ``_bind_model_h5`` survives as a private back-compat alias
        for one release cycle, then is deleted.
        """
        self._opensees_model = results.model
        self._reader = _derive_reader(results)
        self._tag_map_cache = None

    @property
    def tag_map(self) -> "Optional[FemToOpsTagMap]":
        if self._reader is None:
            return None
        if self._tag_map_cache is None:
            from apeGmsh.cuts import FemToOpsTagMap
            self._tag_map_cache = FemToOpsTagMap.from_reader(self._reader)
        return self._tag_map_cache
```

`FemToOpsTagMap.from_reader(reader: H5ModelReader)` is the new
canonical factory; `FemToOpsTagMap.from_h5(path)` becomes a thin
shim that constructs the reader and forwards. The `cuts/` subpackage
keeps its `apeGmsh.opensees.emitter.h5_reader` import (no change to
its dependency graph — it already depends on the same reader).

`_bind_model_h5(path)` survives one release cycle for session-restore
paths that serialise a path string. It internally constructs a
reader and calls `bind_results` against a synthetic Results. After
the cycle, it is deleted along with the `Optional[Path]` field on
the director.

### Migration order

| PR | Scope | Effect |
|----|-------|--------|
| **PR7-a** | Add `H5ModelReader` Protocol + extend `H5Model` with `path` property and the two `has_*` probes. | No behaviour change. AST guard at `test_viewers_pure_h5_consumer.py` still green (no new imports inside `viewers/`). |
| **PR7-b** | `FemToOpsTagMap.from_reader(reader)` factory; `from_h5(path)` becomes a shim. | No call-site change. New tests bind a `FemToOpsTagMap` from a `H5Model` directly. |
| **PR7-c** | `ViewerData.from_reader(reader)` + collapse `_h5_probe.has_opensees_orientation` into `H5ModelReader.has_opensees_orientation()`. | `_h5_probe.py` becomes a one-function shim that opens with the reader and forwards; can be deleted after PR3 collapses the duplicated resolver (D1). |
| **PR7-d** | `ResultsDirector.bind_results(results)` + deprecate `_bind_model_h5(path)`. `results_viewer.py` switches from dual-bind to single-bind. | Closes D5. Two duplicated probe sites in `results_viewer.py` collapse to one. |
| **PR7-e** *(future, not on this ADR)* | First foreign-format adapter (`D3PlotReader` or `XdmfReader`) implementing the Protocol. | Validates the contract under a second user. ADR 0014 INV-1 must be re-affirmed by the AST guard for the new adapter's import surface. |

PR7-a through PR7-d ship behind the existing `tests/viewers/` suite
(1124 passing as of this ADR's draft). PR7-e is a separate
architectural event with its own ADR — this one closes only the
Protocol-definition decision.

## Invariants

- **INV-1** — `H5ModelReader` is structural. Implementing classes
  do **not** subclass it; conformance is by `typing.Protocol`.
  This matches ADR 0014's rule that `viewers/` imports nothing from
  `apeGmsh.opensees.*` beyond `emitter.h5_reader`.
- **INV-2** — The Protocol is consumed inside `apeGmsh.viewers/` and
  `apeGmsh.cuts/` only. `apeGmsh.opensees/` does **not** depend on
  the Protocol — it implements it incidentally on `H5Model`. This
  preserves ADR 0019 INV-4 (no module-level edge from
  `apeGmsh.opensees` to `apeGmsh.mesh`).
- **INV-3** — `H5ModelReader.path` may return `None`. Callers that
  require a filesystem path (e.g. for `subprocess` spawn) must check
  and fall back, not assume. The existing `Results._spawn_viewer_subprocess`
  contract is unaffected because the spawn site reads `Results._path`
  directly, not through any reader.
- **INV-4** — `has_opensees_orientation()` is the sole canonical
  orientation gate. The standalone
  `viewers/data/_h5_probe.has_opensees_orientation(path)` function is
  deleted after PR7-c; any remaining callers go through the reader.
- **INV-5** — `H5ModelReader.close()` is idempotent and may be called
  on a reader whose underlying file is already closed (an adapter
  backed by in-memory arrays has nothing to close). The context
  manager protocol is mandatory; the explicit `close()` exists for
  callers that hold a reader across an async boundary.
- **INV-6** — Adapters that synthesise a neutral view from a foreign
  format (LS-DYNA, Exodus, xDMF) declare `has_neutral_zone() ->
  True` once the synthesis is complete. A False return marks the
  reader as **bridge-only** (orientation present, neutral absent) —
  the viewer degrades to fem-only rendering, mirroring the MPCO
  asymmetry without a special case.

## Rejected alternatives

### A — Inherit from an `H5ModelReader` ABC

Force every adapter to subclass an abstract base. Rejected: ADR 0014
INV-1 forbids `viewers/` from importing apeGmsh.opensees beyond
`emitter.h5_reader`. An ABC would have to live somewhere
(`viewers/data/_protocol.py`), and existing `H5Model` would need to
inherit from it — coupling `apeGmsh.opensees.emitter.h5_reader` back
to `apeGmsh.viewers.data`. `typing.Protocol` solves the same problem
without the import edge. The two-PR cost of adding an ABC after the
fact (if it ever proves necessary) is acceptable.

### B — Codify the contract as Python type stubs only

Write `H5ModelReader` as a `.pyi` stub in `viewers/data/_protocol.pyi`
with no runtime presence. Rejected: the Protocol carries
`has_opensees_orientation` and `has_neutral_zone` methods that are
genuinely additive on existing `H5Model` — they're capability probes
the viewer wants to call, not just type hints. Stubs-only would mean
no place to land those methods.

### C — Keep the implicit contract and document it in `h5-schema.md`

The schema document already lists what a viewer-readable file looks
like at the byte layer. Rejected: that document describes the
**on-disk** contract; the H5ModelReader describes the **in-process
read** contract. A foreign-format adapter does not produce an
apeGmsh-shaped file on disk — it produces a Python object that
satisfies the reader Protocol. The two are complementary, not
overlapping.

### D — Skip Phase A; introduce the Protocol simultaneously with PR7-d's director refactor

Land the Protocol, the adapter wrap, and the director refactor in
one PR. Rejected: PR7-d touches `_director.py`, `_session.py`, and
the cuts auto-load gate at `results_viewer.py:1414-1431` — a non-
trivial refactor. Doing it under a freshly-introduced Protocol is
exactly the wrong order. Stage A first (zero behaviour change, AST
guard re-validates the import surface), then stage B-D incrementally.

### E — Make `H5ModelReader` an interface over the *results* file too

Widen the Protocol to also cover `Results._reader` (stage / timestep /
slab access). Rejected: that is a different contract with different
consumers (`Results.nodes / .elements` composites). Conflating them
makes both Protocols harder to satisfy for foreign-format adapters
that may carry model data without results data, or vice versa.
A separate `H5ResultsReader` Protocol is a candidate for a future
ADR; this one stays scoped to the model side.

## Consequences

**Positive:**

- The implicit "viewer reads model.h5 through h5_reader" contract
  becomes type-checkable. PRs that touch `h5_reader.H5Model` get
  Mypy / Pyright signal when they break the viewer's expected shape.
- `ResultsDirector` collapses two state slots (`_opensees_model:
  OpenSeesModel`, `_model_h5: Optional[Path]`) into one
  (`_reader: H5ModelReader`). The dual-bind ceremony at every
  results.viewer() call vanishes (closes D5).
- The duplicated orientation probe at `results_viewer.py:256-263` and
  `:1414-1420` collapses through `reader.has_opensees_orientation()`
  (closes D1 under the same refactor).
- A foreign-format adapter (LS-DYNA `d3plot`, Exodus, xDMF) needs
  only to implement the Protocol — no `viewers/` code touched. The
  research trajectory toward LS-DYNA result comparison is unblocked
  architecturally.
- Fixture-driven viewer tests can synthesise an in-memory adapter
  (`path=None`, `has_neutral_zone()=True`) without writing temp
  files. The "viewer renders any test fixture from `model.h5` alone"
  property from ADR 0014 generalises to "any reader-satisfying
  source."

**Negative:**

- One new module (`viewers/data/_protocol.py`) with a single
  Protocol class. ~80 LOC including docstrings. Maintenance cost is
  low precisely because Protocols are structural — adding a method
  to the Protocol is an opt-in event (existing adapters that don't
  implement it fail Mypy, which is the desired fail-loud).
- The `H5Model.path` property is new public surface on the bridge
  reader. Any external caller relying on the previous private
  `H5Model._path` attribute will continue to work, but the public
  attribute is now load-bearing.
- `_bind_model_h5(path)` lingers as a deprecated alias for one
  release cycle. Session-restore code that serialises a path string
  goes through this path; deleting it requires migrating session
  payloads to reader-construction descriptors, which is its own
  follow-up.
- Adding a foreign-format adapter (PR7-e or later) reopens the
  ADR 0014 AST guard question: does that adapter live inside
  `apeGmsh.viewers.data/` (allowed) or in its own package
  (`apeGmsh.foreign.d3plot/`)? The current guard at
  `tests/viewers/test_viewers_pure_h5_consumer.py:26-29` whitelists
  one allowed import — that whitelist needs widening as adapters
  land. The ADR governing that widening is out of scope here; it is
  a future event.

## Open questions (closed before adoption)

1. **Should `H5ModelReader` carry write methods too?** No. ADR 0011
   pins the asymmetry: writes go through the bridge (apeSees(fem))
   or the side-feeder (ModelData(fem)). The Protocol is read-only,
   matching the viewer's needs and matching how H5Model has been
   used since Phase 8.
2. **What about `_h5_probe.has_opensees_orientation(path)` callers
   outside `viewers/`?** A grep at draft time shows the function is
   used only inside `viewers/`. After PR7-c, the function is deleted
   outright (not deprecated) — the audit confirms a single consumer
   group.
3. **Does `cuts/FemToOpsTagMap` need its own ADR?** The `from_reader`
   factory is additive (the `from_h5` shim preserves the existing
   surface). No ADR required for the factory itself; it ships under
   this ADR's consequences. A future ADR is owed only if the
   `FemToOpsTagMap` contract itself widens (e.g. inverse-by-type-token
   lookup).

## References

- [ADR 0011](0011-h5-as-fourth-emit-target.md) — HDF5 as a fourth
  emit target. This ADR's read-side counterpart codifies the
  consumer contract that ADR 0011's writers feed.
- [ADR 0014](0014-viewer-is-pure-h5-consumer.md) — viewers as a
  pure-h5 consumer. This ADR formalises the import surface ADR 0014
  draws structurally.
- [ADR 0018](0018-modeldata-vanilla-opensees-enrichment.md) —
  `ModelData` orientation enrichment. The `has_opensees_orientation()`
  Protocol method is the runtime probe for whether ModelData (or the
  bridge) ran against the file.
- [ADR 0019](0019-opensees-model-read-side-broker.md) —
  `OpenSeesModel` read-side broker. The Protocol is the
  *viewer-side* read contract; `OpenSeesModel` is the *broker-side*
  in-process read contract. They are different layers and intentionally
  do not share an interface (one is duck-typed by ADR 0014 INV-2;
  the other is structural by this ADR).
- [ADR 0020](0020-results-carries-opensees-model.md) — Results
  carries OpenSeesModel via Composed file. The `Results._spawn_viewer_subprocess`
  contract preserved by `model_path` (post the PR1 fix in the same
  branch as this draft) is orthogonal to the Protocol — file paths
  cross the subprocess boundary; readers do not.
- [ADR 0023](0023-per-zone-schema-versioning.md) — per-zone schema
  versioning. The Protocol's `schema_version` property surfaces the
  bridge zone's version; adapters for foreign formats may return
  their own format-version string (the consumer is expected to know
  the format's versioning policy).
- [ADR 0024](0024-emitter-protocol-widen-region.md) and
  [ADR 0025](0025-emitter-protocol-widen-eigen.md) — Emitter
  Protocol widening events. Set the precedent that introducing or
  amending a Protocol is "an architecture event" deserving an ADR.
- `tests/viewers/test_viewers_pure_h5_consumer.py` — the AST guard
  this ADR's adoption preserves and (in PR7-e) widens.
- Visualization layer assessment (May 2026 head-engineer review,
  branch `guppi/suspicious-babbage-5b8b0a`, PR #282) — the four-agent
  + red/blue critique that ranked D1 and D5 as the motivating debts.
