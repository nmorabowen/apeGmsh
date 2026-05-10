# ADR 0011 — HDF5 as a fourth emit target

**Status:** Accepted

## Context

The bridge holds model-definition data that exists nowhere else in
the apeGmsh stack:

| Data | In FEMData? | In MPCO results? | In bridge? |
|---|---|---|---|
| Material constitutive parameters | ❌ | ❌ | ✅ |
| Section fiber/patch geometry | ❌ | ❌ | ✅ |
| GeomTransf vecxz (per element when csys is used) | ❌ | ❌ | ✅ |
| TimeSeries values | ❌ | ❌ | ✅ |
| Pattern membership (load → series) | ❌ | ❌ | ✅ |
| Element → section / material / transf refs | ❌ | ❌ | ✅ |
| Recorder topology (what's measured where) | ❌ | ❌ | ✅ |
| Analysis settings | ❌ | ❌ | ✅ |

This data lives in memory as long as the bridge instance lives. Today
it is lost when the bridge is gone. The apeGmsh viewer would benefit
from access to all of it — to render fiber sections on a clicked
beam, to plot the time series of a pattern, to overlay vecxz
glyphs, to inspect recorder coverage.

## Decision

Add an `H5Emitter` to the emitter package (`emitter/h5.py`). It
implements the same `Emitter` Protocol as `LiveOpsEmitter`,
`TclEmitter`, and `PyEmitter` — but instead of accumulating Tcl
strings or driving openseespy, it writes a structured HDF5 file
following the schema in [h5-schema.md](../h5-schema.md).

The bridge exposes:

```python
ops.h5(path)                          # write H5 only
```

This is **a fourth emit target** alongside Tcl, py, and live. It
does not imply analysis was run — the H5 is a *model-definition*
archive.

The viewer integration is owned by the viewer team and specified in
[viewer-integration.md](../viewer-integration.md).

## Alternatives considered

1. **Pickle the bridge instance.** Rejected — Python-version
   coupled, not portable across tools, not introspectable
   without Python, and ties the viewer to the exact bridge class
   shape.
2. **Extend FEMData to hold this data.** Rejected — FEM is
   geometry-only by design; conflating model-definition concerns
   into FEM violates separation. Also FEM is consumed by every
   other apeGmsh module; adding OpenSees specifics there would
   leak.
3. **One sidecar file per concept** (`model.materials.json`,
   `model.sections.json`, …). Rejected — fragmented, no atomic
   read, no cross-reference checking, more file-discovery
   complexity for the viewer.
4. **Use STKO's HDF5 result format.** Rejected — STKO/MPCO is a
   results format, not a definition format. Different schema,
   different lifecycle.

## Consequences

**Positive:**

- Validates the Emitter abstraction empirically — adding a new
  target IS one new file, no primitive changes (P8 satisfied).
- Decouples viewer enrichment from the bridge instance lifecycle.
  Hand someone `model.h5` and they have the full definition.
- Schema-versioned, navigable as a graph (paths as cross-refs),
  HDF5-native types throughout.
- Archival: model-definition snapshots can be diffed, shared,
  reproduced.
- Same producer surface: `ops.h5(path)` reads identically to
  `ops.tcl(path)` / `ops.py(path)`.

**Negative:**

- New file format to maintain and version.
- Hard dependency on `h5py` for any user calling `ops.h5()`.
  Mitigated by lazy import — primitives don't need h5py.
- Schema changes require coordination with the viewer team
  (versioning policy in
  [viewer-integration.md](../viewer-integration.md)).
- Risk that H5 becomes the primary contract instead of the
  in-memory bridge, slowing future API changes. Mitigated by
  treating the H5 strictly as an emit output, never an input —
  the bridge does not read its own H5 back.

## Reference

- Schema: [h5-schema.md](../h5-schema.md)
- Viewer contract: [viewer-integration.md](../viewer-integration.md)
- Emitter abstraction: [emitter.md](../emitter.md)
- [charter.md P2, P8](../charter.md)
- ADR 0008 — three emit targets via Emitter Protocol
