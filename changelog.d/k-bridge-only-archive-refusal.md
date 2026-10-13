### CHANGED — OpenSeesModel.from_h5 refuses a bridge-only archive by name (#1797)

`OpenSeesModel.from_h5` on an archive with `/meta` but no neutral zone (no
`nodes` group, e.g. one written from an `apeSees` built on `FEMStub`) now
raises `BridgeOnlyArchiveError` before any read. The message says the
archive carries no FEM, so it cannot be replayed, and lists the zones that
are present. It used to fail deep in the neutral-zone read with h5py's raw
`KeyError: "... object 'nodes' doesn't exist"`. The new error subclasses
`MalformedH5Error` (a `ValueError`), not `KeyError`; code that caught the
raw `KeyError` should catch `BridgeOnlyArchiveError` instead.
