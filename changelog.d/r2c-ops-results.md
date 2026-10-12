### ADDED — `ops.results(path)` reads a deck run back as `Results`; `NodeSlab.item()` (ADR 0115 D3, program slice R2-c, #1704)

After `ops.py(path, run=True)` or `ops.tcl(path, run=True)`, the same
bridge returns the run's history as the existing `Results`, with no
`np.loadtxt`: `ops.results(path).nodes.get(label="tip",
component="displacement_y").item()`. `path` is required and must be the
deck that bridge ran. What the run recorded picks the reader: one
`recorder.MPCO` is read by `Results.from_mpco`; `recorder.declare(nodes=...)`
and typed `recorder.Node(..., time_format="dt")` records are read by
`Results.from_recorders`, through one recorder spec synthesised from them. A
typed record's DOFs take their component names from the bridge's own DOF
layout, so DOF 3 of a planar `ndf=3` frame node reads as `rotation_z`.

`ops.results` raises `BridgeError`, naming the cause, before it writes any
file: no deck run on this bridge; a live `analyze` since (read that with
`.read`); another `path`; a run that did not finish; declarations changed
after the run; no recorder; or a record it cannot read. It refuses rather
than return a `Results` that silently lacks a record: a typed
`recorder.Element`, a `declare` element category or `raw_*` token, a
`Ladruno` or `Monitor` recorder, a typed record without a time column, a
DOF whose component the node's layout does not establish (a `declare`
`rotation_z` on an `ndf=3` node, a rotational slot no beam-column or shell
touches), the text records of a staged or partitioned run, an MPCO recorder
beside text ones, and a missing, empty or malformed output file. A read
writes beside the deck, under `<path>.results/`: the model archive the
reader binds, the text records in the reader's layout, and its cache.

`NodeSlab.item()` returns a slab's one value as a `float` and raises
`ValueError` unless the slab holds exactly one value (one node at one step).
It replaces `.values[-1, 0]`.
