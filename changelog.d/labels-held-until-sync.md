### FIXED — `add_*(..., sync=False, label=...)` keeps its label: it is held and created at the next synchronising call

A label asked for on entities created with `sync=False` was a physical
group on entities gmsh did not know yet, so gmsh dropped it without a
word and a later `g.labels.entities(...)` or `fragment("name", ...)`
raised `KeyError`. `g.labels.add` now holds such a label on the session
and creates it, after one `synchronize`, at the next of:

- `g.model.sync()`;
- any `g.labels` query;
- a boolean or transform;
- `generate`, `get_fem_data` and the session or `Part` `end()`.

A held entity that is gone by then is reported with a `UserWarning`.
`g.labels.add` returns `0` for a held label.

The batched pattern (`sync=False` with `label=` on every `add_*`, then
one `g.model.sync()`) now gives the same labels as `sync=True`, without
one OCC `synchronize` per entity. Geometry plus labels for a 10-storey,
2,080-entity frame drop from 46 s to 0.8 s.
