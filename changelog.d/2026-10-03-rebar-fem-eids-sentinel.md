### CHANGED — `fem_eids = -1` documented for bridge-minted rows, with the rebar join key (program slice #1290)

`architecture/h5-schema.md` now has a "Bridge-minted rows" section under
`/opensees/element_meta`. It covers interface `zeroLength` springs, node-pair
elements, couplings, and the `CorotTruss` bars from
`g.rebar.place(emit_elements=True)`, which carry `fem_eids = -1` on purpose
(ADR 0049, ADR 0093 S10). The section says why the sentinel is load-bearing
and how a reader joins a bar to its trusses without `fem_eids`: the `-1`
rows form one contiguous block (a user `CorotTruss` PG shares the group with
real ids), matched by node pair in `/rebar_elements` order, with the
material taken from `args[:, 1]` through `/opensees/names`. The doc also
gets a `/rebar_elements` payload field table. `tests/rebar/test_rebar_h5_join.py`
pins the join with raw h5py reads, including a mixed user-truss model. The
written file does not change.
