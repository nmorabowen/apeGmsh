### ADDED — tag-law locks: emit determinism and the replay-minting AST lock (K1-3, #1361)

Today's tag behaviour is now pinned, so the build-time tag plan (#1445) and any
schema bump cannot drift it silently. The `(kind, tag)` stream that
`BuiltModel.emit` drives is the same in a fresh interpreter under two hash
seeds, on the flat, partitioned, staged, staged-partitioned and split paths
(`tests/opensees/subprocess/test_tag_determinism.py`). The `(verb, tag)`
multiset is the same across the Recording, Tcl, Py and H5 emits of one
`BuiltModel` (`tests/opensees/contract/test_tag_streams.py`). An AST lock
(`tests/opensees/contract/test_tag_law_lock.py`) keeps `emitter/`,
`_internal/compose.py` and `opensees_model.py` from minting a tag: no
`allocate*` call, no call to a recorder's `materialize`, no reference to one of `build.py`'s 25 tag-minting helpers (a list
derived from `build.py` and locked), no `TagAllocator._counters` access, and no
`max(...) + 1`. Compose's replay minting (the step-8b reinforce ties and the
initial-stress and absorbing parameter tags) is waived by name in the
shrink-only `tag_law_ledger.txt`, and each waiver is pinned by a test showing
the replay reproduces the forward tags. Behaviour is unchanged.
