### CHANGED — partitioned emit resolves each rank's owned nodes in one lookup (#1670)

The owned-node pass of partitioned emit did one scalar `searchsorted` per
node to find its coordinate row. It now sorts each rank's node ids once,
resolves them with one vectorised `SortedIntToInt.get_many`, and reads the
coordinates in one indexed gather. Emitted decks are byte-identical; on
the emit-cost gate cells the per-node `_find` calls (60k tet, 100k hex)
drop to 4 `get_many` calls.
