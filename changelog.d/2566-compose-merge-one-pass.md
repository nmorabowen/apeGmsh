### CHANGED — composing a module with many constraints is linear: the merge routes its records in one pass (#2566)

`Assembly.bridge` merged each instance's constraint, load and SP records
with one `FEMData.with_constraint` / `with_load` call per record, and each
call copied the whole record list. A module with 50,001 `equalDOF` records
spent about 9 s of an 11 s bridge there. The merge now routes every record
to its bucket in one pass and swaps the record sets in once: the same
records, in the same order, and the same `TypeError` for a record type it
cannot route. The four 50k-record cases of `tests/test_phase_3f_1.py` drop
from about 11 s to 2.6 s each.

`H5Model.commands` and the replay's per-recorder emit-order lookup also
join their rows to emit indices through one map per method,
`emit_index_map`, instead of scanning the whole program for each row.
