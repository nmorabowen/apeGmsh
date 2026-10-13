### ADDED — R1b round-trip oracle with a shrink-only waiver ledger (K2a, #1794)

`tests/opensees/golden/test_roundtrip_oracle.py` checks that
`ops.tcl(flat=True)` and `OpenSeesModel.from_h5(ops.h5()).build("tcl")` give
the same bytes. It runs over 54 real-FEMData `apeSees` builders already in
`tests/opensees/h5`. 18 of them match. The other 36 are rows in
`roundtrip_oracle_ledger.txt`, and each row asserts its exact diff:

- 12 differ only by integral floats (class A).
- 10 differ only by declaration order (class B).
- 6 are semantic replay differences (class C, #1798).
- 8 are FEMStub models that cannot be replayed (class S, #1797).

`N_WAIVED_ORACLE = 36` may only go down, and must reach 0 by 2026-12-11.
The FEMStub golden emit corpus is bridge-only, so it is not part of this
oracle (#1797).
