### TESTS — the kitchen-sink staged fixture's deck now solves, and its known-failure row is gone (#2376)

`_build_kitchen_sink_bridge` (aliased `_real_kitchen_sink_bridge`) removed
the Rock quad and the Base x-fix in one stage, which left the orphaned Base
x-DOFs as exact zero pivots and the Fill quad with two rigid-body modes, so
its deck aborted on stock and on the fork. The stage now re-holds the
released Base x-DOF (`remove_sp` then `support` on the same target) and holds
FillTop in both DOFs. Every stage branch the fixture covers is still in it.
The R1b live net's `_KNOWN_STOCK_FAILURES` drops the row; the table only
shrinks. Test-only; no `src/` change.
