### FIXED — an element whose connectivity repeats a node is refused before any write (program slice B3-a, #1536)

The "duplicate node tags" refusal lived only in the dead `emit_element_spec`
that K1-3d S6 (#1546) deleted, so the live element fan-out wrote
`element Truss 3 3 3 ...` into the deck, and nothing upstream (FEMData,
`select()`, `apeSees.build()`) refused the cell. The tag plan now checks every
planned element row (`_internal/tag_plan.py::check_distinct_nodes`, run by
`ElementTagPlan` inside `plan_tags`), so the first emit raises `BridgeError`
naming the element class and PG, the planned OpenSees tag, the FEM element id
and the repeated node, before `ops.tcl` / `ops.py` / `ops.h5` write anything.
Pinned by `tests/opensees/test_duplicate_node_refusal.py`.
