### ADDED — shrink-only LOC ratchet for the freeze list (program slice X4-b, #1807)

`tests/test_loc_ratchet.py` pins the non-blank Python line count of `viewers/`, `studio/`, `hpc/`,
`ground_motions/`, the from_recorders path (`results/transcoders/` plus the `Results.from_recorders`
span), the section GUI (`sections/_builder_gui.py`, `_inspector.py`, `_properties.py`) and the rebar
paths (`rebar/`, `core/RebarComposite.py`). A surface that grows fails; one that shrinks fails with the
lower number to adopt. The file joins `lock-tests` and `land_pr.py::BASELINE`, so editing it needs the
`human-gate` label.
