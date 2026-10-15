### TESTS — MPCO and .ladruno nodal and Gauss proofs check values, not signs (#1955, G16)

The MPCO nodal proofs asserted only that the apex moved down, and the Gauss
proof only that `stress_zz` was negative and above 1e3. Each now compares
every recorded node and component of `displacement` and `reaction_force`
with `nodeDisp` and `nodeReaction`, and every GP component of `stresses`
with `eleResponse(e, "stresses")`, at `rtol=1e-10` over two load steps.
The oracle comes from the same run (in process, or printed by the Tcl deck).
`tests/opensees/integration_ladruno/test_nodal_gauss_oracle_live.py` runs
the same checks on the fork lane, on MPCO and `.ladruno`, for a
`FourNodeTetrahedron` and an 8-GP `stdBrick`. It also perturbs one recorded
value in a copy of each file and shows the oracle then fails.
