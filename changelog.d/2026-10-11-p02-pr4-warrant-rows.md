### ADDED — warrant rows, and a warrant test that skips fails its live lane (ADR 0121 D3, #1687)

`tests/opensees/warrant_rows.py` holds the warrant rows of ADR 0121 D3: each
row names a member and the two tests that decide its warrant, one run in
`live-stock` and one in `live-fork`. The first two rows are the exemplars:
`TenNodeTetrahedron` is *refused* on stock (a `pytest.raises` at emit time
whose error names the fork; roster D1) and meets `K = E·A/H` on the fork;
`FourNodeTetrahedron` is *honest*, meeting the same closed form on both
engines. Both live lanes now set `APEGMSH_WARRANT_LANE`, and the root
`tests/conftest.py` turns a row test that skips, xfails or goes unselected
in its own lane into a failure. The exception is ruling 3: a `ladruno_mkl`
skip in `live-fork` counts for the lane and never for the member, whose side
the terminal summary reports as not green.
`tests/opensees/unit/test_warrant_rows.py` locks the rows (node ids, lane
markers, a refused row's `pytest.raises` naming the fork, roster
cross-references) and invariant 9: a fork member (fork-only element or
integrator, or a material whose docstring carries the fork command header
or a 33xxx tag) added after the freeze needs a refused row. The 42 fork
members at the freeze point are grandfathered as Inventory (ruling 5, X4).
