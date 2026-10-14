### ADDED — live read-back of four element families off DomainCapture (#1949, #1767 G7)

Truss axial force, LadrunoBrick20 and LadrunoLST Gauss stress, and mixed-ndf
shell-on-solid nodes now have value tests on the recorder routes, each against
the live domain (`eleResponse`, `nodeDisp`). On the fork lane,
`tests/opensees/integration_ladruno/test_element_families_ladruno_live.py`
reads them from `.ladruno` (all four) and MPCO (shell-on-solid) at
`rtol=1e-10`. On the stock lane,
`tests/opensees/integration/test_element_families_fallback_live.py` reads the
truss and shell-on-solid cases from `.out` at the format's six digits. Routes
that cannot carry a family yet are listed on #1767: MPCO for the truss and the
two fork solids, and the truss as a Gauss record on `.ladruno`.
