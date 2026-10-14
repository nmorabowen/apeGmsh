### ADDED — live read-back of four element families off DomainCapture (#1949, #1767 G7)

Four element families now have value tests on the recorder routes: truss axial
force, LadrunoBrick20 and LadrunoLST Gauss stress, and mixed-ndf
shell-on-solid nodes. Each test checks against the live domain (`eleResponse`,
`nodeDisp`):

- The fork lane runs
  `tests/opensees/integration_ladruno/test_element_families_ladruno_live.py`,
  which covers `.ladruno` and MPCO at `rtol=1e-10`.
- The stock lane runs
  `tests/opensees/integration/test_element_families_fallback_live.py`, which
  reads the truss and shell-on-solid cases from `.out` at the format's six
  digits.

Each family and route is pinned. Some routes silently read back an empty
slab, empty GP coordinates or a phantom rotation of 0.0. Those are strict
xfails citing #1969 (the `.ladruno` reader), #1970 (the MPCO reader) and
#1971 (3-dof nodes reading rotation 0.0). The MPCO truss read refuses loudly,
and a `pytest.raises` test pins that gap.
