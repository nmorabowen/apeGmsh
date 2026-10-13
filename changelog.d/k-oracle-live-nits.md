### Tests: R1b oracle live-net nits (#1836)

- The live round-trip net gates on `importlib.util.find_spec("openseespy")`
  instead of `pytest.importorskip`, so it never imports openseespy into the
  shared pytest process; a new test runs one of the module's tests, autouse
  gate included, in a fresh `pytest` process and asserts `sys.modules` holds
  no openseespy afterwards.
- A fork-only class B row on stock openseespy now skips by name, with its
  solve stamp's `requires`, after asserting only that the bridge deck did not
  run, so `-rs` shows it instead of a silent pass.
- The static declare-before-use check documents its first-declaration
  semantics and why they give the nearest-preceding verdict, with a
  synthetic redeclaration case and the real corpus row that re-declares tags.
