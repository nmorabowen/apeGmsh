### ADDED — CI lane `subprocess-tests` runs the `subprocess`-marked tests (K1-3f)

`tests/opensees/subprocess/test_tag_determinism.py` (the K1-3 tag-determinism
pin) was marked `subprocess`, which both `suite` and `live-stock` deselect, so
it ran only locally. A new `subprocess-tests` job in `tests.yml` runs
`pytest -m "subprocess and not live and not qt and not bench"` with `-rs`.
Tests that need the standalone OPENSEES binary skip there; the determinism pin
needs only Python and runs. Not a required check.
