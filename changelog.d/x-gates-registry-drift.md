### FIXED — `gates_registry.py --check` now catches three residual drifts (#1892)

`--check` exits non-zero when the gates.md table header is reworded, when a
non-quirk (lock, ratchet, label) row is duplicated, and when a
`ratchet scripts/quirks_*.txt` row outlives its deleted file. Each drift has a
test that fails without its fix. Regenerating drops the stale ratchet row and
writes the canonical header.
