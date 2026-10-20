### INTERNAL - the deck run-record plumbing moves out of `apeSees.py()` / `apeSees.tcl()` (#2289)

The emit tail (profiler records, `analyze`, run record) now lives in `emit_deck_tail` in `_internal/artifact_write.py`, shared by both exporters. No behaviour change: the decks are byte-identical.
