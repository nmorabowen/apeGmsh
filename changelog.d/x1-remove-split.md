### REMOVED: the `split=` emit mode (ADR 0043, withdrawn)

`apeSees.tcl(path, split=True)`, `apeSees.py(path, split=True)` and `BuiltModel.emit(emitter, split=True)` are gone, together with the `_emit_split` path and the `parts/<module>` fragment writers (about 700 lines). The mode was Proposed for 123 days and no example used it. The single-file deck and the `per_rank=True` fragment layout are unchanged. ADR 0043 is marked withdrawn.
