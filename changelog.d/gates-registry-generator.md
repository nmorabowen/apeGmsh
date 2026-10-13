### Gate registry generated from `check_quirks.py`

`scripts/gates_registry.py` derives the quirk rows of `internal_docs/program/gates.md` from the rule table in `scripts/check_quirks.py` (`--check` fails on drift, and a pytest runs it in the `suite` lane). The registry now lists all 19 rules, including `qt-worker-gc`, `emitter-sniff`, `vtk-import` and `render-tech-import`.
