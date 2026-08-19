# Libraries — related ape* stack

Inventory and roles of libraries that underpin APE Studio habitats. Only
apeGmsh, apeCAD, and apeSketch ship as required infra here — see
`catalog.md` for why, and for agent guidance on any further personal
libraries a habitat's author may reference.

| File | Purpose |
|------|---------|
| `catalog.md` | Human door — layers, roles, availability (edit by hand) |
| `catalog.json` | Machine twin (on-disk status under `<github>/`, empty until harvested) |
| `harvest.py` | Refresh on-disk facts for the `LIBRARIES` list |

```text
python APE/libraries/harvest.py
```

Libraries are not auto-imported; set `PYTHONPATH` / install when needed.
