# A study of variants — `<study>.variants.json` (ADR 0132 D8, D9)

Runs of one model that differ in one parameter form a study. Write
`<study>.variants.json` beside their sets; each `path` names a model
`<stem>.h5` relative to the manifest's folder, and must stay inside it:

```python
from apeGmsh.results.user import VariantDef, write_variants

write_variants("study.variants.json", [
    VariantDef("Uncracked", "plate_uncracked.h5", {"Ie/Ig": 1.0}),
    VariantDef("Cracked", "cracked/plate_cracked.h5", {"Ie/Ig": 0.35}),
])
```

Open the manifest in apeGmshViewer: its first row is the window set, and
the Variants panel lists every row (a refused row says why). **Switch**
opens a row as the window set. **Compare** opens a row read-only. **Show
difference** then colours by window minus compare: one nodal component,
at the stage and step shown, on one mesh (`fem_hash`), never as a deformed
shape. A combination across variants is refused. For a still, a spec names
`files.variants`, and a shot names `compare: "<label>"`.
