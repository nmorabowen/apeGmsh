### ADDED — apeGmshViewer opens a study of variants, switches them and shows their difference (ADR 0132 D8, D9; program slice VU10, #2694)

apeGmshViewer reads `<study>.variants.json` (ADR 0132 D8; written by `apeGmsh.results.user.write_variants`
or by hand) from a path the user gave: a double-click, a
drop, argv, or a view spec's new `files.variants`. Each row that passes opens as the window set on
**Switch**; a refused row is listed with its code. **Compare** opens another row read-only in a compare
slot with its own results host, which never becomes the window set. **Show difference** colours by the
window variant minus the compare variant, one nodal component by node id, at the stage and step shown,
with "Difference: window minus compare (read-only)" on the legend. It is refused with the reason when
the two results do not state one mesh (`fem_hash`), on a magnitude, an element field, a missing stage,
step time or component, and on a User Combination: a combination across variants is refused. It is
never a deformed shape. The page names a variant by row index and label, never a path; main resolves
the row (owner ruling option A, #2753). A view spec's shot takes `compare: "<label>"` for a difference
still. The skill's `references/variants.md` documents the manifest and the viewer's study.
