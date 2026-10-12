### ADDED — apeGmshViewer: results on the model, a time scrubber, and stage activation (V4b, #1728)

apeGmshViewer opens a native `results.h5` the way it opens a `model.h5`
(file association, a drop, the command line, or beside the model) and reads
it lazily in the main process (ADR 0112 D5): the store holds the stages, the
time vectors and the field names, and one step is fetched at a time, the
newest request winning. The results are drawn on the open model only when
their `/meta/lineage/model_hash` equals the model's (else `fem_hash`); a
results file of another model, or one with no model open, is not drawn and
the banner says why. A paired file adds a time scrubber to the phase bar
(results stage, step slider, time, play/pause, step -1/+1). On a results
stage of a staged model, the elements that stage does not have are hidden,
as the model's `/sequence` element rows say (`activate element`,
`remove element`); the results stage pairs with a `/sequence` stage by name,
and no match, two matches or an unnamed stage draws every element under a
banner. The activation equals Python's `build_from_model` on the committed
staged fixture. Contours and the deformed shape come later.
`apeGmshViewer/MEASUREMENTS.md` records when to reopen D5: a real results
file that scrubs below 30 steps/s or peaks above 2 GB.
