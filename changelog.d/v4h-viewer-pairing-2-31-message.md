### CHANGED — apeGmshViewer: the results pairing names the opensees 2.31.0 model_hash change (V4h, #1856)

At opensees 2.31.0 every bridge archive's `model_hash` changes once,
because declaration ids entered the hashed `/opensees/program` column
(#1850). A results file and a model whose hashes differ are still refused,
but when exactly one side's hash includes declaration ids the banner now
names that cause and the remedy (re-run the analysis against the re-saved
model, or open the results with the model they were run on). Each side's
era is its `/opensees/decls@program_decl` completeness marker, as in the
Python half (#1861), never the version stamp: a `to_h5` rewrite of an
older file is stamped 2.31 but keeps its old hash. A missing marker or
`decls` group reads as "absent". With no `/opensees` zone, or an unknown
marker, on either side, and for every other mismatch, the message reads as
before. The 2.31.0 boundary named in the message is one constant,
`PROGRAM_DECL_FROM` in `apeGmshViewer/src/state/results.ts`.
