### CHANGED — apeGmshViewer: the results pairing names the opensees 2.31.0 model_hash change (V4h, #1856)

At opensees 2.31.0 every bridge archive's `model_hash` changes once,
because declaration ids entered the hashed `/opensees/program` column
(#1850). A results file and a model whose opensees stamps fall on either
side of 2.31.0 are still refused when their `model_hash` differs, but the
banner now names that cause and the remedy (re-run the analysis against
the re-saved model, or open the results with the model they were run on).
The model side's stamp is `/meta@opensees_schema_version`; the results
side's is the embedded opensees zone's stamp, which the results reader now
carries to the pairing. Any other mismatch, and a mismatch where either
stamp is absent, reads as before. The boundary is one constant,
`PROGRAM_DECL_FROM` in `apeGmshViewer/src/state/results.ts`, named after
Python's `h5_reader.PROGRAM_DECL_FROM`.
