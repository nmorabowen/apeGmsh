### ADDED — `.ladruno` readers name and kind their stages after the program, as MPCO's do (#2100)

`LadrunoReader` and `LadrunoMultiPartitionReader` now have
`attach_stage_names(names, kinds)`, with the signature, pairing and
refusals of the MPCO pair. The names and kinds pair positionally onto the
non-mode stages. Mode stages keep their own names and kind, and stage ids
do not change. Each `MODEL_STAGE[<k>]` name moves to the stage's aliases.
A count mismatch, or a kind other than `static` or `transient`, raises
`ValueError` and renames nothing. The partitioned reader forwards the call
to every part file.

A `.ladruno` stage stores a `KIND` attribute and an `.mpco` stage does not.
`KIND` is the recorder's `-kind` option: it defaults to `static`, and it
reads `eigen` beside modes. It does not record how the stage was analysed,
so the program's kind replaces it. The stage's `analysis` record in
`model.h5` is the source, as it is for MPCO.

This lets `Results.from_ladruno(..., model_h5=)` name its stages the way
`from_mpco` does (ADR 0122 slice 3b).
