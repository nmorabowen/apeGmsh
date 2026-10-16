### ADDED — `.ladruno` readers name and kind their stages after the program, as MPCO's do (#2100)

`LadrunoReader` and `LadrunoMultiPartitionReader` now have
`attach_stage_names(names, kinds)`, with the signature, pairing and
refusals of the MPCO pair. The names and kinds pair positionally onto the
non-mode stages. Mode stages keep their own names and kind, and stage ids
do not change. Each `MODEL_STAGE[<k>]` name moves to the stage's aliases.
A count mismatch, or a kind other than `static` or `transient`, raises
`ValueError` and renames nothing.

A `.ladruno` stage records its `KIND`, and on the pinned fork that is the
kind the stage ran. When a recorded `static` or `transient` disagrees with
the program kind paired onto it, the pairing is misaligned. Following
#2079 Q7 ("otherwise keep the file's names and warn"), no stage is
renamed: the file keeps its names and kinds, and `StageKindMismatchWarning`
lists each conflict. `eigen` is not evidence, because the recorder writes
it on every stage of a recorder that asks for modes. The program kind
fills in where a stage records `eigen` or no `KIND`. The partitioned
reader decides once over all part files, so the parts are either all
renamed or none is.

This lets `Results.from_ladruno(..., model_h5=)` name its stages the way
`from_mpco` does (ADR 0122 slice 3b).
