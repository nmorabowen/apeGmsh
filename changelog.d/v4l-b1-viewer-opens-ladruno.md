### CHANGED — apeGmshViewer opens `<stem>.ladruno` and pairs it by its INFO/PROVENANCE stamp (#2127)

A set now reads one results file. When `<stem>.ladruno` and
`<stem>.results.h5` both sit beside the model, the file you opened wins;
otherwise the `.ladruno` does (ADR 0122 D1). Before this change, opening the
model read the `.results.h5`. The banner now names the results file that was
not opened. A `<stem>.part-<N>.ladruno` opened directly is refused, and the
refusal says that partitioned sets arrive with ADR 0122 slice 8.

A `.ladruno` pairs with the model by the `model_hash` and `fem_hash` that
`stamp_provenance(path, model_h5)` writes. Lineage needs both hashes. A
stamp written in part used to pair by `fem_hash` alone; it now reads as
unpaired, as schema §7b.1 says. An unstamped file is not drawn either: the
banner gives the reason and names `stamp_provenance` as the remedy. A file
re-stamped while open is re-read, because its whole INFO/PROVENANCE is now
part of its identity, so a change the size and mtime miss is still caught.
`stage_names` names the stages only when it counts as many as the file's
MODEL_STAGE groups. It no longer shifts onto mode stages. Otherwise the file
keeps its own names and a warning says why.
