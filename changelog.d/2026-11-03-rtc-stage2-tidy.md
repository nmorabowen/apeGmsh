### FIXED — stock MPCO route tidy after ADR 0131 stage 2 (program slice X RTc, #2805)

`spec.emit_default(..., overwrite=True)` on a stock MPCO build now removes the earlier run's
`.ladruno` at `__enter__`, as the fork route does, so a read that then refuses leaves no stale
`.ladruno` beside the new `.mpco`. A stage-kind refusal from `ops.results(path)` names each remedy
once and quotes the converter's reason rather than `read_deck_mpco`'s wording. The "close the open
Results" advice now appears only for a Windows sharing violation (`winerror` 32); any other
`PermissionError` during the replace is quoted as it is. The `read_deck_mpco` and `StageKindRefused`
docstrings and the skill (`references/opensees-bridge.md`, `references/workflows.md`) now say that
a stock MPCO run converts or refuses by name (ADR 0131 D5), and show footfall's
`to_results(fem, path, model_h5=)`, which writes `<stem>.user.h5`.
