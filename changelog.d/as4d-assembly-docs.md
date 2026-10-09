### DOCS — Assemble saved models how-to; `bridge()` re-checks that every source is unranked (ADR 0117 P4, AS4-d, #1566)

A new how-to, *Assemble saved models* (`docs/how-to/assemble-saved-models.md`),
documents Assembly v2 end to end with one runnable example: two instances of
one saved `model.h5`, one rotated and translated, a `tie`, a reference
`node` with a `rigid_link` cap, `bridge()`, the serial `tcl()` deck,
`h5()` and `Assembly.from_h5`, a solve that matches the closed form
`P·2H/(E·A)`, and `partition_rank` for OpenSeesMP. It states the carry
rule, what refuses, and that the v1 compose path is pending removal. The
apegmsh skill gains an Assembly v2 section and triggers, ADR 0117 §D6 gains
a dated note on the AS4-b rank layout, and the `Assembly.bridge()`
docstring now lists what is carried and refused.

`Assembly.bridge()` now repeats `instance()`'s unranked-source check before
the merge, so a source file replaced by a ranked assembly archive after
`instance()` raises `AssemblyError` instead of bridging with ranks the
assembly never declared.
