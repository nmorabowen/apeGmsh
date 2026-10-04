### CHANGED — compose warns when it drops a source stream (program slice B2-2, D9)

`g.compose` rebuilds the host's `ElementComposite` from the rewritten bundle,
and the bundle carries every neutral-zone stream but the source module's
`elements.rebar_elements` (the cage's auto-emitted structural rebar from
`g.rebar.place(emit_elements=True)`), which vanished without a word. The
rewriter now emits one `apeGmsh.mesh._compose.ComposeDroppedStreamWarning`
per non-empty uncarried stream, naming the stream and its record count; a
plain compose and a host-side rebar stream (which the merge carries) stay
silent. The uncarried streams are listed in `_UNCARRIED_ELEMENT_STREAMS`
beside `_emit_filter_warnings`; carrying `rebar_elements` (tag offset on
`connectivity`, PG prefix, material-name decision) remains a follow-on.
`tests/mesh/test_compose_dropped_streams.py` pins the warn-iff contract.
