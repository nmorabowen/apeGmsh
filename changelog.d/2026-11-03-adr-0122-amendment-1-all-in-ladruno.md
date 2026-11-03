### CHANGED — ADR 0122 Amendment 1: apeGmsh reads only `.ladruno`; ADR 0115 D3 follows (#2755)

ADR 0122 gains Amendment 1 (Proposed), which records the owner's rulings on
#2655 and on #2079 (Q1–Q5). apeGmshViewer, the v2 stills and the in-memory
render read `.ladruno` only. `convert_mpco` stays as the only consumer of an
`.mpco`, while `Results.from_mpco` and the slab MPCO readers are to be
deleted once the shared code has moved out. Footfall moves to
`<stem>.user.h5`. The amendment also re-slices the remaining plan. ADR 0115
Amendment A1 changes `ops.results`: an MPCO recorder is refused with a
message naming `convert_mpco`, and a Ladruno recorder reads through
`from_ladruno`. Only the ADR text changes here; the code changes in later
slices.
