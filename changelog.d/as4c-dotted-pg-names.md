### FIXED — Assembly rehydrates element specs on dotted physical groups (AS4-c, #1549)

An instance whose source holds a physical group named with a `.` or `/`
(for example `deck.slab`) failed `Assembly.bridge()`: the rehydrator named
the element spec's group `A.deck.slab`, while the merge engine had written
`A/deck.slab` by the ADR 0038 alternation. The rehydrator now asks the merge
engine's own rule (`mesh._compose._prefix_namespaced_name`), so both sides
name the same group. The same fix lets an assembly archive bridge as an
instance of another assembly (`X.A/deck.slab`, `X/A.Right`). Material and
other bridge names, the carried rebar material included, keep the flat
`{instance}.{name}` rule on both sides.
