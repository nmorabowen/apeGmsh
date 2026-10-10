### FIXED — openseespy emitters kept only 3 of a tet4 embedded tie's 4 corners (#1621, slice B4-a #1626)

`ASDEmbeddedNodeElement`'s parser reads the optional 4th retained node with
`OPS_GetString` + `std::stoi` inside a `catch(...)`. Under openseespy an `int`
there is not a string, so the `.py` deck and the live route, which passed every
retained node as an int, silently tied a tet4 host to three of its four corners
(an embedded node at barycentric (0.1, 0.2, 0.3, 0.4) read `u = 0` instead of
`0.4 * u_apex`). `PyEmitter.embeddedNode` and `LiveOpsEmitter.embeddedNode` now
emit the 4th node as its decimal string through one shared helper
(`_embedded_retained_args`); every other node count passes through as before, and
3-node (triangle) calls and the Tcl deck are byte-for-byte unchanged. Proven by
`tests/opensees/live/test_embedded_tet4_live.py` (stock and fork) and
`tests/opensees/unit/test_embedded_node_4th_retained.py`.
