### FIXED — a FEMData-archive run no longer refuses coincident nodes that its live elements pin (#2342, gap b)

On a stock build, `spec.emit_default(...)` over a FEMData archive
(`fem.to_h5`) checks that each recorded node at a shared position (an
`equalDOF` pair, a tie's corner nodes) is pinned by its elements, so a deck
that swapped two coincident tags cannot record one node's motion under the
other (#2251). The check asked every FEM element touching such a node to be
live, including the point, edge and face elements that `get_fem_data()`
holds beside the beams or solids a hand-built deck builds. The curriculum
notebooks 12 (`12_interface_tie`) and 16 (`16_tied_contact_nonmatching`)
were refused with "FEM element 3 has nodes [3], no live element has that
tag".

The check now passes over a FEM element the deck did not build. Each built
element that touches the node must still have its FEM connectivity, in
order, and at least one must exist. A node that only unbuilt elements touch
refuses by name, and a coincident-tag swap in a built element still
refuses, naming the node and the element. Both notebooks' decks now read
back equal to the live `nodeDisp` at rtol 1e-12 on pip openseespy.
