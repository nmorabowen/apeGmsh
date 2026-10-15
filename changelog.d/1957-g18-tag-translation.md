### TESTS — FEM-id to OpenSees-tag translation proven on `.out`, MPCO and `.ladruno` (program slice X/G18, #1957)

One solid + frame model, built through the bridge with sequential tags, now proves the element-id
translation on all three file routes and for every element kind: nodal forces (an elastic strut,
`globalForce`), Gauss (a hex solid, `stresses`) and line stations (a force-based beam, `section.force`).
The model is a gmsh cube with `Base`/`Top` surface groups plus two beam lines, so no family's FEM ids are
its OpenSees tags. For each route and kind the read-back element ids are the group's FEM ids, every row
equals `eleResponse` on the paired tag at `rtol=1e-10`, and an `ids=` filter returns exactly those
elements. The `.out` cases run on the live-stock lane; the MPCO and `.ladruno` cases run on the fork lane
(`tests/opensees/integration_ladruno/test_tag_translation_live.py`) and also cover the two MPCO proofs
the G9 audit found with no fork-lane twin: element nodal forces, checked against `nodeReaction` too, and
`section.force` line stations. Mutation tests read the same files through a pairing that is dropped or
rotated and show the id and value checks fail; on `.out`, the dropped pairing is refused at emit.

On MPCO and `.ladruno`, an `ids=` filter on a FEM id outside the pairing that equals an emitted OpenSees
tag returns that element's rows under the wrong id. The six cases are strict xfails citing #2066.
