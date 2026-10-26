### ADDED — `geomTransf` joint offsets (`jnt_offset=`) and an ETABS insertion-point helper (opensees 2.37.0)

`ops.geomTransf.Linear` / `PDelta` / `Corotational` take
`jnt_offset=((dXi, dYi, dZi), (dXj, dYj, dZj))`, the global vectors from each
node to its element end, emitted as `-jntOffset` (2-D: 2-tuples,
`-jntOffset dXi dYi dXj dYj`). A 3-D `Corotational` refuses one: OpenSees'
`CorotCrdTransf3d` (stock and Ladruno) zeroes the offsets with a warning, so
the element would solve without them.

`jnt_offset=` also takes an `InsertionPoint` rule
(`apeGmsh.opensees.joint_offset`): an ETABS cardinal point 1-11 on the
section's depth/width or bounding box, `Mirror2`/`Mirror3`, joint offsets in
global or local components, and end-length offsets (rigid zones, with the
rigid-zone factor). The bridge resolves it per element from the joints and the
element's `vecxz`, and emits one `geomTransf` per distinct `(vecxz, offsets)`,
as it already does for `orientation=` (ADR 0010), so beams of one section that
run in different directions share one transform. `insertion_point_offsets(...)`
returns the pair for one element. Sign conventions (CSI Analysis Reference,
"Insertion Points"): local 2 is the depth direction, 3 = 1 x 2; `axis_2` maps
ETABS 2 onto OpenSees local `y` or `z`; the centroid sits at minus the cardinal
point's section position, so cardinal 8 on a 0.6 m beam hangs its axis 0.3 m
below the joint.

ETABS puts a frame object's end offsets at its two ends, but the mesh alone
cannot tell a member meshed into several elements from a continuous girder
of several members. So an offset that is not the same transverse shift at
both ends (an end length, an axial component, unequal ends) refuses on any
collinear run of a transform's elements, naming the elements and the node;
mesh each member as one element and give collinear members separate
transforms. Member ends by curve are #2551. One-element members, chains
that turn a corner, and an equal transverse shift on every element work as
before.

Offsets are per transform, not per element spec: a `geomTransf` is OpenSees'
offset group, and the element ↔ transform join in `model.h5` stays one tag
per element.

A global line load (`from_model` `beamUniform`) on an offset beam covers its
full node-to-node length, as ETABS/CSI load a frame object: the share over
each rigid end zone goes to its joint as a nodal force plus moment, the rest
stays a `-beamUniform` on the flexible span (OpenSees alone would load only
the flexible length, losing `w` times the end-zone lengths). `Plain.beam_uniform`
stays OpenSees' own `eleLoad`, on the flexible element. A global line load on
a beam whose offsets turn its axis away from the node chord, or leave it no
flexible length, refuses: the load would be projected on the wrong axes.

Schema: opensees **2.37.0**. A transform group whose line carries
`-jntOffset` gains a `jnt_offset` dataset, `(1, 6)` in 3-D and `(1, 4)` in
2-D; lines without offsets write none. Each transform's `decl_params` row
gains a `jnt_offset` field (`null` without offsets), so an archive with a
transform changes bytes, but not its `model_hash` or replayed deck. H5
replay (staged brackets included), `ModelData`, and a rotated `Assembly`
instance (each half turns by `R d`) carry the offsets.

Validated live on stock openseespy and the Ladruno fork against explicit
offset nodes tied with `rigidLink beam`: static joint displacements agree to
4e-16 and the first six periods to 6e-14 (relative), for Linear and PDelta
(gravity, `loadConst`, eigen, lateral step), in 3-D (cardinal 8, and
cardinal 9 with end zones) and 2-D.
