### FIXED — `capture_live_modes` writes the real rotation at a beam node embedded in a solid (#2407)

On a plain openseespy deck, `capture_live_modes` wrote 0.0 into the
rotation slots of every node an `ASDEmbeddedNodeElement` touches, with
one `RotationSlotUnresolvedWarning`. The ssi_frame_wall bench's 93
grade-beam nodes were affected, since `g.constraints.embedded` emits that
element. The live rotation table treated the class as unknown. It now
reads the class by its nodes' live ndf. The element can carry a pressure
DOF (`-p`) only when every node it names has ndf `ndm + 1`. In any other
case it touches translations, plus, under `-rot`, the constrained node's
rotations, which stay rotations. So a beam node embedded in a solid keeps
the beam's rotation: the value written is the live `nodeEigenvector`'s.
If an embedded node keeps a rotation slot that no element settles (every
node at ndf `ndm + 1`, or no beam or shell at the node), the capture
raises `ValueError` and names the node; it no longer writes 0.0 there.
The same refusal now covers any rotation slot the table cannot settle
whose live value is non-zero in a mode. One example is a beam node of
`mixedBeamColumn` (`MixedBeamColumn3d`), a class the live table does not
map yet. A slot that is 0.0 in every mode keeps the warning. The stock
u-p elements (`quadUP`, `SSPquadUP`, `9_4_QuadUP`, `brickUP`,
`bbarBrickUP`, `SSPbrickUP`, `20_8_BrickUP`) now count as u-p on the
live route. Their pressure slot holds no rotation, so neither a refusal
nor a warning fires.
Bridge models are unaffected: `EigenResult.to_native` already wrote these
rotations.
