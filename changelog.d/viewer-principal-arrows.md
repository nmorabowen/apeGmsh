### ADDED — apeGmshViewer: principal stress and strain directions at the Gauss points

A new Principal card draws, at each Gauss point of a `.ladruno` stress or strain field, an arrow along each
principal direction: its length follows |p|, and its colour is the signed value on a diverging map (RdBu_r by
default, over −max..max), so tension and compression read apart. It is the viewer twin of the Qt
PrincipalDirectionDiagram and follows Python's `principal_frame`. A strain's shear is halved (the file's shear is
engineering), and a zz the element does not record is taken as 0 (plane stress for a stress, plane strain for a
strain); the card says so. A direction is an axis, so `axis` draws a head each way and `arrow` draws one along the
sign-canonical direction. The card also offers p1, p2 and p3 switches, the field, the colour map, scale, head size and
thin-out. The arrows follow the deformed shape. The step read gains a `tensor` request for several Gauss components at
once; the arrows start at their Gauss points through the arrow record's `anchor` field.
