### FIXED — viewer line diagrams never guess a 3-D beam frame (#2473)

The apeGmshViewer line diagram no longer draws a 3-D beam on a guessed local frame.
An element type the viewer cannot decode no longer offers its integer arguments as
`geomTransf` tags, so a node tag that equals a transform tag cannot pick the wrong
`vecxz`. The fan-out fallback from #2142 (no `transfTag` reference) still resolves a
decoded transform tag, but only when exactly one `vecxz` answers it. A `vecxz`
parallel to the beam is no longer swapped for the default in 3-D: `beamLocalAxes`
returns `null`, and shear, moment and curvature skip the beam. Both skips are counted
in the existing frame-skip notice. Axial force, torsion and 2-D models still draw on
the default frame.
