### ADDED — the `.ladruno` reader serves ASDConcrete3D's crack buckets, and `results.plot.cracks` draws a crack map

The reader dropped `material.crackPattern` and `material.crackWidth` with
a `GaussColumnDroppedWarning`, so a crack map had to read the file with
h5py. Both are now Gauss components: `crack_c{1,2,3}_{x,y,z}` holds up to
three vectors along the crack normals, each as long as that crack's
opening (cracking strain x lch), and `crack_width` (also token `cw`) holds
the largest opening. They are listed by `available_components()` and read
with `results.elements.gauss.get(...)`.

`results.plot.cracks(step=, plane="xz", pg=, min_opening=, half_length=)`
draws, on 2-D axes, one segment per open crack per Gauss point, across the
crack normal projected on the chosen plane, coloured and widened by
opening, over a faint outline of the selected elements.
