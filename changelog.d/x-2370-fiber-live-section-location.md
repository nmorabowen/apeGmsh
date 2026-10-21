### FIXED — long-hinge fibres over a FEMData archive read at their live sections (#2370)

A HingeRadau rule with `lp_i + lp_j >= L/4` writes its stations out of
order, legally. Over a FEMData archive (`spec.emit_default` with a raw
openseespy deck), its line-stations record has read since #2328, but its
`fibers` record was still refused with the order check's HingeRadau
message. The fibre read now uses the line-stations proof: each element's
file stations must be the section locations the live domain gave before the
run, and each fibre table is then read at the section that wrote it. A file
whose stations are not the live ones is still refused by FEM id, including
one a single station off by 1e-3 and one with two stations swapped. The
bridge route (`ops.h5(...)`) keeps the order check unchanged.
