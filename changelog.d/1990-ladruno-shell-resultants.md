### FIXED — `.ladruno` reads shell stress resultants, and places rule-0 and triangle-shell Gauss points (#1990, #1998)

`Results.from_ladruno(...).elements.gauss.get(component="membrane_force_xx")`
returned an empty slab, with no warning, for every shell recorded with
`-E stresses`: ASDShellQ4 (plate or layered section), ASDShellT3, ShellMITC4,
ShellDKGQ, ShellNLDKGQ, ShellDKGT and ShellNLDKGT. The fork writes their
resultants as `p11,p22,p12,m11,m22,m12,q1,q2` at each Gauss point, and the
reader knew none of those column names. It now reads them as
`membrane_force_*`, `bending_moment_*` and `transverse_shear_*`, in the
element's source order, which is also the response catalog's 8-component
shell layout. The names must match as a whole tuple. A partial or reordered
set raises `GaussLayoutMismatch`, and so does a class the catalog knows
under another layout or Gauss-point count.

SixNodeTri and EightNodeQuad read their values correctly but returned Gauss
points with natural coordinates of shape `(n, 0)`, with no warning. The fork
writes them under integration rule 0 with no `GP_PARAM`. The triangle shells
write their own rule but no `GP_PARAM` either. When the file has no
coordinates, the reader now takes them from the one catalog layout that fits
the bucket's class, rule (any rule, under rule 0), Gauss points and
components. If fitting layouts disagree on where the points are, it raises
`GaussLayoutMismatch`. If no layout fits, it warns
`GaussCoordsUnavailableWarning`, names the bucket, and returns the values
without coordinates.

The fork lane checks every case against MPCO on the same solve at rtol 1e-10.
The SixNodeTri and EightNodeQuad coordinates are also checked against
`D B(xi) u`. The strict xfails that cited #1990 and #1998 in
`tests/opensees/integration_ladruno/test_results_catalog_hdf5_live.py` are
removed.
