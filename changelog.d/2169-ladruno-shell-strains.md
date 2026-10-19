### FIXED — `.ladruno` reads shell `-E strains` as generalised section strains (#2169)

`Results.from_ladruno(...)` read a shell's `-E strains` bucket as the
continuum `strain_xx` and `strain_yy`, with no warning, and dropped the other
six columns. The fork writes each Gauss point's section deformation as
`eps11,eps22,gamma12,theta11,theta22,theta33,gamma13,gamma23`: membrane
strains, curvatures (`theta33` is the twist) and transverse shear strains.
The reader now reads them as `membrane_strain_*`, `curvature_*` and
`transverse_shear_strain_*`, the response catalog's
`SHELL_GENERALIZED_STRAINS`, for ASDShellQ4, ASDShellT3, ShellMITC4,
ShellMITC9, ShellDKGQ, ShellNLDKGQ, ShellDKGT and ShellNLDKGT. The names must
match as a whole tuple in the element's write order. A partial or reordered
set raises `GaussLayoutMismatch`, and so does a class the catalog knows under
another layout or Gauss-point count. Continuum elements keep reading `eps*`
as `strain_*`.

A named Gauss bucket with no `GP_PARAM` now takes its coordinates from the
catalog under any integration rule its key states, not only rule 0 and the
shell resultants. When the catalog has no layout for the class and rule, or
the key states no rule, the reader warns `GaussCoordsUnavailableWarning`,
names the bucket, and returns the values without coordinates. Before, such a
bucket returned coordinates of shape `(n, 0)` with no warning.

The fork lane checks every shell's `-E strains` against MPCO on the same
solve at rtol 1e-10.
