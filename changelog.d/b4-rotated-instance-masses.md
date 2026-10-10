### FIXED — a rotated instance turns its anisotropic nodal masses or refuses (#1600)

`Assembly.instance(rotate=...)` followed by `bridge` now carries each mass
row through the instance frame rule: the translational triple `(mx, my, mz)`
and the rotary triple `(Ixx, Iyy, Izz)` turn as diagonal tensors, `R·diag·Rᵀ`
(a value per DOF is a tensor, not a DOF index). An axis-aligned rotation
(a multiple of 90° about x, y or z, or any rotation that leaves the tensor
diagonal) emits the permuted source values; a rotation that makes the tensor
non-diagonal raises `ComposeError` naming the instance, the nodes and the
remedy, because OpenSees `mass` takes a diagonal only. Before, every mass row
was copied verbatim, so a rotated anisotropic mass was silently wrong.
Isotropic masses, `rotate=None` and instances without masses are
byte-identical to before. "Diagonal" is decided per off-diagonal entry, with a
relative tolerance of 1e-9 against the larger of its two turned diagonal
values and a floor of 64 eps times the triple's largest value (so a zero pair,
such as the diaphragm rotary `(0, 0, Izz)`, keeps its roundoff admissible); a
rotated instance whose mass is not finite is refused. The mass row joins
the per-type frame table (`_MASS_FIELDS`); an unclassified `MassRecord` field
raises `TypeError`.
