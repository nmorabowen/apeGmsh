### FIXED — MPCO layered-shell rows carry the shell's frame, not an identity quaternion (#1996)

`Results.from_mpco(...).elements.layers.get(...)` filled
`LayerSlab.local_axes_quaternion` with `(1, 0, 0, 0)` for every shell, with no
warning. No shell answers `localAxes`, so the MPCO writer stores no
`MODEL/LOCAL_AXES` row for one, and for any shell outside the global xy plane
the identity is a wrong frame (a strip tilted 30 deg about x read
`(1, 0, 0, 0)` instead of `(cos 15, -sin 15, 0, 0)`). The reader also looked
for frames under a per-class `LOCAL_AXES/<key>/QUATERNIONS` layout that the
writer never produces; it now reads the writer's flat `ID` + `LOCAL_AXES`
pair. An element without a stored frame gets the frame computed from the
file's own connectivity and node coordinates, in the convention DomainCapture
writes (`results._shell_geometry.shell_quaternion`, which is also the MPCO
writer's beam convention). When that cannot be done (a missing connectivity
row or node, or degenerate geometry) the row is NaN and a
`ShellFrameUnavailableWarning` names the elements and the reason; a
`LOCAL_AXES` group in an unrecognised layout warns too and is not read as a
frame.
