### CHANGED — every session run writes `model.h5` and a `<stem>.geometry.h5` sibling (ADR 0112 D1/D2a, V2b, #1305)

`apeGmsh.end()` now writes its artifacts **unconditionally**. With no
`save_to`, `model.h5` goes to the conventional path
`<dir>/<model_name>.h5`, where `<dir>` is `$APEGMSH_ARTIFACT_DIR` when set,
else the directory of the running script, else the current directory
(`apeGmsh._core.default_artifact_dir`); `save_to=` stays as the path
override. Beside it the session writes `<stem>.geometry.h5`, the tessellated
geometry of the CAD model (`/geometry`: CSR per dimension, int32, labels
and physical groups as memberships), written by the new gmsh-only
`mesh/_geometry_h5_io.py`. Surfaces are captured from the real mesh at the
exit of `g.mesh.generation.generate()` (curves sampled with 32 parametric
points); a session that never meshed gets a 2-D surface-only temporary mesh
at `0.03 x bbox diagonal`, captured and cleared again at `end()`. The
session owns one uuid4 `session_id` from `begin()`: every snapshot it
extracts and the geometry sibling carry it, so a reader pairs the two
files by equality. A failed write or capture is a warning
(`GeometryArtifactWarning` for the sibling), never an exception, and an id
or count outside int32 is refused before any file is written. Sessions
without a kernel (`from_h5`) and `Part` sessions write no geometry. The
test suite pins `APEGMSH_ARTIFACT_DIR` to a temporary directory and proves
it leaves no `.h5` in the repository.
