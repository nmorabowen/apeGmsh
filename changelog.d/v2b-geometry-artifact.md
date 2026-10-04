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
files by equality. A model that already carries 2-D elements (a `from_msh`
import) is captured as it is, with nothing generated or cleared. The
automatic write never clobbers a foreign file: it replaces a target only
when it does not exist or is an apeGmsh artifact (a per-zone `/meta`
version key; the generic `schema_version` attribute alone is not proof)
that holds no zone the write would drop (an `apeSees(fem).h5()` at the
model path keeps its `/opensees`), honours `overwrite=False`, and
otherwise warns once and skips that file, the explicit `save_to=` target
included; both files go through a `<target>.tmp-<uuid>` beside the target and
`os.replace`, so a failed write leaves the previous file untouched. A
failed write or capture is a warning (`GeometryArtifactWarning` for the
sibling), never an exception, and an id or count outside int32 is refused
before any file is written. Sessions without a kernel (`from_h5`) and
`Part` sessions write no geometry; library-internal sessions (the section
mesh worker, `interop.solve`, `interop.strut_tie`, the results demo) opt
out through a private constructor flag, and there is no user-facing
opt-out. The test suite pins `APEGMSH_ARTIFACT_DIR` to a temporary
directory and proves it leaves no `.h5` in the repository.
