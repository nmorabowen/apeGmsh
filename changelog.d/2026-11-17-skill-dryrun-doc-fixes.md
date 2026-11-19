### FIXED — skill docs corrected after a blind dry run

A fresh agent building a Scordelis–Lo shell from the apeGmsh skill hit stale docs. The skill now shows the real
`g.model.transforms` signatures: `revolve(tags, angle, *, x, y, z, ax, ay, az, dim=2, ...)`, and the gmsh-style
positional form raises `TypeError`. It shows `g.model.select(...).in_box(lo, hi)` with two 3-tuples instead of six
numbers, and says that `save_to=None` still writes `<model_name>.h5` + `.geometry.h5` on `end()` (ADR 0112 D1). It
also documents `APEGMSH_ARTIFACT_DIR` and the bridge's own automatic `<model_name>.h5` write, which `save_to=` does
not move (`gotchas.md`, `fem-broker.md`, `opensees-bridge.md`, `api-cheatsheet.md`).
