### `<stem>.user.h5`: the `/entities` zone, cuts, storeys and point sets (ADR 0130 slice 8)

`apeGmsh.results.user` now writes and reads `/entities`: section cuts (`CutEntity`),
storeys (`Storey`) and point sets (`PointSet`), held in an `Entities`.
`write_user_file(entities=...)` replaces the zone whole, and `UserFile.entities()`
reads it, with empty tables when the file holds none. `SectionCutDef.to_entity(name=,
reference=)` exports a cut: its plane, kept side, OpenSees element tags and bounding
polygon, with the point that a stored resultant's moments are about (the plane point
by default). `CutEntity.to_cut_def()` gives the def back, so the cut kernel computes the
same resultant from it. A `target="entities"` user result names its rows by
`ENTITY_KIND` (`cut`, `storey` or `point_set`) and `ENTITY_ROW`. A write refuses a
result whose row the file's `/entities` does not hold, including after an `entities`
rewrite, and the reader refuses any other kind. The `entities` zone joins the schema
registry at 1.0.0. `/spectra` stays at 1.0.0, holding nodal and domain values only.
