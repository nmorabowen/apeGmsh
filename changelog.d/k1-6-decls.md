### ADDED — `/opensees/decls`, and `name=` on `fix` / `mass` / the recorders (opensees 2.25.0; K1-6, #1463)

`model.h5` gains `/opensees/decls`, the bridge's declarations (ADR 0114
R5). It holds one row per declaration key, `opensees/<family>/<name|#k>`,
which is the path of the declaration's `/provenance` record.

- **The tag join.** Each tagged declaration is joined on the tag plan's
  `(kind, tag)`: primitives, element and orientation fan-outs (which
  inherit their spec's row), and named, damping and recorder regions.
- **The `rows` columns.** Every row of the tagless stores carries its
  declaration: `bcs/fix`, `bcs/mass`, `recorders`, `initial_stress`, and
  the stage stores `remove_sp`, `remove_element`, `update_material_stage`,
  `activate_absorbing` and the `s.support` HOLD `sp_holds`.
- **Key-only declarations.** `ops.equation_constraint` and `s.profile`
  get a key but no row, since neither archives one.
- **New `name=`.** `ops.fix`, `ops.mass`, `s.fix` and `s.mass` take
  `name=`, and so do the five typed recorders.
- **How names read back.** Through `H5Model.declarations()` and
  `OpenSeesModel.declarations` (`for_row`, `for_tag`, `by_key`).
  `OpenSeesModel.to_h5` echoes the group.
- **A label, not structure.** `decls` is excluded from `model_hash`, so the
  hash is unchanged for every model, and `/opensees/program`'s `decl`
  column stays `-1`. No deck byte changes.
- **Damping.** `ops.damping.rayleigh` (global, region-scoped, staged) and
  `ops.damping.modal` are declarations: each `/opensees/commands` row (a
  global rayleigh, eigen + modal_damping, a stage's `s.profile` bracket),
  each stage `rayleigh` row and each region-scoped Rayleigh's region
  carries its declaration.

**Changed (behaviour): new refusals.** Each refusal names the duplicate
and both call sites. Names are unique per family and stay out of the
bridge-wide alias table (`/opensees/names`), so one name may serve, say,
a material and a recorder. Migration for every refusal: rename one of the
two.

- A repeated `name=` among the `fix` declarations (flat and staged
  together), and the same for `mass`.
- A repeated given `ops.recorder.declare(name=)`, or a repeated `name=` on
  any recorder. Without a name, `declare` stays unnamed and keeps its
  `"default"` file stem.
- A user `name=` of the form `@<k>`. That form is reserved for objects
  apeGmsh registers inside a recorded user call.
