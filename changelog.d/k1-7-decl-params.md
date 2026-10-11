### ADDED — `/opensees/decl_params`: every declaration's parameters by field name, `transf_ref` / `integration_ref` / `section_ref`, and `params_names` under a ratchet (opensees 2.26.0; K1-7, #1464)

`model.h5` gains `/opensees/decl_params` (ADR 0114 A6/Q4, the K0-8
record): one entry per owner of every `/opensees/decls` row (a registered
primitive, a fix / mass / region / damping / initial-stress /
equation-constraint / stage record, or a model-wide declaration; a
`region` name declared by several calls lists every call, in order),
holding the owner's class and its parameters **by field name**, encoded
generically from `dataclasses.fields(owner)`. `DeclarationTable.params`
and `params_for(key)` return the entries as a tuple.

- **Every field shape is stored or refused, never skipped.** Scalars,
  tuples (nested kept), `str`-keyed mappings, a referenced primitive as
  its declaration key (`{"$decl": key}`), a value dataclass a field holds
  (a `Fiber` patch, a `ShellLayer`) as a struct, and the orientation and
  `SectionProperties` objects by class name. An `ndarray`, a set, an
  `Enum`, a `Fraction`, a non-finite float, an unlisted object or an
  unregistered primitive raises `H5DeclParamsError` at write.
- **References are declaration keys.** `transf_ref`, `integration_ref`
  and `section_ref` list every `GeomTransf` / `BeamIntegration` /
  `Section` key the row references, in field order (a `HingeRadau`'s
  three sections), as index runs (the V1 add-on).
- **`params_names`** names the argv slots of a declaration's store row
  where the archive finds the argv equal to the fields (`Steel01`:
  `fy E b`); a flag without a field (`Parallel -factors`), an element
  row, a `Fiber` block or a chain component stays unnamed. Every
  concrete registered primitive (187, inherited `_emit` included) the
  archive does not name is a line of
  `tests/opensees/contract/params_names_ledger.txt`
  (`unnamed` 35, `uncheckable` 21, `nostore` 107), checked in
  `lock-tests` by `test_verbs_lock.py` with the writer's own rule; the
  `unnamed` + `uncheckable` count may only shrink, and a new primitive
  that is neither named nor listed fails the test.
- **How it reads back.** `H5Model.declarations()` returns the rows as
  `DeclarationTable.params` (`DeclParamsRO`, with `DeclRef` /
  `DeclStruct` / `DeclOpaque` values; `params_for(key)`);
  `OpenSeesModel.declarations` carries them, and `to_h5` echoes the
  group verbatim.
- **A derived view, hash-excluded.** `decl_params` joins `decls` in
  `MODEL_HASH_EXCLUDED_CHILDREN`: `model_hash` is unchanged for every
  model, and no deck byte changes. Additive minor, 2.26.0; a 2.25.x
  reader refuses a 2.26.x file.
