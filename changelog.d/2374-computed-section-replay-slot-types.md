### FIXED — a `ComputedSection` replays the bridge's literals without a `ReplaySlotTypeWarning`; the rehydrator's unreachable unknown-family refusal is gone (#2374)

Every replay of an archive that held a `ComputedSection` warned
`ReplaySlotTypeWarning`. The archive records the section's
`SectionProperties` analyzer by class name only, so the slot-type recovery
could not rebuild the declaration. The replay then wrote
`section Elastic 1 200000` where the bridge wrote `200000.0`, and an
integral float `G=80000.0` came back as `80000`.

The slot types do not depend on the analyzer, so the recovery no longer
rebuilds it. It dry runs the plain section the lowering emits (`section
Elastic` in the 2-D or 3-D form, or `section Fiber` with its `-GJ` flag).
The numbers the analyzer computes by division (`A`, `Iz`, `J`, the shear
factors) are always floats. The moduli `E` and `G`, and the fiber `GJ`,
take the values that the section's `/opensees/computed_sections` row
records (`E_ref`, `G_ref`, `GJ`). That row's JSON keeps each value's
int/float type. A replayed computed section now writes the bridge's line,
literal for literal, and does not warn:

- `SectionMaterial(E=200e3, nu=0.3)` replays `200000.0`, and
  `SectionMaterial(E=200000, nu=0.3, G=80000)` replays `200000` and
  `80000`, as the bridge wrote them.
- `E` and `G` the user set on the section keep their type: `E=210000`
  stays `210000`.
- A computed section whose tag has no single well-formed row in that table
  warns and recovers by value, as before. So does any other field the
  archive records by name only.

In `apeGmsh.assembly`, the rehydrator's `AssemblyError` for an unknown
material family is deleted, because it could never fire. The model reader
already raises `MalformedH5Error` naming the family before a model exists.
A test pins that path, and a second test holds the rehydrator's family
table equal to the reader's. A family the reader gains later therefore
fails a test instead of being skipped.
