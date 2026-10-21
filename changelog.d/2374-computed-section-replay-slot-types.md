### FIXED — a `ComputedSection` replays the bridge's literals without a `ReplaySlotTypeWarning`; the rehydrator's unreachable unknown-family refusal is gone (#2374)

Every replay of an archive that held a `ComputedSection` warned
`ReplaySlotTypeWarning`. The archive records the section's
`SectionProperties` analyzer by class name only, so the slot-type recovery
could not rebuild the declaration. The replay then wrote
`section Elastic 1 200000` where the bridge wrote `200000.0`, and an
integral float `G=80000.0` came back as `80000`.

The slot types do not depend on the analyzer, so the recovery no longer
rebuilds it. It dry runs the plain section the lowering emits (`section
Elastic` in the 2-D or 3-D form, or `section Fiber` with its `-GJ` flag),
with `1.0` for each number the analyzer computes and the archived value of
each field the user set. A replayed computed section now writes the
bridge's line, literal for literal, and does not warn:

- `E` and `G` the user set keep their type: `E=210000` stays `210000`.
- An `E` or `G` that the lowering takes from the analyzer's material
  replays as a float. `SectionMaterial` declares both as `float`, and the
  archive does not record the material. A material written with an int
  modulus (`SectionMaterial(E=200000, ...)`) therefore replays as
  `200000.0` where the bridge wrote `200000`. The two values are equal.
- Any other field the archive records by name only still warns.

In `apeGmsh.assembly`, the rehydrator's `AssemblyError` for an unknown
material family is deleted, because it could never fire. The model reader
already raises `MalformedH5Error` naming the family before a model exists.
A test pins that path, and a second test holds the rehydrator's family
table equal to the reader's. A family the reader gains later therefore
fails a test instead of being skipped.
