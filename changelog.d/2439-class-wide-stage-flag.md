### FIXED — `s.update_material_stage` refuses a flip that names only some of a class-wide group's materials (#2439)

The stage flag of `ManzariDafalias`, `LadrunoSANISAND` and `SAniSandMS` is a
C++ `static`, so it holds one value per process. `updateMaterialStage
-material A` therefore switches every material of the group, not just `A`.
On a flip to 1 only `A` runs `Elastic2Plastic()`, so a material the call did
not name ends up with an answer that matches neither the elastic run nor a
proper flip. This is measured on stock openseespy 3.8.0 and on the fork.
`LadrunoSANISAND` inherits `ManzariDafalias`'s flag, so the two classes
switch each other.

A stage that flips some of a group's materials but not all of them now
raises `BridgeError` when the stage closes. So does a stage that flips them
to different values. The message lists the materials left out and names the
static. Flipping every material of the group, in one call or in several
calls within the same stage, still passes. The groups are listed in
`CLASS_WIDE_STAGE_FLAG` (`apeGmsh.opensees.stage._builder`), where a new
class-wide class takes one entry.

The check counts both routes that set the flag:

- `s.update_material_stage`;
- `s.update_parameter("materialState" | "updateMaterialStage", v, material=m)`.

**Fork-pinned bridges also refuse a stage that activates elements after an
earlier stage left the group plastic.** This applies to a bridge built with
`OpenSeesTarget(mode="fork")`, when the activated elements carry a
`ManzariDafalias` or `LadrunoSANISAND` material. The fork's constructor
resets the flag to 0, so the whole group would silently go back to elastic.
Stock keeps the flag. The refusal is gated by `FORK_RESETS_STAGE_STATIC`,
which is emptied once the fork fix lands.

**On every build, a flip to stage 1 or higher is refused for a
`ManzariDafalias` or `LadrunoSANISAND` material that more than one element
carries.** This covers both routes (#2445). `updateMaterialStage` runs
`Elastic2Plastic()` only on the first element it finds, so the material's
other elements turn plastic without being initialised. Stage 0 flips and
materials on a single element still pass, and `SAniSandMS`, `PM4Sand` and
`PM4Silt` are unaffected.

**Behaviour change:** a model that flipped a single `ManzariDafalias` or
`SAniSandMS` material beside others of its group used to build, and silently
switched the others too. It now refuses. Name every material of the group in
the flip.
