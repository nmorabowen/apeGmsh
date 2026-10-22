### FIXED — a plastic `ManzariDafalias` / `LadrunoSANISAND` flip now initialises every element (#2445, #2446)

`updateMaterialStage` runs `Elastic2Plastic()` only on the first element that
carries the material, because `MaterialStageParameter::setDomain` stops
there. Every other element of the material turned plastic without being
initialised, on stock openseespy 3.8.0 and on the fork alike.

**Behaviour change:** `s.update_material_stage(..., stage=1)` of such a
material on two or more elements now updates each of them before the flip
line, on the Tcl and Python decks and in the archive's replay:

```tcl
setParameter -val 1.0 -ele <every element carrying the material> materialState <tag>
updateMaterialStage -material <tag> -stage 1
```

Each listed element runs its own stage update, `Elastic2Plastic()`
included. The list names the carriers only. DruckerPrager, BoundingCamClay
and the stress-density models accept `materialState` without checking the
tag, so an element that carries one of them is never offered it. OpenSees
numbers and removes the transient parameter itself, so the deck carries no
new tag.

The archive is unchanged: it still stores the flip as `(tag, stage)`, so
`model_hash` does not move. A stage 0 flip, a flip of any other class and a
material on a single element still emit the single `updateMaterialStage`
line.

The answers change, and they are now correct. The cases measured were two
disconnected elements on one material, a four-brick column and LadrunoUP
elements. In each, the flip matches the reference to the last bit, on stock
3.8.0 and on the fork; in the reference each element has its own material
and its own flip.

The #2445 refusal of every multi-element plastic flip, added in #2442, is
lifted where the update reaches every element. That is the case when every
element carrying the material is a `SixNodeTri`, `FourNodeTetrahedron`,
`BezierTri6`, `BezierTet10` or `LadrunoUP`
(`MATERIAL_STATE_ELEMENTS` in `apeGmsh.opensees.stage._builder`). The table
also lists `SSPquad` and `SSPbrick`, measured on both builds, for the bridge
primitive that will emit them. No bridge primitive emits either one yet.

The flip still raises `BridgeError`, naming the class, when another class
carries the material on more than one element. `FourNodeQuad`, `Tri31` and
`stdBrick` read any `material...` parameter as a per-point address, so they
drop the update silently. `LadrunoQuad`, `LadrunoCST`, `LadrunoLST`,
`LadrunoBrick` and `LadrunoBrick20` dropped it on fork build 79e062367. That
build is not `FORK_PIN`, and the fork source at the pin forwards it. They stay
refused until #2457 re-measures them at the pin, and their refusal cites that
issue.

An element whose material is a `PlaneStrain` or `StagedStrain` wrapper over
the sand counts as one of its elements, and the update reaches the wrapped
sand. Reaching the sand through any other primitive raises `BridgeError`,
naming that primitive. This includes `LogStrain`, `LogStrain2D` and
`InitDefGrad`, which are not yet measured.

The other rules:

- `s.update_parameter("materialState", v, material=m)` on two or more
  elements now passes when its targets in the stage cover every element
  carrying `m`, and refuses when they do not.
- `s.update_parameter("updateMaterialStage", ...)` refuses on two or more
  elements: that parameter id reads an integer that `updateParameter` never
  sets.
- On every build, a stage that activates an element of such a material
  after an earlier stage left its group plastic now refuses, unless the
  stage flips the group again.
