### FIXED — a plastic `ManzariDafalias` / `LadrunoSANISAND` flip now initialises every element (#2445, #2446)

`updateMaterialStage` runs `Elastic2Plastic()` only on the first element that
carries the material, because `MaterialStageParameter::setDomain` stops
there. Every other element of the material turned plastic without being
initialised, on stock openseespy 3.8.0 and on the fork alike.

**Behaviour change:** `s.update_material_stage(..., stage=1)` now emits an
element sweep before the flip line, on the Tcl and Python decks and in the
archive's replay:

```tcl
setParameter -val 1.0 -eleRange 1 2147483647 materialState <tag>
updateMaterialStage -material <tag> -stage 1
```

The sweep offers `materialState <tag>` to every element. Each element that
carries the material runs its own stage update, `Elastic2Plastic()`
included, and other elements decline it. OpenSees numbers and removes the
transient parameter itself, so the deck carries no new tag. The archive is
unchanged: it still stores the flip as `(tag, stage)`, so `model_hash` does
not move. A stage 0 flip is still the single `updateMaterialStage` line.

The answers change, and they are now correct. On two disconnected elements
on one material, a four-brick column and LadrunoUP elements, the swept flip
matches the reference to the last bit. In the reference each element has its
own material and its own flip. The match holds on stock 3.8.0 and on the
fork.

The #2445 refusal of every multi-element plastic flip, added in #2442, is
lifted where the sweep reaches every element. That is the case when every
element carrying the material is a `SixNodeTri`, `FourNodeTetrahedron`,
`BezierTri6`, `BezierTet10` or `LadrunoUP`
(`MATERIAL_STATE_ELEMENTS` in `apeGmsh.opensees.stage._builder`).

The flip still raises `BridgeError`, naming the class, when another class
carries the material on more than one element. `FourNodeQuad`, `Tri31` and
`stdBrick` read any `material...` parameter as a per-point address, so they
drop the sweep silently. On the fork build that was measured, `LadrunoQuad`,
`LadrunoCST`, `LadrunoLST`, `LadrunoBrick` and `LadrunoBrick20` drop it too.

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
