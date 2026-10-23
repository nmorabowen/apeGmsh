### CHANGED — a plastic `ManzariDafalias` flip on Ladruno solids now passes and initialises every element (#2457)

#2446 refused a multi-element plastic `ManzariDafalias` / `LadrunoSANISAND`
flip carried by `LadrunoQuad`, `LadrunoCST`, `LadrunoLST`, `LadrunoBrick` or
`LadrunoBrick20`. On fork build 79e062367 those classes dropped the
`materialState` update that initialises each element.

They were re-measured on the live-fork lane at `FORK_PIN` (100eb9ae). Each
case had two elements on one material, and the flip ran against a reference
in which each element has its own material and its own `updateMaterialStage`.
Every formulation the bridge emits matches the reference to the last bit:
`LadrunoQuad` std, bbar and ssp; `LadrunoCST`; `LadrunoLST`; `LadrunoBrick`
std, bbar, uri, ssp and eas, plus corot std and bbar; `LadrunoBrick20` std
and uri.

**Behaviour change:** the five classes join `MATERIAL_STATE_ELEMENTS`. The flip
of such a material now emits

```tcl
setParameter -val 1.0 -ele <every carrier> materialState <tag>
updateMaterialStage -material <tag> -stage 1
```

and `s.update_parameter("materialState", ...)` on their elements passes.

These classes still drop the update on a fork build older than the pin. On
such a build, a deck that sweeps them stops at its start and names the fork
floor (the #2504 guard), so the wrong answer is never reached. The five
classes come from that guard's table, `MATERIAL_STATE_FORK_FLOOR`.

`LogStrain`, `LogStrain2D` and `InitDefGrad` over `ManzariDafalias` stay
refused for a multi-element plastic flip:

- `LogStrain` and `LogStrain2D` end the process at the pin
  (`ManzariDafalias::getType`, fork defect #2499).
- `InitDefGrad` refuses a small-strain inner.

`LiveOpsEmitter.update_material_stage` sends a bare `updateMaterialStage`
with no carrier sweep. It now raises `BridgeError` for a stage ≥ 1 flip of a
swept material that two or more elements may carry, on any build. No bridge
path reaches it today, because live execution refuses staged models.
