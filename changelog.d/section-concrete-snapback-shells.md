### CHANGED — the emit-time snap-back sweep reaches concrete inside shell sections

The emit-time element-size sweep (ADR 0044) now checks crack-band concrete that sits inside an element's section, not only concrete assigned directly as `material`. It walks the section's material graph, so it reaches `LayeredShell` layers, `PlateFiber` wrappers and a `LadrunoShellModifier` inner section. It covers ASDConcrete, `LadrunoConcrete3D` and, under `auto_regularization`, `LadrunoRCConcrete`/`LadrunoRCFiniteStrain`. The length compared with the ceiling is the host shell's own `getCharacteristicLength` at the fork pin:

- `ASDShellQ4`: the minimum node distance, halved while the EAS membrane is on (the default; `no_eas=True` keeps the full length);
- `ShellMITC4`, `ShellDKGQ`, `ShellNLDKGQ` and `ASDShellT3`: the minimum node distance.

A `LadrunoConcrete3D` without `auto_regularize` is checked once against its fixed `lch`. Each physical group gets one `ASDRegularizationWarning`, which lists every concrete over its ceiling. Fiber beams are not covered yet. The sweep never blocks emit.
