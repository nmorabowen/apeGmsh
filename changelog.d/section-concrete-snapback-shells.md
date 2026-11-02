### CHANGED — the emit-time snap-back sweep reaches concrete inside shell sections

The emit-time element-size sweep (ADR 0044) now checks crack-band concrete that sits inside an element's section, not only concrete assigned directly as `material`. It walks the section's material graph, so it reaches `LayeredShell` layers, `PlateFiber` wrappers and a `LadrunoShellModifier` inner section. It covers ASDConcrete, `LadrunoConcrete3D` and, under `auto_regularization`, `LadrunoRCConcrete`/`LadrunoRCFiniteStrain`. The length compared with the ceiling is the host shell's own `getCharacteristicLength` at the fork pin:

- `ASDShellQ4`: the minimum node distance, halved while the EAS membrane is on (the default; `no_eas=True` keeps the full length);
- `ASDShellT3`: the minimum node distance.

`ShellMITC4`, `ShellDKGQ` and `ShellNLDKGQ` have no length to compare: the fork latches the characteristic length from another element for these hosts (#2680). An auto-regularized concrete in their section gets its own `ASDRegularizationWarning`, once per physical group, whatever the element size.

A `LadrunoConcrete3D` without `auto_regularize`, or an `_LadrunoRC` with `auto_regularization` and an explicit `lch`, is checked once against that fixed length, on any shell. Each physical group gets one `ASDRegularizationWarning` that lists every concrete over its ceiling. Fiber beams are not covered yet. A material with a zero `ft` or `Gf` is skipped, so the sweep never blocks emit.

This also changes concrete assigned directly as `material`: an ASDConcrete without `auto_regularize` no longer warns, because the fork uses its curves as given at any element size, and an `_LadrunoRC` with an explicit `lch` is now checked against that `lch`.
