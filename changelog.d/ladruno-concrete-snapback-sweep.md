### CHANGED — the emit-time snap-back sweep covers the Ladruno concretes

The emit-time element-size sweep (ADR 0044) now checks `LadrunoConcrete3D` and `LadrunoRCConcrete`/`LadrunoRCFiniteStrain` as well as ASDConcrete. It warns once per physical group with `ASDRegularizationWarning` when the length the fork regularizes with exceeds the material's snap-back ceiling `min(l_max_t, l_max_c)`:

- under `auto_regularize`, each element's own characteristic length;
- otherwise, the fixed `lch` that every element uses, so a model built in SI with the default `lch=1.0` can now warn;
- for the RC materials, only under `auto_regularization`.

An element's length follows the fork's per-class `getCharacteristicLength` override: `cbrt(V)` for `LadrunoBrick`/`LadrunoBrick20`, `cbrt(6V)` for `BezierTet10`, `sqrt(A)` for `LadrunoQuad`, and `sqrt(2A)` for `LadrunoCST`/`LadrunoLST`/`BezierTri6`. Every other class uses the minimum node distance. The sweep never blocks emit; promote it in CI with `-W error::...ASDRegularizationWarning`.
