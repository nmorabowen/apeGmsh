### ADDED — turnkey concrete calibration from `fc` (DR-6)

`LadrunoConcrete3D.from_fc(fc, units=..., aggregate=...)` builds the material from the mean strength.
`fib_mc2010_concrete_params` in `apeGmsh.opensees.material._asdconcrete_laws` derives the parameters:
`ft` (fib MC2010 Eq. 5.1-3), `E` = `Eci` (MC2010 §5.1.7.2, aggregate factor from Table 5.1-6), `Gf`
(MC2010 Eq. 5.1-9), `Gc = 250·Gf` (the Nakamura & Higai ratio; MC2010 gives no compressive fracture
energy), `eps_c1` (EN 1992-1-1 Table 3.1), and the CDPM2 `Df = 0.85`, `Hp = 0.01`, `As = 15`
(Grassl et al. 2013). `units` is an explicit preset (`MPa-mm-N`, `Pa-m-N`, `kPa-m-kN`, `psi-in-lbf`,
`ksi-in-kip`). Nothing is inferred from the size of `fc`. `ceb_fip_Gf`, `ASDConcrete3D.from_fc` and
`ASDConcrete1D.from_fc` take the same `units=` keyword. Its default (`MPa-mm-N`) gives the same results
as before.

`LadrunoConcrete3D.l_max()` and `check_element_size()` return the crack-band snap-back ceiling,
`k·E·Gf/ft²`, with `k = 20/21` for the bilinear law and `k = 1` for the exponential law. Nothing calls
them automatically. `LadrunoRCConcrete` / `LadrunoRCFiniteStrain` gain `l_max()`, read off the emitted
tables as the fork's `-autoRegularization` floor sees them. `LadrunoRC*.from_fc` keeps its historical
calibration by default; `calibration="mc2010"` opts into the new one.

Changed: a `LadrunoConcrete3D` deck always carries `-Df` and `-hardening`, so it runs the values Python reports whatever the fork's
constructor default is. The defaults are unchanged (`Df = 1.0`, `hardening = (0.3, 0.5)`, the pinned
fork's own), so existing models give the same results. Only the deck text is longer. `hardening` must be
a 2-tuple, as the fork reads two values. `Gc` may be `None`, in which case `250·Gf` is emitted.
`make_tension` / `make_compression` now refuse a backbone point with damage `d >= 1` instead of passing it
on. `ASDConcrete3D` and `LadrunoRC*` already rejected such a point at construction.
