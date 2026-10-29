### CHANGED — The RSA parity tolerance compounds its parts, and refuses incomplete input (chain A, A6 follow-up)

`ParityTolerance.total` is now `(1 + base)(1 + period) - 1`. This
supersedes the `total = base + period` that #2616 shipped: the plain sum is first order and read 6.27 % where the exact
gate is 6.32 % for a 1 % base on a 5.27 % period part. `str()` prints the
compounded form. `combined_period_tolerance` with the default `rule="CQC"` and
no `damping=` now raises a `ValueError` that names the fix, instead of an error
from the combination kernel. `spectrum_slope` refuses a `rel_step` outside
`(0, 1)`, which gave an infinite slope. Every function now takes a spectrum
object whose `.at(T)` is its only `Sa` (a codeSpectra `Spectrum`), as
`DesignSpectrum` does.
