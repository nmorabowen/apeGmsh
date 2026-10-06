### CHANGED — verify_move compares relative imports by resolved module (S1-v, #1468)

`scripts/verify_move.py` now rewrites every relative `from . import` to its
absolute module, using the file's path under `src/`, before comparing bodies.
A def moved one package deeper (`from .x` becoming `from ..x`) now passes when
both forms name the same module, and still fails when the target differs. Files
outside `src/` keep the raw comparison.
