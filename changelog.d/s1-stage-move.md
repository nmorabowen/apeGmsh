### CHANGED — `_StageBuilder` moved to `apeGmsh.opensees.stage._builder` (S1-a, #1449)

The private `_StageBuilder` class (about 1,850 lines) left the `apesees.py` hub and now lives in `src/apeGmsh/opensees/stage/_builder.py`, together with `STAGED_MATERIAL_CLASSES` and `_build_initial_stress_record`, which `apeSees.initial_stress` imports back. Class bodies are unchanged (two function-local relative imports gained one dot for the deeper package). The name is underscore-private, so `apesees.py` keeps no re-export; `ops.stage(...)` is unaffected.
