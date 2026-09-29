### ADDED — element-capability lookups fail closed on an unknown class (program slice C1.3, #1228)

`apeGmsh.opensees._element_capabilities` gains `element_capability(class_name)`,
the single registry lookup, and the `Unknown` sentinel it returns for any
class `_ELEM_REGISTRY` cannot resolve after `_CLASS_TOKEN_ALIASES`. `Unknown`
is neither `None` nor falsy: `bool(Unknown)` raises, so a caller cannot read
it as the permissive "skip" that the per-helper `None`/`False` answers still
mean. Every `element_*` helper now routes through it; their public return
contracts are unchanged. The new lock
`tests/opensees/unit/test_element_capability_unknown.py` enumerates every
concrete `Element` subclass under `apeGmsh.opensees.element` and fails when
its lookup is `Unknown`, unless the class sits on a reasoned, shrink-only
exception list (the ten classes carried by `_EXTRA_CLASS_NDF_OK` today: the
beamIntegration beams, the zero-length family, `InertiaTruss`, `ASDShellT3`).
A stale exception fails too. Callers in `_internal/build.py` still use the
`None`-returning helpers; switching them to the fail-closed lookup is a
follow-up.
