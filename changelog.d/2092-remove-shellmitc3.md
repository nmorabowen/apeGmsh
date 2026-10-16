### REMOVED — `ops.element.ShellMITC3`; use `ops.element.ASDShellT3` for a 3-node shell (#2092)

No OpenSees build has a `ShellMITC3` element: the pinned Ladruno fork and
upstream both answer `element type ShellMITC3 is unknown`, so every model
that declared it failed at emit. The bridge has dropped the class, its
`ops.element` method, its readback token and its capability row.

Every spelling fails at once. These paths raise an error that names the
replacements (`ops.element.ASDShellT3`, and `ShellDKGT`, the other working
3-node OpenSees shell, which the bridge has no typed primitive for):

- `ops.element.ShellMITC3(...)` raises `AttributeError`;
- a capability lookup by class name (`element_capability("ShellMITC3")` and
  every `element_*` helper) raises `ValueError`;
- a rehydrated model whose element token is `ShellMITC3` raises `ValueError`
  when its rotation roles are read.

The module spellings raise Python's own error, which does not name the
replacements: `from apeGmsh.opensees.element import ShellMITC3` raises
`ImportError`, and `apeGmsh.opensees.element.ShellMITC3` raises
`AttributeError`. A module-level hook could name them, but it would make
the static checks (the stale-patch-target lint, the skill's fence resolver)
accept every misspelled name in those modules. The skill's removed-symbols
registry lists the symbol, so the model-script lint (rule R1) names the
replacement for these spellings.

The MPCO and `.ladruno` readers still name a `ShellMITC3` bucket from
another writer's file in their unread-bucket warning.
