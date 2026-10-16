### REMOVED — `ops.element.ShellMITC3`; use `ops.element.ASDShellT3` for a 3-node shell (#2092)

No OpenSees build has a `ShellMITC3` element: the pinned Ladruno fork and
upstream both answer `element type ShellMITC3 is unknown`, so every model
that declared it failed at emit. The bridge has dropped the class, its
`ops.element` method, its readback token and its capability row.

Every spelling fails at once with a message that names the replacements:
`ops.element.ShellMITC3(...)` and `apeGmsh.opensees.element.ShellMITC3`
raise `AttributeError`, and `from apeGmsh.opensees.element import ShellMITC3`
raises `ImportError`. Use `ops.element.ASDShellT3`. `ShellDKGT` is the other
working 3-node OpenSees shell, but the bridge has no typed primitive for it.
The skill's removed-symbols registry lists the symbol, so the model-script
lint (rule R1) flags it. The MPCO and `.ladruno` readers still name a
`ShellMITC3` bucket from another writer's file in their unread-bucket warning.
