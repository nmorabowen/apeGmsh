# Parts — `g.parts`

A `Part` owns an isolated Gmsh session and exports a shape to STEP.
`g.parts` is the assembly-side registry that imports those STEPs back
in and tracks which tags belong to which label.

## `Part`

::: apeGmsh.core.Part.Part

## `g.parts` — registry

::: apeGmsh.core._parts_registry.PartsRegistry

## Instance

::: apeGmsh.core._parts_registry.Instance

## Part edit composite — `part.edit`

::: apeGmsh.core._part_edit.PartEdit

## Instance edit composite — `inst.edit`

::: apeGmsh.core._instance_edit.InstanceEdit

## Labels

::: apeGmsh.core.Labels.Labels
