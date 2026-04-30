# Sections — `g.sections`

Parametric structural cross-section builders. Each method creates
3D geometry directly in the active session and returns an `Instance`
with named sub-regions (flanges, web, end faces) ready for constraints
and loads.

## `g.sections`

::: apeGmsh.sections._builder.SectionsBuilder

## Solid sections

3D volumes suitable for solid elements (dim=3).

::: apeGmsh.sections.solid.W_solid

::: apeGmsh.sections.solid.rect_solid

::: apeGmsh.sections.solid.rect_hollow

::: apeGmsh.sections.solid.pipe_solid

::: apeGmsh.sections.solid.pipe_hollow

::: apeGmsh.sections.solid.angle_solid

::: apeGmsh.sections.solid.channel_solid

::: apeGmsh.sections.solid.tee_solid

## Shell sections

Mid-surface geometry for shell elements (dim=2).

::: apeGmsh.sections.shell.W_shell

## Profile sections

2D cross-sections for fiber analysis or sweep operations.

::: apeGmsh.sections.profile.W_profile
