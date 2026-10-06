### FIXED — Region tags come from the build-time tag plan; a stage-claimed filtered recorder no longer writes its regions twice under a partitioned staged emit (K1-3d S4, #1456, #1446)

The region allocation loop moved out of the emit into `plan_regions`, which
`plan_tags` runs once per emit mode over `BuiltModel._region_sites`: named
regions (global and per stage), region-scoped Rayleigh, damping attaches,
and recorder filter and energy regions, in the order each mode's deck
always numbered them. A partitioned emit numbers each named region on the
first rank that holds a member, as before. The emit's region writers read
their tags from the plan, and the emit fork now freezes `region`, so an
emit-time region mint raises `TagLawError`. The writers refuse a plan that
lacks a region they write or holds two regions swapped.
`FilterableRecorder.materialize` keeps its signature and reads its tags
through the new `planned_region_tags`, which plans them from a plain
`TagAllocator` for a direct caller until K1-3d S6.

Fix (#1446): the partitioned pre-plan of filtered recorders also gave each
stage-claimed recorder's regions a tag and wrote them in every rank block
before the first stage, and the stage then wrote them again under fresh
tags. The pre-planned regions were declared and never referenced. A
stage-claimed recorder's regions are now written only in its stage, under
one tag each, so a partitioned staged deck with such a recorder declares
fewer regions and numbers the later ones lower. Every other deck, and
every golden cell, is byte-identical.
