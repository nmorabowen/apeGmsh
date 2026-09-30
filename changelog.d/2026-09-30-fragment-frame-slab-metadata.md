### FIXED — `boolean.fragment` of column lines into slab surfaces: no stale `_metadata`, and `remove_orphans()` keeps the free column pieces

`g.model.boolean.fragment(slabs, columns + points, dim=2)` left keys of
consumed entities in `model._metadata`, so `g.mesh.generation.generate()`
raised `GeometryValidationError: model._metadata has stale entries`. The
leaked keys were the consumed lines' end points: they are sub-entities of a
fragment input, not inputs, and only inputs were popped. The 2D-only branch
skips `sweep_dangling`, so nothing reaped them later. `_bool_op` now reaps
every consumed input *and its boundary closure* itself. It also judges
"consumed" by the input's own `result_map` entry, because OCC reuses freed
tags: a column's old tag coming back as a slab edge no longer keeps a
mis-registered key.

`g.model.geometry.remove_orphans()` then deleted the fragmented column
segments, because they bound no face and their new tags were not in
`_metadata`. The pieces of a registered curve or point now inherit its
registration, so free beam and column segments and embedded points survive
the sweep. Surfaces are excluded: the overhang of a fragmented cutting plane
is still swept. The frame + slab gotcha in the skill no longer recommends
`remove_orphans()` as a workaround, and `examples/footfall_two_bay_shell.py`
drops the call.
