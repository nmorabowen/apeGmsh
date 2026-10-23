### FIXED — a replayed archive runs the stage-flag route checks the stage close runs (#2465)

An archive written before the stage-flag route checks (#2442, #2443, #2445,
#2458, #2460) could hold a flip those checks now refuse.
`OpenSeesModel.from_h5(...).build(...)` replayed it silently, so the
replayed deck carried the wrong answer the checks exist to stop. The cases
are:

- an `updateMaterialStage` through `updateParameter`;
- a `materialState` through a host that drops it, such as `FourNodeQuad`;
- a tag-blind neighbour such as DruckerPrager;
- a target that carries no flipped material;
- a multi-element plastic flip;
- an activation after a plastic flip, or a fork constructor reset;
- part of a class-wide group.

The checks now live in one core in `stage/_builder.py`
(`check_stage_routes`, over a `RouteModel`). The stage close builds that
model from its primitives. The replay builds the same model from the
archive's `/opensees/decls` and `/opensees/decl_params`, then runs the same
checks, and the emit-time class-wide check, before anything is emitted. It
runs in `OpenSeesModel.build` for every target and in
`compose._replay_staged_into`. Such an archive is refused by name, with the
stage-close message. The valid archives tested here replay byte for byte.
The replay checks the final model, while the stage close sees only the
elements registered so far, so the replay may refuse an archive the bridge
accepted: one whose elements were registered after a flip (#2464). An
archive with a stage flip but no `/opensees/decl_params` (written below
opensees 2.26.0) cannot be checked, so it is refused too.
