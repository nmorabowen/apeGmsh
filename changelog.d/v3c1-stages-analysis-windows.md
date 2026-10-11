### ADDED — apeGmshViewer: the stage diagram and the analysis window (V3c1, #1642)

The **Stages** window now draws the model's sequence from the root
`/sequence` zone as a horizontal timeline. Each stage shows its name, its
analysis type, its integrator, algorithm and test, its steps and dt (`static`
when there is no dt), and the changes it makes (add, remove, activate or set
of a pattern, recorder, fix and so on), each by name, or by its `#k` key when
unnamed, with its count. The declarations made outside any stage are listed
once, before the timeline. An unstaged deck shows its one implicit stage, and
a deck that runs its own analysis loop says that no step count is archived
instead of showing `-1`. A file without `/sequence` says it predates readable
stages. A note on the diagram says that `update_parameter`,
`zero_velocities`, `set_creep_on` and `pre_analyze_reset` are not recorded
in `/sequence` (#1646).

The **Analysis** window lists, for each stage, the system, numberer,
constraints handler, integrator, algorithm, test and analysis, each with its
parameters by field name from `/opensees/decl_params` (for example
`LoadControl{dlam=0.5, ...}`), never from positional arguments. When a chain
cannot be matched to its declarations, the window says why and lists no
parameters. In both windows, a click on a declaration with a recorded source
opens that source.
