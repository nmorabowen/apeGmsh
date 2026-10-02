### ADDED — STKO `.scd` translator: STKO's own mesh into a session and an apeSees deck (ADR 0111)

`apeGmsh.interop.stko` now translates an STKO document, not only reads it.
`translate_scd(g, scd)` puts STKO's own mesh into an empty session as gmsh
discrete entities, with STKO's node and element ids and its shell node order.
It adds physical groups for each element group, selection set and condition,
and declares the rigid diaphragms on `g.constraints`. It also returns a pure
data plan of the masses, loads, fixities, time series, patterns, Rayleigh
damping and analysis stages, computed with STKO's own exporter rules.
`build_opensees(fem, result)` declares the sections, materials (including
STKO's ASDConcrete "Concrete (9P)" preset and the stored fibers) and elements
on an `apeSees` bridge, then the conditions and the static stages. The deck's
element tags are STKO's element ids.
`collect_unsupported(scd)` lists every type or option this version does not
translate. Unknown and tier-only types (stdBrick, zeroLength, absorbing
boundaries, embedded nodes, H5DRM, …) raise one `UnsupportedSTKOTypes` that
names them all, before the session is touched; nothing is skipped silently.
The registry is keyed by STKO's `XOBJ_META`, so a later type is one entry.

The reader now also reads the `LOCAL_AXES/LOCAX_n` datasets, which it used to
skip.

Rigid diaphragms are STKO's link pairs, verbatim: each master and its slaves
sit on their own carrier entities, the constraint keeps a slave that is off
the master's plane, and `build_conditions` checks the FEM's resolved records
against the plan before declaring anything. Each static stage starts with
STKO's IMPL-EX `dTime` reset of its target elements; after it, a time-history
driver must set `dTime` every step, as STKO's deck does. The translator also
refuses, rather than guesses: a non-square `sections.Elastic` (which `PROPS`
entry is `Iyy` is unverified), `ImplexAutoErrorControlActivate` before an
analysis, a static stage after the transient one, the properties an
interaction carries (named, e.g. `zeroLength`), and a session whose nodes are
not exactly STKO's.

`apeSees(fem, element_tags="fem")` is a new bridge option: every element
gets its FEM element id as OpenSees tag, and tags the bridge makes itself
(interface springs, couplings, node-pair specs) land above the largest FEM
element id. It refuses an id of 0 or less and an element declared by two
specs. The default, `"sequential"`, writes the same decks as before.
`MeshSelection.connectivity` and iterating a selection now follow the order
of `.ids`; before, the connectivity rows came in mesh storage order while
`.ids` did not, so the two could disagree. A selection id that is not in the
mesh now raises `KeyError` instead of being dropped.

On the four San Ramon Tier-1 documents, the deck matches STKO's own
partitioned Tcl export in every gated category. Node ids, element ids,
connectivity in node order, shell flags and `-local`, beam transforms,
sections and materials, fix and rigid-diaphragm pairs are exact. Nodal
masses and loads match per node to 1e-9 relative; time series, Rayleigh and
stages match within float tolerance. The translated 1A deck gives
T1 = 1.66477 s, against 1.66477 s for STKO's run (relative 5e-11).

The local STKO oracle tests now all use one environment variable,
`APEGMSH_STKO_ORACLES` (the reader's San Ramon tests used
`APEGMSH_STKO_SAMPLES`).

Not in this version:
- The transient stage, UniformExcitation patterns, STKO's adaptive driver
  (its test written with twice the iterations) and the per-step IMPL-EX
  `dTime` update stay as data in the plan.
- `-crackPlanes` goes through a subclass of the bridge's `ASDConcrete3D`.
