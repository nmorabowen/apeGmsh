### FIXED — a `geomTransf` or beam element under a non-frame `ops.model` ndf raises at emit, and `stamp_provenance` verifies a file whose only unpaired elements are bridge-minted and carry no results

Both came from the YOO Guayaquil solid mat: 95 780 `LadrunoBrick` hexes,
`elasticBeamColumn` piles and eleven `LadrunoDistributingCoupling` RBE3s,
built with `apeSees(fem, element_tags="fem")`.

**Transform envelope.** The model was declared
`ops.model(ndm=3, ndf=3)` with `geomTransf.Linear` piles. The bridge
emitted the deck, and OpenSees refused its first `geomTransf` line on the
cluster, at run time ("current NDM and NDF is incompatible with frame
elements"). The transform parsers (`OpenSeesCrdTransfCommands.cpp` for
openseespy, `TclGeomTransfCommand.cpp` for Tcl), upstream and the Ladruno
fork alike, check the **builder's** ndf, never a node's `-ndf`. So do the
beam element parsers, each on its own. On both interpreters:

- `Linear` and `PDelta` accept 3 at `ndm=2` and 6 at `ndm=3`;
- `Corotational` also accepts its warping form 4 at `ndm=2`;
- `elasticBeamColumn`, `forceBeamColumn`, `dispBeamColumn`,
  `ElasticTimoshenkoBeam`, `LadrunoDispBeamColumn`,
  `LadrunoForceBeamColumn` and `LadrunoIMKBeam` accept only 3 at `ndm=2`
  and 6 at `ndm=3`.

`BuiltModel.emit` now raises `BridgeError` before writing anything when a
declared transform or beam element does not accept the envelope. The
message names the fix, for example `ops.model(ndm=3, ndf=6)`. The solid
nodes still get `-ndf 3` from inference (ADR 0048). The bridge does not
pick the envelope itself: per ADR 0048's shipped variant, the envelope is
also the ndf of every node inference cannot see, so picking it silently
would change those nodes. Nor does it bracket the beams under a frame
builder: on Tcl, a bracket purges the `geomTransf`, `beamIntegration` and
`damping` they reference (ADR 0099).

The ADR 0093 interface tests that declared `ops.model(ndm=2, ndf=2)` with
an `elasticBeamColumn` emitted decks neither interpreter could run; they
now declare `ndf=3`, and the quad master nodes keep their inferred
`-ndf 2`. Four compose/carry tests that build a phantom interface through
`Assembly` cannot follow: `Assembly` does not rehydrate a 2-D continuum,
so their master nodes are element-less and INV-1 refuses them at `ndf=3`.
They are `xfail(strict=True)` on #2830.

**`element_index_space`.** The stamp left the attribute out because the
eleven bridge-minted RBE3s (`fem_eid` -1) sat in `MODEL/ELEMENTS`, so
apeGmshViewer read every element result of the file as unverified. The
attribute speaks for the result rows (h5-schema.md,
§`element_index_space`). It is now `"fem"` when all of these hold:

- every `ON_ELEMENTS` bucket `ID` pairs with itself in the archive;
- every other `MODEL/ELEMENTS` id either pairs with itself, or is one the
  archive records as bridge-minted and that no FEM element carries;
- at least one element is FEM-paired.

A minted element with result rows, an id unknown to the archive, and a
minted id equal to a FEM id still leave the attribute out. A copy of
`T2_solid.ladruno` re-stamped this way reads as verified in apeGmshViewer.
`Results.from_ladruno` does not gate on the attribute.
