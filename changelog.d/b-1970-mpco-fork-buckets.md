### FIXED — MPCO reads fork-recorded Gauss stress, material state and element-only files instead of returning empty (#1970)

The fork's MPCO recorder wrote the data, but `Results.from_mpco(...)` returned
an empty result and no warning in these cases:

- **The bracket's integration rule is 0.** The fork writes rule 0
  (NoIntegrationRule) for its own classes and for the stock SixNodeTri and
  EightNodeQuad, for example `33018-LadrunoBrick20[0:0:0]`. No catalog row is
  keyed on rule 0, so the bucket was skipped and
  `elements.gauss.get("stress_xx")` came back `(1, 0)`. A rule-0 bucket now
  takes the class's one catalogued rule for the token. Its width and
  Gauss-point blocks are still checked against that row, and a class with more
  than one catalogued rule raises instead of guessing. This covers BezierTri6,
  BezierTet10, LadrunoBrick, LadrunoQuad, LadrunoCST, LadrunoLST,
  LadrunoBrick20, SixNodeTri and EightNodeQuad. No bucket that read before
  changes, because no catalog row has rule 0.
- **Material state on those classes.** `material.damage` and
  `material.equivalentPlasticStrain` buckets (LadrunoBrick with
  ASDConcrete3D) take their Gauss points from the same lookup, so they read
  instead of coming back empty.
- **A rule-0 bucket the row does not fit refuses loudly.** For example,
  LadrunoBrick20 `formulation="uri"` writes 8 Gauss points against the 27-point
  row. The error says the layout was resolved from rule 0, and names the
  remedy: the `.ladruno` recorder, which stores each element's own Gauss
  points and reads that run correctly.
- **The response sits under `stress`.** The ON_ELEMENTS group is named by the
  keyword the deck asked for, and only `stresses` / `material.stress` were
  searched. The catalog token (`stress`, `strain`) is now searched too.
- **Untagged elements write one unnamed element-level block.** LadrunoQuad,
  LadrunoCST and LadrunoLST write `C1..Cn` columns that only the catalog can
  name. The block is read by the catalog layout when its width is exactly the
  layout's. A block whose columns carry their own names (the truss `N`), or one
  of another width, still refuses with the bucket named.
- **An element-only file read 0 steps.** A recorder with only `-E` channels
  writes no node results, and the stage's steps came from those alone. The
  steps now come from the element buckets when there are no node results, with
  the same step-count cross-check.
