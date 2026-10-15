### FIXED — MPCO reads fork-solid Gauss stress and element-only files instead of returning empty (#1970)

The fork's MPCO recorder wrote the data, but `Results.from_mpco(...)` returned
an empty result and no warning in three cases:

- **The bracket's integration rule is 0.** The fork writes rule 0
  (NoIntegrationRule) for every Ladruno-band class, for example
  `33018-LadrunoBrick20[0:0:0]`. No catalog row is keyed on rule 0, so the
  bucket was skipped and `elements.gauss.get("stress_xx")` came back `(1, 0)`.
  A rule-0 bucket now takes the class's one catalogued rule for the token. Its
  width and Gauss-point blocks are still checked against that row, and a class
  with more than one catalogued rule raises instead of guessing. The buckets
  this reaches are those of the catalogued Ladruno-band classes (BezierTri6,
  BezierTet10, LadrunoBrick, LadrunoQuad, LadrunoCST, LadrunoLST,
  LadrunoBrick20) and of any other catalogued class MPCO writes at rule 0.
  No bucket that read before changes, because no catalog row has rule 0.
- **The response sits under `stress`.** The ON_ELEMENTS group is named by the
  keyword the deck asked for, and only `stresses` / `material.stress` were
  searched. The catalog token (`stress`, `strain`) is now searched too.
- **LadrunoLST writes one unnamed element-level block.** Its nine `C1..C9`
  columns can only be named by the catalog, so the block is read by the
  catalog layout when its width is exactly the layout's. A block whose columns
  carry their own names (the truss `N`), or one of another width, still
  refuses with the bucket named.
- **An element-only file read 0 steps.** A recorder with only `-E` channels
  writes no node results, and the stage's steps came from those alone. The
  steps now come from the element buckets when there are no node results, with
  the same step-count cross-check.
