### FIXED — apeGmshViewer contours `.ladruno` node results only on the axis labels the fork schema documents (#2214)

The contour panel groups a `.ladruno` ON_NODES component `<RESULT>/<label>`
into an axis of its field only when the fork schema
(`ladruno_schema_v1.md` §7.1.1 and §7.1.2, fork ≥ d370ed8b) documents the
label. Each label set in the code cites its section:

- **Kept:** `Ux..Uz` and `Rx..Rz` (§7.1.1); `Vx..Vz`, `Ax..Az`, `RFx..RFz`,
  `Fx..Fz`, `TFx..TFz`, and the rotational `RVx..`, `RAx..`, `RMx..` and
  `Mx..` (§7.1.2).
- **Dropped:** `X`, `Y`, `Z`, which no fork writer emits. A component with
  such a label is now drawn as itself, never as an axis.
- **Magnitude:** reaction forces now get one (`REACTION_FORCE` and its
  inertia and Rayleigh variants, through `RF*`), and so do unbalanced and
  tie forces. The undocumented name `RAYLEIGH_FORCES` is replaced by the
  schema's `RAYLEIGH_FORCE`.
- **Rotations at ndm 2:** the u-p caveat now covers every rotational group,
  not only `ROTATION`. A group whose only label is its `z` label (`Rz`,
  `RVz`, `RAz`, `RMz`, `Mz`) is drawn as itself, because on a u-p node that
  column holds the pore pressure.
- **Scalars:** `p` and the sensitivity labels `d_*/d_q<g>` are never axes.

Every committed `.ladruno` contours exactly as before.
