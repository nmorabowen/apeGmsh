// The deform slice: the deformed shape.
// One slice of the store's types; src/state/types.ts re-exports it, and the
// panels import it from there. A slice file imports sibling files, never the barrel.

/** The deformed shape (V4d, #1765): the user's toggle, scale and reference ghost, and the effects' count of drawn nodes with no displacement. */
export interface DeformState {
  on: boolean;
  /** positions are `x + scale * u`; 1 is true scale (the Qt `deform_scale` default) */
  scale: number;
  /** the undeformed mesh drawn faint behind the deformed one (Qt `add_reference_ghost`) */
  ghost: boolean;
  /** the drawn nodes and those with no displacement at the step shown; `key` is selectors `deformView(...).keys.count` */
  measured: { key: string; drawn: number; missing: number } | null;
}
