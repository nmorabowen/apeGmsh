// The deform slice's events: its part of the store's Event union (types/event.ts).
// Not re-exported by src/state/types.ts, which exports only the whole union.

export type DeformEvent =
  /** V4d: draw the mesh deformed by the nodal displacement of the results step shown, or not */
  | { type: "setDeform"; on: boolean }
  /** V4d: the deformation scale (finite, above 0; 1 is true scale) */
  | { type: "setDeformScale"; scale: number }
  /** V4d: the undeformed reference ghost behind the deformed mesh */
  | { type: "setReferenceGhost"; on: boolean }
  /** V4d (effects): how many drawn nodes the step shown leaves with no displacement, for the key it was counted on */
  | { type: "deformMeasured"; key: string; drawn: number; missing: number };
