// The clip slice's events: its part of the store's Event union (types/event.ts).
// Not re-exported by src/state/types.ts, which exports only the whole union.

export type ClipEvent =
  /** V4e1: add a section plane through the reference mesh centre (created inactive, with a notice, when six already cut) */
  | { type: "addClipPlane"; normal: readonly number[] }
  | { type: "removeClipPlane"; id: string }
  /** V4e1: activating a 7th plane is refused with a notice (C-LIMIT) */
  | { type: "setClipActive"; id: string; active: boolean }
  | { type: "setClipFlipped"; id: string; flipped: boolean }
  /** V4e1: unclamped; a non-finite offset is refused with a notice */
  | { type: "setClipOffset"; id: string; offset: number }
  /** V4e1: normalised on entry; a zero or non-finite normal is refused with a notice */
  | { type: "setClipNormal"; id: string; normal: readonly number[] }
  /** V4e1: the global "Apply cuts" toggle */
  | { type: "setApplyCuts"; on: boolean }
  /**
   * V4e2: the gizmo's rotate, one event per move: the plane `normal · p = offset`
   * (scaled together to a unit normal). A zero or non-finite normal, or a
   * non-finite offset, is refused with a notice and the pose is kept.
   */
  | { type: "setClipPose"; id: string; normal: readonly number[]; offset: number }
  /** V4e2: the footer's "Show plane gizmos" toggle */
  | { type: "setShowGizmos"; on: boolean }
  /** V4e2: one plane's gizmo eye */
  | { type: "setClipGizmo"; id: string; visible: boolean }
  /** V4e3: the footer's "Contour on cut face" toggle (a cap on every cutting plane through a solid) */
  | { type: "setCutFace"; on: boolean };
