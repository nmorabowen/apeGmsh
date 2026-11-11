// The clip slice: the section planes.
// One slice of the store's types; src/state/types.ts re-exports it, and the
// panels import it from there. A slice file imports sibling files, never the barrel.

/**
 * V4e1 (#1834, ADR 0083 Part 1): one section plane. World-space: it cuts
 * whatever is drawn, so the deformed shape moves through it while scrubbing.
 * The half-space `normal` points into survives; `flipped` keeps the other.
 */
export interface ClipPlane {
  /** `plane-<n>`, unique for the window session */
  id: string;
  /** unit, world-space */
  normal: readonly [number, number, number];
  /** signed distance from the origin to the plane along `normal`; never clamped */
  offset: number;
  flipped: boolean;
  /** cutting; at most MAX_ACTIVE_PLANES (state/clip.ts) are */
  active: boolean;
  /** V4e2: its gizmo is drawn while "Show plane gizmos" is on (the panel row's eye) */
  gizmo: boolean;
}

/** V4e1: the section planes and the global "Apply cuts" toggle (window session only: the app has no session store yet). */
export interface ClipState {
  planes: readonly ClipPlane[];
  applyCuts: boolean;
  /** V4e2: the footer's "Show plane gizmos" toggle (each plane's eye still applies) */
  showGizmos: boolean;
  /**
   * V4e3: the footer's "Contour on cut face" toggle: each cutting plane fills
   * the solid it cuts with a cap carrying the contour (a flat fill without
   * one). On by default: without it a plane through a solid opens onto an
   * empty box (ADR 0083: S3 is required for solid models).
   */
  cutFace: boolean;
  /** the last plane number handed out (ids are never reused) */
  seq: number;
  /** why the last clip event was refused (the 7th active plane, a zero normal, a NaN offset); cleared by the next accepted one */
  notice: string | null;
}
