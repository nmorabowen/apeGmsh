/**
 * Tier-3 stand-in cross-sections: a section known only by its stiffness
 * properties. A leaf module, so both the reader and the frame-extrusion state
 * can use it without an import cycle.
 */

import type { SectionProfile } from "../model/types.ts";

function isPositiveFinite(v: unknown): v is number {
  return typeof v === "number" && Number.isFinite(v) && v > 0;
}

/**
 * Tier 3: a solid rectangle standing in for a section known only by its
 * `(A, Iz[, Iy])`. It keeps the area `A`; in 2-D it also keeps `Iz`, in 3-D the
 * aspect ratio `h/b = sqrt(Iz/Iy)`. Both are exact for a true rectangle. It is
 * never flagged axisymmetric: a square is not rotation-invariant, so it needs
 * a resolved local frame like any other section.
 */
export function equivalentRectProfile(A: unknown, Iz: unknown, Iy?: unknown): SectionProfile | null {
  if (!isPositiveFinite(A) || !isPositiveFinite(Iz)) return null;
  let hy: number;
  if (Iy === undefined) {
    hy = Math.sqrt((12 * Iz) / A);
  } else {
    if (!isPositiveFinite(Iy)) return null;
    hy = Math.sqrt(A * Math.sqrt(Iz / Iy));
  }
  const bz = A / hy;
  if (!isPositiveFinite(hy) || !isPositiveFinite(bz)) return null;
  return {
    loops: [
      [
        [-hy / 2, -bz / 2],
        [hy / 2, -bz / 2],
        [hy / 2, bz / 2],
        [-hy / 2, bz / 2],
      ],
    ],
    source: "elastic",
  };
}
