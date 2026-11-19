// 2-D polygon triangulation (hole bridging + ear clipping), shared by the
// frame cross-sections (state/frameExtrusion.ts) and the membrane panels
// (./membrane.ts). Pure: no three.js and no DOM.

import type { SectionProfile } from "../model/types.ts";

function signedArea2D(pts: readonly (readonly [number, number])[]): number {
  let a2 = 0;
  for (let i = 0; i < pts.length; i++) {
    const [y0, z0] = pts[i]!;
    const [y1, z1] = pts[(i + 1) % pts.length]!;
    a2 += y0 * z1 - y1 * z0;
  }
  return 0.5 * a2;
}

function cross2D(
  ay: number,
  az: number,
  by: number,
  bz: number,
  cy: number,
  cz: number,
): number {
  return (by - ay) * (cz - az) - (bz - az) * (cy - ay);
}

function pointInTriangle2D(
  py: number,
  pz: number,
  ay: number,
  az: number,
  by: number,
  bz: number,
  cy: number,
  cz: number,
): boolean {
  const c1 = cross2D(ay, az, by, bz, py, pz);
  const c2 = cross2D(by, bz, cy, cz, py, pz);
  const c3 = cross2D(cy, cz, ay, az, py, pz);
  return c1 >= -1e-12 && c2 >= -1e-12 && c3 >= -1e-12;
}

interface RingVertex {
  y: number;
  z: number;
  origIndex: number;
}

function pointInRing2D(py: number, pz: number, ring: readonly RingVertex[]): boolean {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const yi = ring[i]!.y, zi = ring[i]!.z;
    const yj = ring[j]!.y, zj = ring[j]!.z;
    if ((zi > pz) !== (zj > pz) && py < ((yj - yi) * (pz - zi)) / (zj - zi) + yi) {
      inside = !inside;
    }
  }
  return inside;
}

/** Bridge a CW hole loop into a CCW outer loop along a horizontal ray (+y) from the hole's rightmost vertex. */
function spliceHoleIntoOuter(outer: RingVertex[], hole: RingVertex[]): RingVertex[] {
  let mIdx = 0;
  for (let i = 1; i < hole.length; i++) {
    if (hole[i]!.y > hole[mIdx]!.y || (hole[i]!.y === hole[mIdx]!.y && hole[i]!.z < hole[mIdx]!.z)) {
      mIdx = i;
    }
  }
  const m = hole[mIdx]!;
  let bestDist = Number.POSITIVE_INFINITY;
  let bestOuterIdx = 0;

  // Find closest visible vertex on outer ring
  for (let i = 0; i < outer.length; i++) {
    const a = outer[i]!;
    const b = outer[(i + 1) % outer.length]!;
    // Check if horizontal ray from (m.y, m.z) in +y intersects segment a-b
    if ((a.z > m.z) !== (b.z > m.z)) {
      const yInt = a.y + ((m.z - a.z) * (b.y - a.y)) / (b.z - a.z);
      if (yInt >= m.y - 1e-12) {
        const d = yInt - m.y;
        if (d < bestDist) {
          bestDist = d;
          bestOuterIdx = a.y >= b.y ? i : (i + 1) % outer.length;
        }
      }
    } else if (Math.abs(a.z - m.z) <= 1e-12 && a.y >= m.y - 1e-12) {
      const d = a.y - m.y;
      if (d < bestDist) {
        bestDist = d;
        bestOuterIdx = i;
      }
    }
  }

  const out: RingVertex[] = [];
  for (let i = 0; i <= bestOuterIdx; i++) out.push(outer[i]!);
  for (let k = 0; k <= hole.length; k++) out.push(hole[(mIdx + k) % hole.length]!);
  for (let i = bestOuterIdx; i < outer.length; i++) out.push(outer[i]!);
  return out;
}

function earClipRing(ring: readonly RingVertex[]): [number, number, number][] {
  const verts = [...ring];
  const tris: [number, number, number][] = [];
  let guard = verts.length * verts.length + 16;

  while (verts.length > 3 && guard-- > 0) {
    const n = verts.length;
    let clipped = false;
    for (let i = 0; i < n; i++) {
      const prev = verts[(i - 1 + n) % n]!;
      const curr = verts[i]!;
      const next = verts[(i + 1) % n]!;
      const area2 = cross2D(prev.y, prev.z, curr.y, curr.z, next.y, next.z);
      if (area2 <= 1e-12) continue;

      let hasInside = false;
      for (let j = 0; j < n; j++) {
        if (j === (i - 1 + n) % n || j === i || j === (i + 1) % n) continue;
        const p = verts[j]!;
        if (
          Math.hypot(p.y - prev.y, p.z - prev.z) <= 1e-11 ||
          Math.hypot(p.y - curr.y, p.z - curr.z) <= 1e-11 ||
          Math.hypot(p.y - next.y, p.z - next.z) <= 1e-11
        ) {
          continue;
        }
        if (pointInTriangle2D(p.y, p.z, prev.y, prev.z, curr.y, curr.z, next.y, next.z)) {
          hasInside = true;
          break;
        }
      }
      if (!hasInside) {
        tris.push([prev.origIndex, curr.origIndex, next.origIndex]);
        verts.splice(i, 1);
        clipped = true;
        break;
      }
    }
    if (!clipped) {
      // Degenerate or near-collinear vertex: drop the most collinear vertex or clip largest positive ear
      let bestI = 0;
      let bestA = Number.NEGATIVE_INFINITY;
      for (let i = 0; i < n; i++) {
        const prev = verts[(i - 1 + n) % n]!;
        const curr = verts[i]!;
        const next = verts[(i + 1) % n]!;
        const a2 = cross2D(prev.y, prev.z, curr.y, curr.z, next.y, next.z);
        if (a2 > bestA) {
          bestA = a2;
          bestI = i;
        }
      }
      const prev = verts[(bestI - 1 + n) % n]!;
      const curr = verts[bestI]!;
      const next = verts[(bestI + 1) % n]!;
      if (bestA > 0) tris.push([prev.origIndex, curr.origIndex, next.origIndex]);
      verts.splice(bestI, 1);
    }
  }
  if (verts.length === 3) {
    const [a, b, c] = verts;
    if (cross2D(a!.y, a!.z, b!.y, b!.z, c!.y, c!.z) > 0) {
      tris.push([a!.origIndex, b!.origIndex, c!.origIndex]);
    }
  }
  return tris;
}

/** Flatten `profile.loops` into a vertex table and triangulate the 2-D cross-section. */
export function triangulateSectionProfile(profile: SectionProfile): {
  vertices: readonly [number, number][];
  loopRanges: readonly { start: number; count: number }[];
  triangles: readonly [number, number, number][];
} {
  const vertices: [number, number][] = [];
  const loopRanges: { start: number; count: number }[] = [];
  const outerRings: RingVertex[][] = [];
  const holeRings: RingVertex[][] = [];

  for (const loop of profile.loops) {
    if (loop.length < 3) continue;
    const start = vertices.length;
    const ring: RingVertex[] = [];
    for (let i = 0; i < loop.length; i++) {
      const [y, z] = loop[i]!;
      const origIndex = vertices.length;
      vertices.push([y, z]);
      ring.push({ y, z, origIndex });
    }
    loopRanges.push({ start, count: loop.length });
    const a = signedArea2D(loop);
    if (a >= 0) {
      outerRings.push(ring);
    } else {
      holeRings.push(ring);
    }
  }

  if (outerRings.length === 0 && holeRings.length > 0) {
    // If single loop had reversed CW orientation, treat it as outer
    const flipped = [...holeRings.shift()!].reverse();
    outerRings.push(flipped);
  }

  // Group holes by the outer ring that contains them
  const holesByOuter: RingVertex[][][] = outerRings.map(() => []);
  for (const h of holeRings) {
    let ownerIdx = 0;
    if (outerRings.length > 1 && h.length > 0) {
      const found = outerRings.findIndex((o) => pointInRing2D(h[0]!.y, h[0]!.z, o));
      if (found >= 0) ownerIdx = found;
    }
    holesByOuter[ownerIdx]?.push(h);
  }

  const triangles: [number, number, number][] = [];
  for (let idx = 0; idx < outerRings.length; idx++) {
    let ring = outerRings[idx]!;
    const holes = holesByOuter[idx] ?? [];
    // Sort holes by descending max y so rightmost holes are bridged first
    holes.sort((a, b) => {
      const maxYa = Math.max(...a.map((v) => v.y));
      const maxYb = Math.max(...b.map((v) => v.y));
      return maxYb - maxYa;
    });
    for (const h of holes) {
      ring = spliceHoleIntoOuter(ring, h);
    }
    triangles.push(...earClipRing(ring));
  }

  return { vertices, loopRanges, triangles };
}
