// From the /geometry zone (src/reader/geometry.ts) to the `fileLoaded`
// payload of the geometry artifact: render blobs for the curves (segments),
// the surfaces (triangles in world coordinates) and the points, and the
// artifact record. The pairing with the model is a selector
// (selectors.geometryPairing), not decided here: either file can load first.

import type { GeometryZone } from "../reader/geometry.ts";
import type { BlobStore } from "./blobs.ts";
import type { ArtifactInfo, GeometryLoad } from "./types.ts";

/** Polylines (CSR rows) as segments: 6 floats per segment. */
function segments(offsets: Int32Array, vertices: Float64Array): Float32Array {
  let n = 0;
  for (let r = 0; r + 1 < offsets.length; r++) n += Math.max(0, offsets[r + 1]! - offsets[r]! - 1);
  const out = new Float32Array(n * 6);
  let k = 0;
  for (let r = 0; r + 1 < offsets.length; r++) {
    for (let v = offsets[r]! + 1; v < offsets[r + 1]!; v++) {
      for (const i of [v - 1, v]) for (let c = 0; c < 3; c++) out[k++] = vertices[3 * i + c]!;
    }
  }
  return out;
}

/** Each surface's triangles in world coordinates: 9 floats per triangle. */
function triangles(z: GeometryZone["surfaces"]): Float32Array {
  const out = new Float32Array(z.triangles.length * 3);
  let k = 0;
  for (let s = 0; s + 1 < z.offsets.length; s++) {
    const base = z.offsets[s]!;
    for (let t = z.triangleOffsets[s]! * 3; t < z.triangleOffsets[s + 1]! * 3; t++) {
      const v = base + z.triangles[t]!;
      for (let c = 0; c < 3; c++) out[k++] = z.vertices[3 * v + c]!;
    }
  }
  return out;
}

export function loadGeometry(z: GeometryZone, blobs: BlobStore, sizeBytes: number, readMs: number): GeometryLoad {
  const curves = segments(z.curves.offsets, z.curves.vertices);
  const tris = triangles(z.surfaces);
  const points = Float32Array.from(z.points.xyz);
  const [x0, y0, z0, x1, y1, z1] = z.bbox as [number, number, number, number, number, number];
  const info: ArtifactInfo = {
    path: z.path,
    status: "ready",
    error: null,
    stale: false,
    sizeBytes,
    readMs,
    zones: { geometry: { status: "ready", version: z.version } },
    warnings: [...z.warnings],
    counts: { nodes: 0, cells: 0, opsOnly: 0 },
    sessionId: z.sessionId,
  };
  return {
    info,
    geometry: {
      path: z.path,
      sessionId: z.sessionId,
      source: z.source,
      status: z.status,
      curvePositions: blobs.put("geometry/curves", curves, [curves.length / 6, 6]),
      surfacePositions: blobs.put("geometry/surfaces", tris, [tris.length / 9, 9]),
      pointPositions: blobs.put("geometry/points", points, [points.length / 3, 3]),
      counts: {
        points: z.points.entity.length,
        curves: z.curves.entity.length,
        surfaces: z.surfaces.entity.length,
        volumes: z.volumes.entity.length,
      },
      center: [(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2],
      radius: Math.max(1e-12, Math.hypot(x1 - x0, y1 - y0, z1 - z0) / 2),
    },
  };
}
