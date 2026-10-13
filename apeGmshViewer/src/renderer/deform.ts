// The viewport's side of the deformed shape (V4d, #1765): the positions of
// every primitive at the step shown (state/deform.ts computes them from the
// effects' per-vertex displacement), written in place into the drawn faces
// and lines so raycasting, lighting and picking all see the deformed mesh;
// the reference ghost (the undeformed mesh, faint, Qt `add_reference_ghost`);
// and the markers on drawn nodes that have no displacement.
//
// Beams and trusses stay straight chords between their displaced end nodes
// (one segment per pair of nodes, as mesh/build.ts draws them; the Qt
// deformed geometry does the same): nothing here curves a line.

import * as THREE from "three";
import type { LineSegments2 } from "three/addons/lines/LineSegments2.js";
import type { DisplacementData } from "../effects.ts";
import type { BlobStore } from "../state/blobs.ts";
import { deformedPositions, GHOST_OPACITY, missingMarkers } from "../state/deform.ts";
import { deformView } from "../state/selectors.ts";
import type { MeshInfo, State } from "../state/types.ts";
import { DARK, OFFICE, toHexNumber } from "../theme/tokens.ts";

/** Every primitive's positions (6 floats per segment, 9 per triangle, original order), and what they show. */
export interface FullPositions {
  line: Float32Array;
  tri: Float32Array;
  /** `base` for the undeformed mesh, else `<displacement key>|scale <s>` */
  key: string;
  /** the displacement drawn (its `missing` flags mark nodes drawn undeformed); null for the base */
  disp: DisplacementData | null;
}

/** The undeformed positions, from the mesh blobs (never written to). */
export function basePositions(info: MeshInfo, blobs: BlobStore): FullPositions {
  return { line: blobs.f32(info.linePositions), tri: blobs.f32(info.triPositions), key: "base", disp: null };
}

/**
 * The positions the state asks for: deformed by the effects' displacement of
 * the step shown when it is here, the undeformed mesh when the deformed
 * shape is off or cannot be drawn, or `null` to keep what is drawn (the
 * step's displacement is still on its way: a scrub never flashes the base).
 */
export function positionsFor(s: State, info: MeshInfo, blobs: BlobStore, disp: DisplacementData | null, current: FullPositions | null): FullPositions | null {
  const v = deformView(s);
  if (v.state !== "loading" && v.state !== "ready" && v.state !== "partial") return current?.key === "base" ? current : basePositions(info, blobs);
  if (!disp || disp.key !== v.keys.disp) return current;
  const key = `${disp.key}|scale ${v.scale}`;
  if (current?.key === key) return current;
  const base = basePositions(info, blobs);
  return { line: deformedPositions(base.line, disp.line.d, v.scale), tri: deformedPositions(base.tri, disp.tri.d, v.scale), key, disp };
}

/** The drawn primitives' coordinates (`map`: original indices) copied out of `src`, `stride` floats each, into `dst` (a new array when absent). */
export function gatherInto(src: Float32Array, map: Int32Array, stride: number, dst: Float32Array = new Float32Array(map.length * stride)): Float32Array {
  if (dst.length !== map.length * stride) throw new RangeError(`gatherInto: ${dst.length} floats for ${map.length} primitives of ${stride}`);
  for (let d = 0; d < map.length; d++) {
    const o = map[d]! * stride;
    for (let k = 0; k < stride; k++) dst[d * stride + k] = src[o + k]!;
  }
  return dst;
}

/**
 * Write the drawn primitives' positions into the faces and lines in place,
 * then refresh what reads them: the face normals (lighting), and both
 * objects' bounding box and sphere, which the raycaster tests before any
 * triangle or segment (stale bounds would pick against the old shape).
 */
export function writePositions(
  faces: THREE.Mesh | null,
  lines: LineSegments2 | null,
  full: Pick<FullPositions, "line" | "tri">,
  drawn: { lineMap: Int32Array; triMap: Int32Array },
): void {
  if (faces) {
    const a = faces.geometry.getAttribute("position") as THREE.BufferAttribute;
    gatherInto(full.tri, drawn.triMap, 9, a.array as Float32Array);
    a.needsUpdate = true;
    faces.geometry.computeVertexNormals();
    faces.geometry.computeBoundingBox();
    faces.geometry.computeBoundingSphere();
  }
  if (lines) {
    // LineSegmentsGeometry keeps the positions as one interleaved buffer, 6 floats per segment (start xyz, end xyz).
    const a = lines.geometry.getAttribute("instanceStart") as THREE.InterleavedBufferAttribute;
    gatherInto(full.line, drawn.lineMap, 6, a.data.array as Float32Array);
    a.data.needsUpdate = true;
    lines.geometry.computeBoundingBox();
    lines.geometry.computeBoundingSphere();
  }
}

/** The colour of a marker's core: the contour's no-data grey (state/contour.ts `NO_DATA_RGB`), ringed in the text colour (the legend chip repeats both). */
export const MISSING_CORE = toHexNumber(OFFICE.muted);
export const MISSING_RING = toHexNumber(DARK.text);
/** Marker sizes in CSS pixels: the white ring, then the grey core over it. */
export const MISSING_PX = [13, 8] as const;

/**
 * The markers of the drawn nodes that have no displacement (each node
 * once): drawn over everything, at a fixed pixel size, where those nodes
 * sit (undeformed). Empty when every drawn node has a displacement.
 */
export function missingMarkerObjects(full: FullPositions, info: MeshInfo, blobs: BlobStore, drawn: { lineMap: Int32Array; triMap: Int32Array }): THREE.Object3D[] {
  const disp = full.disp;
  if (!disp || (disp.line.count === 0 && disp.tri.count === 0)) return [];
  const seen = new Set<number>();
  const pts = [
    ...missingMarkers(full.line, blobs.get(info.vertexNodes.line) as Float64Array, disp.line.missing, drawn.lineMap, 2, seen),
    ...missingMarkers(full.tri, blobs.get(info.vertexNodes.tri) as Float64Array, disp.tri.missing, drawn.triMap, 3, seen),
  ];
  if (!pts.length) return [];
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(pts), 3));
  return [MISSING_RING, MISSING_CORE].map((colour, i) => {
    const p = new THREE.Points(g, new THREE.PointsMaterial({ color: colour, size: MISSING_PX[i]!, sizeAttenuation: false, depthTest: false }));
    p.renderOrder = 20 + i;
    return p;
  });
}

/**
 * The reference ghost: the drawn primitives undeformed, faces and lines in
 * one neutral colour at the Qt ghost opacity, behind the deformed mesh
 * (no depth write, drawn first).
 */
export function ghostObjects(base: FullPositions, drawn: { lineMap: Int32Array; triMap: Int32Array }, edges: Float32Array | null): THREE.Object3D[] {
  const out: THREE.Object3D[] = [];
  if (edges?.length) {
    // The undeformed outline edges, as the Qt ghost keeps its wireframe.
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(edges, 3));
    const e = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: toHexNumber(DARK.geometryCurve), transparent: true, opacity: GHOST_OPACITY, depthWrite: false }));
    e.renderOrder = -1;
    out.push(e);
  }
  const tri = gatherInto(base.tri, drawn.triMap, 9);
  if (tri.length) {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(tri, 3));
    g.computeVertexNormals();
    const m = new THREE.MeshStandardMaterial({
      color: toHexNumber(DARK.geometrySurface), side: THREE.DoubleSide, flatShading: true,
      transparent: true, opacity: GHOST_OPACITY, depthWrite: false,
    });
    const mesh = new THREE.Mesh(g, m);
    mesh.renderOrder = -1;
    out.push(mesh);
  }
  const line = gatherInto(base.line, drawn.lineMap, 6);
  if (line.length) {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(line, 3));
    const l = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: toHexNumber(DARK.geometryCurve), transparent: true, opacity: GHOST_OPACITY, depthWrite: false }));
    l.renderOrder = -1;
    out.push(l);
  }
  return out;
}
