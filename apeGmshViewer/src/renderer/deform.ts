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
import { DARK, OFFICE, themeTokens, toHexNumber, type ThemeMode } from "../theme/tokens.ts";
import { highlightObjects, selectionBounds, type Focus, type Prims } from "./selection.ts";

/** Every primitive's positions (6 floats per segment, 9 per triangle, 6 per outline edge; original order), and what they show. */
export interface FullPositions {
  line: Float32Array;
  tri: Float32Array;
  /** the outline edges, moved through their own node ids (MeshInfo.vertexNodes.edge), like the faces */
  edge: Float32Array;
  /** `base` for the undeformed mesh, else `<displacement key>|scale <s>` */
  key: string;
  /** the displacement drawn (its `missing` flags mark nodes drawn undeformed); null for the base */
  disp: DisplacementData | null;
}

/** The undeformed positions, from the mesh blobs (never written to). */
export function basePositions(info: MeshInfo, blobs: BlobStore): FullPositions {
  return { line: blobs.f32(info.linePositions), tri: blobs.f32(info.triPositions), edge: blobs.f32(info.edgePositions), key: "base", disp: null };
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
  return deformFull(basePositions(info, blobs), disp, v.scale, key);
}

/**
 * The deformed positions of every segment, triangle and outline edge: each
 * vertex moved by the displacement of its own node (the effects computed it
 * per vertex through the vertex node ids), never by a node found at its
 * position, so coincident nodes move apart.
 */
export function deformFull(base: FullPositions, disp: DisplacementData, scale: number, key: string): FullPositions {
  return {
    line: deformedPositions(base.line, disp.line.d, scale),
    tri: deformedPositions(base.tri, disp.tri.d, scale),
    edge: deformedPositions(base.edge, disp.edge.d, scale),
    key,
    disp,
  };
}

/**
 * The positions everything drawn over the mesh reads: the selection outline,
 * `F` framing and the screen point of a declaration. They are the drawn ones,
 * deformed while the deformed shape shows, so the outline and the frame sit
 * where the mesh is drawn.
 */
export function drawnPositions(full: FullPositions | null, info: MeshInfo, blobs: BlobStore): FullPositions {
  return full ?? basePositions(info, blobs);
}

/** The selection outline (thick lines and fill, selection.ts `highlightObjects`) over the drawn positions. */
export function selectionOutline(
  full: FullPositions | null,
  info: MeshInfo,
  blobs: BlobStore,
  prims: readonly Prims[],
  makeLines: (positions: Float32Array, widthPx: number) => THREE.Object3D,
): THREE.Object3D[] {
  const pos = drawnPositions(full, info, blobs);
  return highlightObjects(pos.line, pos.tri, prims, makeLines);
}

/**
 * What `F` frames. With a selection: the bounds of its primitives where they
 * are drawn (the effects' `focus`, measured on the undeformed blobs, is the
 * fallback when nothing of it is drawn). With nothing selected: the whole
 * model, which is `null` (the mesh's own bounds) when undeformed, and the
 * bounds of the drawn deformed primitives while the deformed shape shows, so
 * a large scale never frames the shape out of view.
 */
export function frameFocus(
  full: FullPositions | null,
  info: MeshInfo,
  blobs: BlobStore,
  drawn: { lineMap: Int32Array; triMap: Int32Array } | null,
  selected: readonly Prims[] | null,
  fallback: Focus | null,
): Focus | null {
  const pos = drawnPositions(full, info, blobs);
  const min = 1e-3 * info.radius;
  if (selected && selected.length) {
    return selectionBounds(pos.line, pos.tri, selected.flatMap((p) => p.lines), selected.flatMap((p) => p.tris), min) ?? fallback;
  }
  if (pos.key === "base" || !drawn) return null;
  return selectionBounds(pos.line, pos.tri, Array.from(drawn.lineMap), Array.from(drawn.triMap), min);
}

/** Whether the outline edges are drawn: the user's toggle, while surfaces are shown (opacity > 0). */
export const edgesShown = (edgesOn: boolean, full: FullPositions | null, opacity = 1): boolean => edgesOn && full !== null && opacity > 0;

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
  full: Pick<FullPositions, "line" | "tri" | "edge">,
  drawn: { lineMap: Int32Array; triMap: Int32Array },
  edges: { object: THREE.LineSegments; map: Int32Array } | null = null,
): void {
  if (edges) {
    // The outline edges, 6 floats each, through the same original-index map the viewport drew them with.
    const a = edges.object.geometry.getAttribute("position") as THREE.BufferAttribute;
    gatherInto(full.edge, edges.map, 6, a.array as Float32Array);
    a.needsUpdate = true;
    edges.object.geometry.computeBoundingSphere();
  }
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
export function missingMarkerObjects(
  full: FullPositions,
  info: MeshInfo,
  blobs: BlobStore,
  drawn: { lineMap: Int32Array; triMap: Int32Array },
  mode: ThemeMode = "dark",
): THREE.Object3D[] {
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
  const ringColour = toHexNumber(themeTokens(mode).text);
  return [ringColour, MISSING_CORE].map((colour, i) => {
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
export function ghostObjects(
  info: MeshInfo,
  blobs: BlobStore,
  drawn: { lineMap: Int32Array; triMap: Int32Array },
  edgeMap: Int32Array | null,
  mode: ThemeMode = "dark",
): THREE.Object3D[] {
  // Always the mesh blobs: the ghost is the undeformed mesh, whatever is drawn.
  const base = basePositions(info, blobs);
  const edges = edgeMap ? gatherInto(base.edge, edgeMap, 6) : null;
  const t = themeTokens(mode);
  const out: THREE.Object3D[] = [];
  if (edges?.length) {
    // The undeformed outline edges, as the Qt ghost keeps its wireframe.
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(edges, 3));
    const e = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: toHexNumber(t.geometryCurve), transparent: true, opacity: GHOST_OPACITY, depthWrite: false }));
    e.name = "ghost edges";
    e.renderOrder = -1;
    out.push(e);
  }
  const tri = gatherInto(base.tri, drawn.triMap, 9);
  if (tri.length) {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(tri, 3));
    g.computeVertexNormals();
    const m = new THREE.MeshStandardMaterial({
      color: toHexNumber(t.geometrySurface), side: THREE.DoubleSide, flatShading: true,
      transparent: true, opacity: GHOST_OPACITY, depthWrite: false,
    });
    const mesh = new THREE.Mesh(g, m);
    mesh.name = "ghost faces";
    mesh.renderOrder = -1;
    out.push(mesh);
  }
  const line = gatherInto(base.line, drawn.lineMap, 6);
  if (line.length) {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(line, 3));
    const l = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: toHexNumber(t.geometryCurve), transparent: true, opacity: GHOST_OPACITY, depthWrite: false }));
    l.name = "ghost lines";
    l.renderOrder = -1;
    out.push(l);
  }
  return out;
}

/**
 * Sync the reference ghost's 2-D/3-D surfaces and outline edges with the user's
 * surface opacity (`off` hides ghost surfaces and edges) and edges toggle
 * (`edges` off hides ghost outline edges), while keeping 1-D ghost lines shown.
 */
export function applyGhostVisibility(ghost: THREE.Group, edgesOn: boolean, full: FullPositions | null, opacity = 1): void {
  const showEdges = edgesShown(edgesOn, full, opacity);
  const showFaces = opacity > 0;
  for (const o of ghost.children) {
    if (o.name === "ghost edges") o.visible = showEdges;
    else if (o.name === "ghost faces" || o instanceof THREE.Mesh) o.visible = showFaces;
  }
}
