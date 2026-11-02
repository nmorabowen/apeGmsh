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
  clearLine?: Float32Array;
  frameTri?: Float32Array;
  frameEdge?: Float32Array;
  offsetLine?: Float32Array;
  offsetJoint?: Float32Array;
  /** `base` for the undeformed mesh, else `<displacement key>|scale <s>` */
  key: string;
  /** the displacement drawn (its `missing` flags mark nodes drawn undeformed); null for the base */
  disp: DisplacementData | null;
}

/** The undeformed positions, from the mesh blobs (never written to). */
export function basePositions(info: MeshInfo, blobs: BlobStore): FullPositions {
  const fe = info.frameExtrusion;
  return {
    line: blobs.f32(info.linePositions),
    tri: blobs.f32(info.triPositions),
    edge: blobs.f32(info.edgePositions),
    ...(fe
      ? {
          clearLine: blobs.f32(fe.clearLinePositions),
          frameTri: blobs.f32(fe.triPositions),
          frameEdge: blobs.f32(fe.edgePositions),
          offsetLine: blobs.f32(fe.offsetLinePositions),
          offsetJoint: blobs.f32(fe.offsetJointPositions),
        }
      : {}),
    key: "base",
    disp: null,
  };
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
    ...(base.clearLine ? { clearLine: deformedPositions(base.clearLine, disp.line.d, scale) } : {}),
    ...(base.frameTri && disp.frameTri ? { frameTri: deformedPositions(base.frameTri, disp.frameTri.d, scale) } : {}),
    ...(base.frameEdge && disp.frameEdge ? { frameEdge: deformedPositions(base.frameEdge, disp.frameEdge.d, scale) } : {}),
    ...(base.offsetLine && disp.offsetLine ? { offsetLine: deformedPositions(base.offsetLine, disp.offsetLine.d, scale) } : {}),
    ...(base.offsetJoint && disp.offsetJoint ? { offsetJoint: deformedPositions(base.offsetJoint, disp.offsetJoint.d, scale) } : {}),
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
  showOffsets = false,
): THREE.Object3D[] {
  const pos = drawnPositions(full, info, blobs);
  const linePos = showOffsets && pos.clearLine ? pos.clearLine : pos.line;
  return highlightObjects(linePos, pos.tri, prims, makeLines);
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

/** The camera layer of the ghost's surfaces (selection.ts HALO_LAYER is another); the main pass never draws them. */
const GHOST_LAYER = 2;
/** The ghost target's size over the drawing buffer's: with no MSAA there, this is its anti-aliasing. */
const GHOST_SUPERSAMPLE = 2;

/** Where the composite draws in the transparent pass: over the opaque mesh's blended edges (renderOrder 0), or ahead of every blended part of a translucent mesh. */
const COMPOSITE_OVER = 0.5;
const COMPOSITE_UNDER = -2;

/**
 * Whether the scene shows a translucent surface, the case where the ghost
 * composites under the mesh (owner rulings on #2664): a visible mesh of
 * triangles, not the ghost's own, whose material blends at an opacity below 1.
 * That is the user's surface opacity below 1, and also the 3-D frames the line
 * diagram draws at 0.25 under an opacity of 1. Fat lines and outline edges are
 * not surfaces, and neither is what draws after both triangles (renderOrder from
 * COMPOSITE_OVER up: the plane gizmos and the selection fill).
 */
export function translucentSurfaces(scene: THREE.Object3D): boolean {
  let found = false;
  scene.traverseVisible((o) => {
    // Only what draws before the over triangle: the plane gizmos (renderOrder 1) and the selection fill (9) draw after both, as on main.
    if (found || o.type !== "Mesh" || o.renderOrder >= COMPOSITE_OVER || o.layers.isEnabled(GHOST_LAYER) || o.name.startsWith("ghost composite")) return;
    const mat = (o as THREE.Mesh).material as THREE.Material | THREE.Material[];
    found = (Array.isArray(mat) ? mat : [mat]).some((x) => x.visible && x.transparent && x.opacity < 1);
  });
  return found;
}

/** The ghost's surfaces write sRGB themselves: three writes linear into a render target, and the blending must stay in the space the canvas blends in. */
function encodeSrgb(m: THREE.Material): void {
  m.onBeforeCompile = (p) => {
    p.fragmentShader = p.fragmentShader.replace("#include <colorspace_fragment>", "gl_FragColor = sRGBTransferOETF( gl_FragColor );");
  };
}

/**
 * Pixel-stable (#2664): a blended triangle in the 4x MSAA canvas gives other
 * pixels from frame to frame with nothing changed (D3D11 on an RTX 3070: up to
 * 1200 px and 8 levels against depth already there, 1 or 2 levels in about one
 * frame in 40 against none). With no MSAA it does not. So the ghost's surfaces are
 * drawn into their own target, with no MSAA, tested against the depth of the
 * opaque scene drawn there first, and this full-screen triangle lays the result
 * over the canvas with the same blend, covering every sample of every pixel. It
 * goes after the mesh's own blended outline edges (renderOrder 0), which differed
 * in about 1 frame in 100 when they blended over it (5 levels; 0 of 12,000 frames
 * this way), and before the gizmos and the selection (renderOrder 1 and up).
 *
 * Over translucent surfaces (`translucentSurfaces`) it goes first, under them
 * (owner rulings on #2664: that look stays). The pass sorts before any object
 * draws, so there are two triangles, one at each place, and each frame the first
 * one (which draws the target) decides which of the two draws; the other draws
 * nothing (an empty draw range). No pixel identity is claimed there:
 * the mesh's own blended triangles are in the canvas, and blended triangles there
 * differ from frame to frame, so a view spec refuses a ghost shot over them
 * (spec/verdict.ts). The app-chrome capture race is apart (#2696).
 */
function ghostComposite(): THREE.Mesh[] {
  const target = new THREE.WebGLRenderTarget(1, 1, { samples: 0, depthBuffer: true, minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter });
  const m = new THREE.ShaderMaterial({
    uniforms: { ghost: { value: target.texture } },
    vertexShader: "varying vec2 vUv; void main() { vUv = 0.5 * position.xy + 0.5; gl_Position = vec4(position.xy, 0.0, 1.0); }",
    fragmentShader: "uniform sampler2D ghost; varying vec2 vUv; void main() { gl_FragColor = texture2D(ghost, vUv); }",
    transparent: true,
    depthTest: false,
    depthWrite: false,
    toneMapped: false,
    // The target holds what blending over nothing leaves: colour already multiplied by alpha.
    blending: THREE.CustomBlending,
    blendSrc: THREE.OneFactor,
    blendDst: THREE.OneMinusSrcAlphaFactor,
    blendSrcAlpha: THREE.OneFactor,
    blendDstAlpha: THREE.OneMinusSrcAlphaFactor,
  });
  m.addEventListener("dispose", () => target.dispose());
  const triangle = (name: string, renderOrder: number): THREE.Mesh => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(new Float32Array([-1, -1, 0, 3, -1, 0, -1, 3, 0]), 3));
    const t = new THREE.Mesh(g, m);
    t.name = name;
    t.renderOrder = renderOrder;
    t.frustumCulled = false;
    t.userData["clipExempt"] = true;
    return t;
  };
  const quad = triangle("ghost composite under", COMPOSITE_UNDER);
  const over = triangle("ghost composite", COMPOSITE_OVER);
  let under = false;
  over.onBeforeRender = () => over.geometry.setDrawRange(0, under ? 0 : Infinity);
  const size = new THREE.Vector2();
  const clearColor = new THREE.Color();
  quad.onBeforeRender = (renderer, scene, camera) => {
    under = translucentSurfaces(scene);
    quad.geometry.setDrawRange(0, under ? Infinity : 0);
    renderer.getDrawingBufferSize(size);
    target.setSize(size.x * GHOST_SUPERSAMPLE, size.y * GHOST_SUPERSAMPLE);
    const back = renderer.getRenderTarget(), autoClear = renderer.autoClear, mask = camera.layers.mask, clearAlpha = renderer.getClearAlpha();
    renderer.getClearColor(clearColor);
    const buffers = renderer.state.buffers;
    renderer.setRenderTarget(target);
    renderer.setClearColor(0, 0);
    buffers.color.setMask(true);
    buffers.depth.setMask(true);
    renderer.clear();
    renderer.autoClear = false;
    // The depth of what is opaque: the ghost was never tested against what blends after it. The lights join the ghost's layer on the way.
    const blended: THREE.Object3D[] = [quad, over];
    scene.traverseVisible((o) => {
      if ((o as THREE.Light).isLight) o.layers.enable(GHOST_LAYER);
      const mat = (o as THREE.Mesh).material as THREE.Material | THREE.Material[] | undefined;
      if (mat && (Array.isArray(mat) ? mat : [mat]).some((x) => x.transparent)) blended.push(o);
    });
    for (const o of blended) o.visible = false;
    buffers.color.setMask(false);
    buffers.color.setLocked(true);
    renderer.render(scene, camera);
    buffers.color.setLocked(false);
    buffers.color.setMask(true);
    for (const o of blended) o.visible = true;
    camera.layers.set(GHOST_LAYER);
    renderer.render(scene, camera);
    camera.layers.mask = mask;
    renderer.autoClear = autoClear;
    renderer.setClearColor(clearColor, clearAlpha);
    renderer.setRenderTarget(back);
  };
  return [quad, over];
}

/**
 * The reference ghost: the drawn primitives undeformed, faces and lines in
 * one neutral colour at the Qt ghost opacity (no depth write). The surfaces go
 * through `ghostComposite`, over what blends in the deformed mesh; 1-D lines
 * blend in the canvas ahead of it, where lines draw the same pixels every time.
 */
export function ghostObjects(
  info: MeshInfo,
  blobs: BlobStore,
  drawn: { lineMap: Int32Array; triMap: Int32Array },
  edgeMap: Int32Array | null,
  mode: ThemeMode = "dark",
  frameTriMap: Int32Array | null = null,
  frameEdgeMap: Int32Array | null = null,
  drawnLineMap: Int32Array | null = null,
  showOffsets = false,
): THREE.Object3D[] {
  // Always the mesh blobs: the ghost is the undeformed mesh, whatever is drawn.
  const base = basePositions(info, blobs);
  const surfEdges = edgeMap ? gatherInto(base.edge, edgeMap, 6) : null;
  const fEdges = frameEdgeMap && base.frameEdge ? gatherInto(base.frameEdge, frameEdgeMap, 6) : null;
  const edges =
    surfEdges && fEdges
      ? (() => {
          const out = new Float32Array(surfEdges.length + fEdges.length);
          out.set(surfEdges, 0);
          out.set(fEdges, surfEdges.length);
          return out;
        })()
      : (surfEdges ?? fEdges);
  const t = themeTokens(mode);
  const out: THREE.Object3D[] = [];
  if (edges?.length) {
    // The undeformed outline edges, as the Qt ghost keeps its wireframe.
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(edges, 3));
    // One target pixel wide is 1 / GHOST_SUPERSAMPLE of a canvas pixel, so the opacity makes up what the width lost.
    const e = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: toHexNumber(t.geometryCurve), transparent: true, opacity: GHOST_OPACITY * GHOST_SUPERSAMPLE, depthWrite: false }));
    encodeSrgb(e.material);
    e.name = "ghost edges";
    e.renderOrder = -1;
    e.layers.set(GHOST_LAYER);
    out.push(e);
  }
  const surfTri = gatherInto(base.tri, drawn.triMap, 9);
  const fTri = frameTriMap && base.frameTri ? gatherInto(base.frameTri, frameTriMap, 9) : null;
  const tri =
    surfTri.length && fTri?.length
      ? (() => {
          const out = new Float32Array(surfTri.length + fTri.length);
          out.set(surfTri, 0);
          out.set(fTri, surfTri.length);
          return out;
        })()
      : surfTri.length
        ? surfTri
        : (fTri ?? surfTri);
  if (tri.length) {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(tri, 3));
    g.computeVertexNormals();
    const m = new THREE.MeshStandardMaterial({
      color: toHexNumber(t.geometrySurface), side: THREE.DoubleSide, flatShading: true,
      transparent: true, opacity: GHOST_OPACITY, depthWrite: false,
    });
    encodeSrgb(m);
    const mesh = new THREE.Mesh(g, m);
    mesh.name = "ghost faces";
    mesh.renderOrder = -1;
    mesh.layers.set(GHOST_LAYER);
    out.push(mesh);
  }
  if (out.length) out.push(...ghostComposite());
  const lineSrc = showOffsets && base.clearLine ? base.clearLine : base.line;
  const line = gatherInto(lineSrc, drawnLineMap ?? drawn.lineMap, 6);
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
    else if (o.name.startsWith("ghost composite")) o.visible = showFaces;
    else if (o.name === "ghost faces" || o instanceof THREE.Mesh) o.visible = showFaces;
  }
}
