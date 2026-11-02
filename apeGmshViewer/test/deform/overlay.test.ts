// V4d review round 1 (#1820): what is drawn over a deformed mesh reads the
// drawn positions, and the markers mark drawn nodes only. Each test here
// fails when its guard is reverted (the PR lists the mutations):
//
// - `missingMarkers` marks only the vertices of the drawn primitives (a
//   hidden group's or an inactive element's nodes are not marked);
// - the selection outline sits on the deformed positions;
// - `F` frames the selection where it is drawn, and with nothing selected
//   the deformed model's bounds while the deformed shape shows (the mesh's
//   own bounds when it is off);
// - the reference ghost is the undeformed mesh, whatever is drawn.

import assert from "node:assert/strict";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import * as THREE from "three";
import { applyGhostVisibility, translucentSurfaces, basePositions, frameFocus, gatherInto, ghostObjects, missingMarkerObjects, positionsFor, selectionOutline, type FullPositions } from "../../src/renderer/deform.ts";
import { selectionBounds, type Prims } from "../../src/renderer/selection.ts";
import { elementMaskOf, visibleMaps } from "../../src/renderer/visible.ts";
import { missingMarkers } from "../../src/state/deform.ts";
import { deformView } from "../../src/state/selectors.ts";
import type { State } from "../../src/state/types.ts";
import { NODATA, NODATA_RESULTS, opened, until } from "../contour/harness.ts";
import { openModel } from "../../src/reader/node.ts";

const DEFORM_2D = fileURLToPath(new URL("../../fixtures/deform_2d.h5", import.meta.url));
const DEFORM_2D_RESULTS = fileURLToPath(new URL("../../fixtures/deform_2d.results.h5", import.meta.url));

type H = Awaited<ReturnType<typeof opened>>;

async function deformed(h: H, stage: number, k: number, scale: number): Promise<{ s: State; full: FullPositions }> {
  h.store.dispatch({ type: "setPhase", at: { kind: "results", stage, step: 0 } });
  if (k) h.store.dispatch({ type: "setResultStep", step: k });
  h.store.dispatch({ type: "setDeform", on: true });
  h.store.dispatch({ type: "setDeformScale", scale });
  await until(() => {
    const v = deformView(h.store.get());
    return (v.state === "ready" || v.state === "partial") && h.effects.currentDisplacement()?.key === v.keys.disp;
  }, "the deformed shape");
  const s = h.store.get();
  return { s, full: positionsFor(s, s.mesh!, h.blobs, h.effects.currentDisplacement(), null)! };
}

/** The drawn maps the viewport would use (renderer/visible.ts). */
const drawnOf = (h: H, s: State) =>
  visibleMaps(s.mesh!.legend, s.visibility.hidden, h.blobs.i32(s.mesh!.lineGroup), h.blobs.i32(s.mesh!.triGroup), elementMaskOf(s, s.mesh!, h.blobs));

/** Node ids of the marker positions (3 floats each), read back through the model's coordinates. */
async function markerNodes(path: string, pts: Float32Array): Promise<Set<number>> {
  const m = await openModel(path);
  const out = new Set<number>();
  for (let i = 0; i < pts.length; i += 3) {
    const j = [...m.nodeIds.keys()].find((n) => [0, 1, 2].every((a) => Math.fround(m.nodeCoords[3 * n + a]!) === pts[i + a]));
    assert.ok(j !== undefined, `marker ${i / 3} sits on a node`);
    out.add(m.nodeIds[j]!);
  }
  return out;
}

test("missingMarkers: only the drawn primitives' vertices are marked, each node once", () => {
  // Two triangles; every vertex missing; only the first is drawn.
  const nodes = Float64Array.from([1, 2, 3, 4, 5, 6]);
  const base = Float32Array.from([0, 0, 0, 1, 0, 0, 0, 1, 0, 5, 5, 5, 6, 5, 5, 5, 6, 5]);
  const missing = Uint8Array.from([1, 1, 1, 1, 1, 1]);
  assert.deepEqual(missingMarkers(base, nodes, missing, Int32Array.from([0]), 3), [0, 0, 0, 1, 0, 0, 0, 1, 0]);
  assert.deepEqual(missingMarkers(base, nodes, missing, Int32Array.from([1]), 3), [5, 5, 5, 6, 5, 5, 5, 6, 5]);
  assert.deepEqual(missingMarkers(base, nodes, missing, new Int32Array(0), 3), []);
});

test("markers in the app: Top's nodes are marked when Top is drawn, never when it is hidden or not yet active", async () => {
  const h = await opened(NODATA, NODATA_RESULTS);
  const pts = (s: State, full: FullPositions) => {
    const objs = missingMarkerObjects(full, s.mesh!, h.blobs, drawnOf(h, s));
    return objs.length ? ((objs[0] as THREE.Points).geometry.getAttribute("position").array as Float32Array) : new Float32Array(0);
  };
  // Stage erect: Top is active and drawn; its own nodes (z > 3) have no displacement.
  const erect = await deformed(h, 1, 1, 10);
  const v = deformView(erect.s);
  assert.equal(v.state, "partial");
  const marked = await markerNodes(NODATA, pts(erect.s, erect.full));
  assert.equal(marked.size, v.state === "partial" ? v.missing : -1, "one marker per drawn node with no displacement");
  // Top hidden: none of its nodes is drawn, so none is marked.
  const top = Object.keys(erect.s.decls).find((p) => p.endsWith("/physical_group/Top"))!;
  h.store.dispatch({ type: "setHidden", decls: [top], hidden: true });
  assert.equal(pts(h.store.get(), erect.full).length, 0, "a hidden group's nodes are not marked");
  h.store.dispatch({ type: "setHidden", decls: [top], hidden: false });
  // Stage base: Top is not activated yet (stage activation), so its nodes are not marked.
  const base = await deformed(h, 0, 2, 10);
  assert.equal(pts(base.s, base.full).length, 0, "an inactive element's nodes are not marked");
  h.effects.dispose();
});

/** The prims of the drawn element at index `e` of `mesh.elements`. */
function primsOf(h: H, s: State, e: number): Prims {
  const lines: number[] = [], tris: number[] = [];
  h.blobs.i32(s.mesh!.lineElement).forEach((x, i) => x === e && lines.push(i));
  h.blobs.i32(s.mesh!.triElement).forEach((x, i) => x === e && tris.push(i));
  return { lines, tris };
}

test("selection outline: drawn on the deformed positions while the deformed shape shows, on the mesh otherwise", async () => {
  const h = await opened(DEFORM_2D, DEFORM_2D_RESULTS);
  const { s, full } = await deformed(h, 0, 2, 30);
  const e = h.blobs.i32(s.mesh!.triElement)[0]!;
  const p = primsOf(h, s, e);
  assert.ok(p.tris.length > 0);
  const lines = (pos: Float32Array) => new THREE.LineSegments(new THREE.BufferGeometry().setAttribute("position", new THREE.BufferAttribute(pos, 3)));
  const fillOf = (objs: THREE.Object3D[]) => (objs.find((o) => o instanceof THREE.Mesh) as THREE.Mesh).geometry.getAttribute("position").array as Float32Array;
  const want = (src: Float32Array) => Float32Array.from(p.tris.flatMap((t) => Array.from(src.subarray(9 * t, 9 * t + 9))));
  assert.deepEqual(fillOf(selectionOutline(full, s.mesh!, h.blobs, [p], lines)), want(full.tri), "the fill sits on the deformed triangles");
  const base = basePositions(s.mesh!, h.blobs);
  assert.notDeepEqual(want(full.tri), want(base.tri), "the test can tell the two shapes apart");
  assert.deepEqual(fillOf(selectionOutline(null, s.mesh!, h.blobs, [p], lines)), want(base.tri), "undeformed: the mesh blobs");
  h.effects.dispose();
});

test("F framing: the selection where it is drawn, and with nothing selected the deformed model's bounds (the mesh's own when off)", async () => {
  const h = await opened(DEFORM_2D, DEFORM_2D_RESULTS);
  const { s, full } = await deformed(h, 0, 2, 30);
  const info = s.mesh!;
  const drawn = drawnOf(h, s);
  const min = 1e-3 * info.radius;
  // Nothing selected, deformed: the bounds of every drawn deformed primitive, not the mesh's.
  const whole = frameFocus(full, info, h.blobs, drawn, [], null)!;
  assert.deepEqual(whole, selectionBounds(full.line, full.tri, Array.from(drawn.lineMap), Array.from(drawn.triMap), min));
  assert.notDeepEqual([...whole.center, whole.radius], [...info.center, info.radius], "the deformed shape has other bounds");
  // Every drawn deformed vertex lies inside the framed sphere.
  for (const a of [full.tri, full.line]) {
    for (let i = 0; i < a.length; i += 3) assert.ok(Math.hypot(a[i]! - whole.center[0], a[i + 1]! - whole.center[1], a[i + 2]! - whole.center[2]) <= whole.radius * (1 + 1e-6));
  }
  // Nothing selected, undeformed: null, which frames the mesh's own bounds.
  assert.equal(frameFocus(null, info, h.blobs, drawn, [], null), null);
  assert.equal(frameFocus(basePositions(info, h.blobs), info, h.blobs, drawn, [], null), null);
  // A selection: its deformed primitives' bounds, not the effects' undeformed focus.
  const p = primsOf(h, s, h.blobs.i32(info.triElement)[0]!);
  const effectsFocus = selectionBounds(basePositions(info, h.blobs).line, basePositions(info, h.blobs).tri, p.lines, p.tris, min);
  const sel = frameFocus(full, info, h.blobs, drawn, [p], effectsFocus)!;
  assert.deepEqual(sel, selectionBounds(full.line, full.tri, p.lines, p.tris, min));
  assert.notDeepEqual(sel, effectsFocus);
  h.effects.dispose();
});

test("reference ghost: the undeformed mesh of what is drawn, and applyGhostVisibility respects surface opacity and edges toggle", async () => {
  const h = await opened(DEFORM_2D, DEFORM_2D_RESULTS);
  const { s, full } = await deformed(h, 0, 2, 30);
  const info = s.mesh!;
  const drawn = drawnOf(h, s);
  const edgeMap = Int32Array.from({ length: info.edgePositions.shape[0]! }, (_, i) => i);
  const base = basePositions(info, h.blobs);
  const objs = ghostObjects(info, h.blobs, drawn, edgeMap);
  const pos = (o: THREE.Object3D) => (o as THREE.Mesh).geometry.getAttribute("position").array as Float32Array;
  const faces = objs.find((o) => o instanceof THREE.Mesh)!;
  assert.equal(faces.name, "ghost faces");
  assert.deepEqual(pos(faces), gatherInto(base.tri, drawn.triMap, 9));
  // #2664: the surfaces stay out of the MSAA canvas, the one way they draw the same pixels twice:
  // on their own camera layer, and laid over the canvas by one full-screen triangle, multiplied by alpha already.
  const main = new THREE.Layers();
  for (const name of ["ghost edges", "ghost faces"]) assert.equal(objs.find((o) => o.name === name)!.layers.test(main), false, name);
  const composite = objs.find((o) => o.name === "ghost composite") as THREE.Mesh;
  assert.equal(composite.layers.test(main), true);
  assert.ok(composite.renderOrder > 0 && composite.renderOrder < 1, "after the deformed mesh's blended edges (0), before the gizmos and the selection (1 and up)");
  assert.equal(composite.frustumCulled, false);
  const cm = composite.material as THREE.ShaderMaterial;
  assert.deepEqual([cm.transparent, cm.depthTest, cm.depthWrite], [true, false, false]);
  // #2715: the canvas's own blend over a pixel whose samples differ is not reproducible, so the triangle does not
  // blend in the canvas: it writes, to every sample, what its shader blends over a copy of the canvas taken just before.
  assert.equal(cm.blending, THREE.NoBlending, "the canvas does not blend the composite");
  assert.ok(cm.uniforms["below"]!.value instanceof THREE.FramebufferTexture, "the copy of the canvas under the composite");
  assert.match(cm.fragmentShader, /if \(g == vec4\(0\.0\)\) discard;/, "a pixel the ghost does not reach keeps its own samples");
  assert.match(cm.fragmentShader, /gl_FragColor = g \+ texelFetch\(below, ivec2\(gl_FragCoord\.xy\), 0\) \* \(1\.0 - g\.a\);/, "premultiplied over, of this pixel's copy");
  const segs = objs.filter((o) => o instanceof THREE.LineSegments);
  assert.deepEqual(segs.map((o) => o.name), ["ghost edges", "ghost lines"]);
  assert.deepEqual(segs.map(pos), [gatherInto(base.edge, edgeMap, 6), gatherInto(base.line, drawn.lineMap, 6)]);

  const group = new THREE.Group();
  for (const o of objs) group.add(o);
  const ghostEdges = segs[0]!, ghostLines = segs[1]!;

  // Default (edges on, opacity 1): faces, outline edges and 1-D lines all visible.
  applyGhostVisibility(group, true, full, 1);
  assert.equal(faces.visible, true);
  assert.equal(ghostEdges.visible, true);
  assert.equal(ghostLines.visible, true);

  // Edges off, opacity 1: outline edges hidden, faces and 1-D lines stay visible.
  applyGhostVisibility(group, false, full, 1);
  assert.equal(faces.visible, true);
  assert.equal(ghostEdges.visible, false);
  assert.equal(ghostLines.visible, true);

  // Surface opacity 0 ("off"), even with edges toggle true: ghost faces and ghost outline edges both hidden, 1-D ghost lines stay visible.
  applyGhostVisibility(group, true, full, 0);
  assert.equal(faces.visible, false, "ghost surfaces hidden when opacity is 0");
  assert.equal(ghostEdges.visible, false, "ghost outline edges hidden when opacity is 0");
  assert.equal(ghostLines.visible, true, "1-D ghost lines remain visible when 2-D/3-D surfaces are off");

  // Owner rulings on #2664: over translucent surfaces the ghost composites first, under every blended part of the mesh (main's look);
  // over opaque ones, after the mesh's edges. Each frame the first triangle decides, from the scene, which of the two draws.
  const under = objs.find((o) => o.name === "ghost composite under") as THREE.Mesh;
  assert.ok(under.renderOrder < ghostLines.renderOrder && ghostLines.renderOrder < 0 && composite.renderOrder > 0);
  applyGhostVisibility(group, true, full, 1);
  const scene = new THREE.Scene();
  scene.add(group);
  const solid = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshStandardMaterial());
  const frames = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshStandardMaterial());
  scene.add(solid, frames);
  // A renderer that records nothing: the target pass is the GPU's; the decision is not.
  const noop = () => {};
  const state = { color: { setMask: noop, setLocked: noop }, depth: { setMask: noop } };
  // The canvas copies it records: when, and how large.
  const copies: { at: "under" | "over"; w: number; h: number }[] = [];
  let at: "under" | "over" = "under";
  const renderer = {
    getDrawingBufferSize: (v: THREE.Vector2) => v.set(4, 4), getRenderTarget: () => null, autoClear: true, getClearAlpha: () => 0,
    getClearColor: (c: THREE.Color) => c, setRenderTarget: noop, setClearColor: noop, clear: noop, render: noop, state: { buffers: state },
    copyFramebufferToTexture: (t: THREE.FramebufferTexture) => copies.push({ at, w: t.image.width, h: t.image.height }),
  } as unknown as THREE.WebGLRenderer;
  const draws = (): [boolean, boolean] => {
    copies.length = 0;
    at = "under";
    under.onBeforeRender(renderer, scene, new THREE.PerspectiveCamera(), under.geometry, under.material as THREE.Material, new THREE.Group());
    at = "over";
    composite.onBeforeRender(renderer, scene, new THREE.PerspectiveCamera(), composite.geometry, composite.material as THREE.Material, new THREE.Group());
    const d: [boolean, boolean] = [under.geometry.drawRange.count > 0, composite.geometry.drawRange.count > 0];
    // The canvas is copied once a frame, at the place of the triangle that draws, at the drawing buffer's size, into the texture it reads.
    assert.deepEqual(copies, [{ at: d[0] ? "under" : "over", w: 4, h: 4 }]);
    assert.equal((cm.uniforms["below"]!.value as THREE.FramebufferTexture).image.width, 4);
    return d;
  };
  const set = (m: THREE.Mesh, opacity: number) => Object.assign(m.material as THREE.Material, { transparent: opacity < 1, opacity });
  for (const [what, solidOpacity, frameOpacity, frameShown, isUnder] of [
    ["everything opaque", 1, 1, true, false],
    ["surface opacity 0.35", 0.35, 0.35, true, true],
    ["the line diagram's 3-D frames at 0.25 under an opacity of 1 (owner ruling: main's look)", 1, 0.25, true, true],
    ["those frames hidden", 1, 0.25, false, false],
  ] as const) {
    set(solid, solidOpacity);
    set(frames, frameOpacity);
    frames.visible = frameShown;
    assert.equal(translucentSurfaces(scene), isUnder, what);
    assert.deepEqual(draws(), [isUnder, !isUnder], `${what}: exactly one of the two triangles draws`);
  }
  // The ghost's own translucent faces and the outline edges' blending never count.
  set(solid, 1);
  set(frames, 1);
  assert.equal(translucentSurfaces(scene), false);
  // Nor does what draws after both triangles: a plane gizmo's quad (renderOrder 1, 0.25) or the selection fill (9).
  for (const renderOrder of [1, 9]) {
    const late = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.25 }));
    late.renderOrder = renderOrder;
    scene.add(late);
    assert.equal(translucentSurfaces(scene), false, `a blended mesh at renderOrder ${renderOrder}`);
    assert.deepEqual(draws(), [false, true], `renderOrder ${renderOrder}: the ghost stays over the opaque mesh`);
    scene.remove(late);
  }

  h.effects.dispose();
});
