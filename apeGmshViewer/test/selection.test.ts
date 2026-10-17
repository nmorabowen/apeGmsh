// The selection halo's pure parts (R1 follow-up on #1308), against closed
// forms: the pulse envelope, the bounds `F` frames, and the fit distance.

import assert from "node:assert/strict";
import { test } from "node:test";
import * as THREE from "three";
import { LineMaterial } from "three/addons/lines/LineMaterial.js";
import { LineSegments2 } from "three/addons/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/addons/lines/LineSegmentsGeometry.js";
import { mountLegend, SURFACE_OPACITY_PRESETS } from "../src/panels/legend.ts";
import { edgesShown, type FullPositions } from "../src/renderer/deform.ts";
import { fitDistance, HALO_COLOUR, HALO_LAYER, HALO_PX, highlightObjects, hitElement, maskBytes, outlineOf, pulseAt, PULSE_MS, PULSE_SCALE, SELECTION_COLOUR, selectionBounds } from "../src/renderer/selection.ts";
import { Viewport } from "../src/renderer/viewport.ts";
import { initialState, reduce, Store } from "../src/state/store.ts";
import type { Event, MeshInfo, Pick, State } from "../src/state/types.ts";
import { DARK, OFFICE } from "../src/theme/tokens.ts";

test("the selection core is the office accent #E69F00 and the halo is white (#1308 round 3, item 3)", () => {
  // The maintainer's decision names both colours; the token module is the only place they live.
  assert.equal(OFFICE.accent, "#E69F00");
  assert.equal(DARK.selection, OFFICE.accent);
  assert.equal(SELECTION_COLOUR, 0xe69f00, "the core (outline and fill) is the office accent");
  assert.equal(HALO_COLOUR, 0xffffff, "the halo stays white");
  assert.equal(DARK.halo.toUpperCase(), "#FFFFFF");
});

test("every highlight object sits on HALO_LAYER (the mask pass) and on the default layer (the frame), without depth test", () => {
  // One segment and one quad (two triangles, fanned from the first corner).
  const lp = new Float32Array([0, 0, 0, 1, 0, 0]);
  const tp = new Float32Array([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0]);
  const made: number[] = [];
  const objs = highlightObjects(lp, tp, [{ lines: [0], tris: [0, 1] }], (pos, width) => {
    made.push(width);
    return new THREE.LineSegments(new THREE.BufferGeometry().setAttribute("position", new THREE.BufferAttribute(pos, 3)), new THREE.LineBasicMaterial());
  });
  assert.equal(objs.length, 3, "segments, outline, fill");
  assert.deepEqual(made, [9, 6], "thick segments, thinner outline");
  for (const o of objs) {
    assert.ok(o.layers.isEnabled(HALO_LAYER), `${o.type} on the halo layer`);
    assert.ok(o.layers.isEnabled(0), `${o.type} still on the default layer`);
    assert.equal(((o as THREE.Mesh).material as THREE.Material).depthTest, false, `${o.type} draws over everything`);
  }
  // The quad's outline is its four sides, not its diagonal.
  assert.equal(outlineOf(tp, [0, 1]).length / 6, 4);
  assert.deepEqual(highlightObjects(lp, tp, [], () => new THREE.Object3D()), []);
});

test("the mask target is one RGBA8 texel per device pixel", () => {
  assert.equal(maskBytes(1584, 961, 1), 1584 * 961 * 4);
  assert.equal(maskBytes(1584, 961, 2), 4 * 1584 * 961 * 4);
});

const close = (a: number, b: number, tol: number, what: string) => assert.ok(Math.abs(a - b) <= tol, `${what}: ${a} vs ${b}`);

test("halo: a fixed width of a few device pixels, and the pulse swells to PULSE_SCALE and settles", () => {
  assert.equal(HALO_PX, 3);
  assert.equal(pulseAt(0), 1, "at the select, the resting width");
  close(pulseAt(PULSE_MS / 2), PULSE_SCALE, 1e-12, "the middle of the pulse");
  close(pulseAt(PULSE_MS - 1e-9), 1, 1e-6, "the end of the pulse");
  assert.equal(pulseAt(PULSE_MS), 1);
  assert.equal(pulseAt(PULSE_MS * 10), 1);
  assert.equal(pulseAt(Infinity), 1, "no pulse running");
  assert.equal(pulseAt(NaN), 1);
  for (let t = 0; t < PULSE_MS; t += 25) assert.ok(pulseAt(t) >= 1 && pulseAt(t) <= PULSE_SCALE, `t=${t}`);
});

test("selection bounds: the box of the selected primitives, half its diagonal as the radius", () => {
  // Two segments and one triangle; only the triangle and the first segment are selected.
  const lp = new Float32Array([0, 0, 0, 1, 0, 0, 100, 100, 100, 101, 101, 101]);
  const tp = new Float32Array([0, 0, 0, 2, 0, 0, 0, 3, 0]);
  const b = selectionBounds(lp, tp, [0], [0], 1e-3)!;
  assert.deepEqual(b.center, [1, 1.5, 0]);
  close(b.radius, Math.hypot(2, 3, 0) / 2, 1e-12, "radius");
  // Only the far segment: a different box.
  assert.deepEqual(selectionBounds(lp, tp, [1], [], 1e-3)!.center, [100.5, 100.5, 100.5]);
  // Nothing drawn for the selection: null. A degenerate selection keeps the floor radius.
  assert.equal(selectionBounds(lp, tp, [], [], 1e-3), null);
  assert.equal(selectionBounds(new Float32Array([5, 5, 5, 5, 5, 5]), tp, [0], [], 0.25)!.radius, 0.25);
});

test("fit distance: the sphere touches the narrower half-angle of the view, with 8 % of margin", () => {
  const fov = 35;
  const half = (fov / 2) * (Math.PI / 180);
  // A wide free area: the vertical half-angle is the narrower one.
  close(fitDistance(10, fov, 1600, 900), (10 / Math.sin(half)) * 1.08, 1e-9, "wide");
  // A tall, narrow free area: the horizontal half-angle limits.
  const halfH = Math.atan(Math.tan(half) * (300 / 900));
  close(fitDistance(10, fov, 300, 900), (10 / Math.sin(halfH)) * 1.08, 1e-9, "narrow");
  // Framing a selection ten times smaller brings the camera ten times closer.
  close(fitDistance(1, fov, 1600, 900) * 10, fitDistance(10, fov, 1600, 900), 1e-9, "linear in the radius");
});

const proto = Viewport.prototype as unknown as {
  selectAt(this: unknown, x: number, y: number, add?: boolean): void;
  applyVisibility(this: unknown, s: State, rebuild?: boolean): void;
  pointAt(this: unknown, x: number, y: number): THREE.Vector3 | null;
  aim(this: unknown, x: number, y: number): void;
  wireInput(this: unknown): { nav: { dispose(): void } };
};

test("selectAt: plain click selects or clears on a miss; modified click (add=true) dispatches selectAdd on a hit and preserves selection on a miss", () => {
  const decls = Object.assign(Object.create(null), {
    "mesh/element/1": { path: "mesh/element/1" },
    "mesh/element/2": { path: "mesh/element/2" },
  });
  let state: State = { ...initialState, decls };
  const events: Event[] = [];
  let nextPick: Pick | null = null;
  const receiver = {
    pickAt: () => nextPick,
    store: {
      get: () => state,
      dispatch: (e: Event) => {
        events.push(e);
        state = reduce(state, e);
      },
    },
  };
  // Plain click on element 1 -> select.
  nextPick = { decl: "mesh/element/1", at: [0, 0, 0], nodes: [1, 2] };
  proto.selectAt.call(receiver, 10, 10, false);
  assert.deepEqual(state.selection.decls, ["mesh/element/1"]);
  // Modified click on element 2 -> selectAdd appends.
  nextPick = { decl: "mesh/element/2", at: [1, 0, 0], nodes: [2, 3] };
  proto.selectAt.call(receiver, 20, 20, true);
  assert.deepEqual(state.selection.decls, ["mesh/element/1", "mesh/element/2"]);
  // Modified click on a miss -> does NOT clear selection!
  nextPick = null;
  const countBeforeMiss = events.length;
  proto.selectAt.call(receiver, 50, 50, true);
  assert.equal(events.length, countBeforeMiss, "modifier click on a miss dispatches nothing");
  assert.deepEqual(state.selection.decls, ["mesh/element/1", "mesh/element/2"]);
  // Modified click on element 1 again -> selectAdd toggles element 1 off, keeping element 2.
  nextPick = { decl: "mesh/element/1", at: [0, 0, 0], nodes: [1, 2] };
  proto.selectAt.call(receiver, 10, 10, true);
  assert.deepEqual(state.selection.decls, ["mesh/element/2"]);
  // Plain click on a miss -> clearSelection.
  nextPick = null;
  proto.selectAt.call(receiver, 50, 50, false);
  assert.deepEqual(state.selection.decls, []);
});

test("applyVisibility & hitElement: opacity 1 / 0.35 / 0 controls faces.visible and edges.visible, and opacity 0 lets clicks pick 1-D beams behind slabs", () => {
  const full = { line: new Float32Array(0), tri: new Float32Array(0), edge: new Float32Array(0), key: "base", disp: null } satisfies FullPositions;
  assert.equal(edgesShown(true, full, 1), true);
  assert.equal(edgesShown(true, full, 0.35), true);
  assert.equal(edgesShown(true, full, 0), false, "edges are hidden when surface opacity is 0");
  assert.equal(edgesShown(false, full, 1), false);

  // A 2x2 slab face at z = 1 in front of a 1-D beam along y = 1 at z = 0, viewed from z = 10 looking down -Z.
  const slabPos = new Float32Array([0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 0, 1, 2, 2, 1, 0, 2, 1]);
  const faces = new THREE.Mesh(
    new THREE.BufferGeometry().setAttribute("position", new THREE.BufferAttribute(slabPos, 3)),
    new THREE.MeshStandardMaterial({ side: THREE.DoubleSide }),
  );
  const edges = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial());
  const lg = new LineSegmentsGeometry();
  lg.setPositions(new Float32Array([0, 1, 0, 2, 1, 0]));
  const lm = new LineMaterial({ linewidth: 3, worldUnits: false });
  lm.resolution.set(800, 800);
  const lines = new LineSegments2(lg, lm);
  const camera = new THREE.OrthographicCamera(-4, 4, 4, -4, 0.1, 100);
  camera.position.set(1, 1, 10);
  camera.lookAt(1, 1, 0);
  camera.updateMatrixWorld();
  faces.updateMatrixWorld();
  lines.updateMatrixWorld();

  const mesh = {
    counts: { nodes: 6, cells: 2, lineCells: 1, surfaceCells: 1, solidCells: 0, opsOnly: 0 },
    linePositions: { id: "lp", dtype: "f32", shape: [1, 6] },
    triPositions: { id: "tp", dtype: "f32", shape: [2, 9] },
  } as unknown as MeshInfo;
  const receiver = {
    faces,
    edges,
    lines,
    full,
    drawn: { lineMap: Int32Array.from([0]), triMap: Int32Array.from([0, 1]), edgeMask: null },
    camera,
    raycaster: new THREE.Raycaster(),
    renderer: { domElement: { getBoundingClientRect: () => ({ left: 0, top: 0, width: 800, height: 800 }) } },
    aim: proto.aim,
    requestRender: () => {},
  };

  let s: State = { ...initialState, mesh };
  // Preset "solid" (opacity 1, edges on):
  proto.applyVisibility.call(receiver, s, false);
  assert.equal(faces.visible, true);
  assert.equal(edges.visible, true);
  assert.equal((faces.material as THREE.MeshStandardMaterial).opacity, 1);
  assert.equal((faces.material as THREE.MeshStandardMaterial).transparent, false);
  assert.equal((faces.material as THREE.MeshStandardMaterial).depthWrite, true);

  // Preset "ghost" (opacity 0.35):
  s = reduce(s, { type: "setOpacity", value: 0.35 });
  proto.applyVisibility.call(receiver, s, false);
  assert.equal(faces.visible, true);
  assert.equal(edges.visible, true);
  close((faces.material as THREE.MeshStandardMaterial).opacity, 0.35, 1e-6, "ghost opacity");
  assert.equal((faces.material as THREE.MeshStandardMaterial).transparent, true);
  assert.equal((faces.material as THREE.MeshStandardMaterial).depthWrite, false);

  // Click on the slab at (0.5, 0.2, 1), away from the beam at y = 1:
  // When opacity > 0, the slab is hit; when opacity === 0 ("off"), faces.visible is false so the slab is neither hit nor picked.
  const slabScreen = new THREE.Vector3(0.5, 0.2, 1).project(camera);
  const sx = ((slabScreen.x + 1) / 2) * 800, sy = ((1 - slabScreen.y) / 2) * 800;
  proto.aim.call(receiver, sx, sy);
  assert.equal(
    hitElement(receiver.raycaster, lines, faces, receiver.drawn, ["mesh/element/beam", "mesh/element/slab"], Int32Array.from([0]), Int32Array.from([1, 1]), 2)?.element,
    1,
    "at opacity 0.35, clicking the slab picks the slab",
  );
  assert.ok(proto.pointAt.call(receiver, sx, sy) !== null, "pointAt hits the visible slab");

  // Preset "off" (opacity 0): faces.visible and edges.visible both become false!
  s = reduce(s, { type: "setOpacity", value: 0 });
  proto.applyVisibility.call(receiver, s, false);
  assert.equal(faces.visible, false, "faces.visible is false at opacity 0");
  assert.equal(edges.visible, false, "edges.visible is false at opacity 0");
  proto.aim.call(receiver, sx, sy);
  assert.equal(
    hitElement(receiver.raycaster, lines, faces, receiver.drawn, ["mesh/element/beam", "mesh/element/slab"], Int32Array.from([0]), Int32Array.from([1, 1]), 2),
    null,
    "at opacity 0, the hidden slab does not intercept the raycaster",
  );
  assert.equal(proto.pointAt.call(receiver, sx, sy), null, "pointAt ignores hidden faces");

  // Clicking on the 1-D beam at (1, 1, 0) with faces off picks the beam directly:
  receiver.raycaster.params.Line2 = { threshold: 7 };
  const beamScreen = new THREE.Vector3(1, 1, 0).project(camera);
  const bx = ((beamScreen.x + 1) / 2) * 800, by = ((1 - beamScreen.y) / 2) * 800;
  proto.aim.call(receiver, bx, by);
  assert.equal(
    hitElement(receiver.raycaster, lines, faces, receiver.drawn, ["mesh/element/beam", "mesh/element/slab"], Int32Array.from([0]), Int32Array.from([1, 1]), 2)?.element,
    0,
    "at opacity 0, clicking the 1-D beam picks the beam",
  );

  // Mixed solid + 1-D model (solidCells > 0 && lineCells > 0): preset "solid" (opacity = 1) must be truly opaque (opacity 1, not clamped to 0.35),
  // while "ghost" (0.35) is translucent and "off" (0) hides surfaces.
  const mixedSolidMesh = {
    counts: { lineCells: 1, faceCells: 0, solidCells: 1, opsOnly: 0, points: 0 },
    linePositions: { id: "lp", dtype: "f32", shape: [1, 6] },
    triPositions: { id: "tp", dtype: "f32", shape: [2, 9] },
  } as unknown as MeshInfo;
  let sMixed: State = { ...initialState, mesh: mixedSolidMesh };
  proto.applyVisibility.call(receiver, sMixed, false);
  assert.equal(faces.visible, true);
  assert.equal((faces.material as THREE.MeshStandardMaterial).opacity, 1, "solid preset on solid+line mesh is truly opaque (1)");
  assert.equal((faces.material as THREE.MeshStandardMaterial).transparent, false);
  assert.equal((faces.material as THREE.MeshStandardMaterial).depthWrite, true);

  sMixed = reduce(sMixed, { type: "setOpacity", value: 0.35 });
  proto.applyVisibility.call(receiver, sMixed, false);
  close((faces.material as THREE.MeshStandardMaterial).opacity, 0.35, 1e-6, "ghost preset on solid+line mesh is 0.35");
  assert.equal((faces.material as THREE.MeshStandardMaterial).transparent, true);
  assert.equal((faces.material as THREE.MeshStandardMaterial).depthWrite, false);

  sMixed = reduce(sMixed, { type: "setOpacity", value: 0 });
  proto.applyVisibility.call(receiver, sMixed, false);
  assert.equal(faces.visible, false, "off preset on solid+line mesh hides faces");
  assert.equal(edges.visible, false, "off preset on solid+line mesh hides edges");
});

class FakeLegendEl {
  children: FakeLegendEl[] = [];
  className = "";
  id = "";
  title = "";
  hidden = false;
  disabled = false;
  value = "";
  type = "";
  step = "";
  min = "";
  label = "";
  selected = false;
  style: Record<string, string> = {};
  private text = "";
  private listeners = new Map<string, (() => void)[]>();
  readonly tagName: string;
  constructor(tagName: string) {
    this.tagName = tagName;
  }
  get textContent(): string {
    return this.text + this.children.map((c) => c.textContent).join("");
  }
  set textContent(t: string) {
    this.text = t;
    this.children = [];
  }
  append(...xs: (FakeLegendEl | string)[]): void {
    for (const x of xs) {
      if (typeof x === "string") {
        const n = new FakeLegendEl("#text");
        n.text = x;
        this.children.push(n);
      } else this.children.push(x);
    }
  }
  replaceChildren(...xs: FakeLegendEl[]): void {
    this.children = [];
    this.append(...xs);
  }
  addEventListener(t: string, fn: () => void): void {
    (this.listeners.get(t) ?? this.listeners.set(t, []).get(t)!).push(fn);
  }
  removeEventListener(t: string, fn: () => void): void {
    const l = this.listeners.get(t) ?? [];
    this.listeners.set(t, l.filter((f) => f !== fn));
  }
  click(): void {
    for (const f of this.listeners.get("click") ?? []) f();
  }
  has(cls: string): boolean {
    return this.className.split(" ").includes(cls);
  }
  all(pred: (e: FakeLegendEl) => boolean): FakeLegendEl[] {
    const out: FakeLegendEl[] = [];
    for (const c of this.children) {
      if (pred(c)) out.push(c);
      out.push(...c.all(pred));
    }
    return out;
  }
}

test("mountLegend: exposes .legend-mesh only when 2-D/3-D cells exist, and disables/unhighlights edges when surface opacity is 0", () => {
  assert.deepEqual(SURFACE_OPACITY_PRESETS.map((p) => [p.label, p.opacity]), [
    ["solid", 1],
    ["ghost", 0.35],
    ["off", 0],
  ]);
  const legendRoot = new FakeLegendEl("DIV");
  legendRoot.id = "legend";
  const prevDoc = (globalThis as { document?: unknown }).document;
  (globalThis as { document?: unknown }).document = {
    getElementById: (id: string) => (id === "legend" ? legendRoot : null),
    createElement: (t: string) => new FakeLegendEl(t.toUpperCase()),
  };
  // 1-D-only mesh (faceCells = 0, solidCells = 0): .legend-mesh is omitted.
  const store1d = new Store();
  (store1d as unknown as { state: State }).state = {
    ...store1d.get(),
    mesh: { legend: [], counts: { lineCells: 2, faceCells: 0, solidCells: 0, opsOnly: 0, points: 0 } } as unknown as MeshInfo,
  };
  const unmount1d = mountLegend(store1d);
  assert.equal(legendRoot.all((e) => e.has("legend-mesh")).length, 0, "1-D-only model has no .legend-mesh row");
  unmount1d();

  // Mesh with 2-D/3-D cells: .legend-mesh appears.
  const store = new Store();
  (store as unknown as { state: State }).state = {
    ...store.get(),
    mesh: { legend: [], counts: { lineCells: 1, faceCells: 2, solidCells: 0, opsOnly: 0, points: 0 } } as unknown as MeshInfo,
  };
  const unmount = mountLegend(store);
  try {
    const btn = (cls: string) => legendRoot.all((e) => e.has(cls))[0]!;
    assert.equal(legendRoot.all((e) => e.has("legend-mesh")).length, 1);
    assert.ok(btn("mesh-edges").has("on"), "edges starts on");
    assert.equal(btn("mesh-edges").disabled, false);
    assert.ok(btn("mesh-opacity-solid").has("on"), "solid (1) starts on");
    assert.ok(!btn("mesh-opacity-ghost").has("on"));
    assert.ok(!btn("mesh-opacity-off").has("on"));

    btn("mesh-edges").click();
    assert.equal(store.get().visibility.edges, false);
    assert.ok(!btn("mesh-edges").has("on"), "edges button updates on setEdges");
    btn("mesh-edges").click();
    assert.equal(store.get().visibility.edges, true);
    assert.ok(btn("mesh-edges").has("on"));

    btn("mesh-opacity-ghost").click();
    assert.equal(store.get().visibility.opacity, 0.35);
    assert.ok(btn("mesh-opacity-ghost").has("on"));
    assert.ok(!btn("mesh-opacity-solid").has("on"));
    assert.ok(btn("mesh-edges").has("on"), "edges stays on in ghost mode");
    assert.equal(btn("mesh-edges").disabled, false);

    btn("mesh-opacity-off").click();
    assert.equal(store.get().visibility.opacity, 0);
    assert.ok(btn("mesh-opacity-off").has("on"));
    assert.ok(!btn("mesh-opacity-ghost").has("on"));
    assert.ok(!btn("mesh-edges").has("on"), "edges is not shown as .on while surfaces are off");
    assert.equal(btn("mesh-edges").disabled, true, "edges button is disabled while surfaces are off");

    btn("mesh-opacity-solid").click();
    assert.equal(store.get().visibility.opacity, 1);
    assert.ok(btn("mesh-opacity-solid").has("on"));
    assert.ok(btn("mesh-edges").has("on"), "edges is shown as .on again when surfaces return");
    assert.equal(btn("mesh-edges").disabled, false);

    // Keyboard shortcut E (Viewport.wireInput toggleEdges) toggles edges while surfaces are on,
    // and is a no-op while surfaces are off (opacity === 0) or on a 1-D-only mesh.
    const prevWin = (globalThis as { window?: unknown }).window;
    const winTarget = new EventTarget();
    (globalThis as { window?: unknown }).window = winTarget;
    const canvas = Object.assign(new EventTarget(), {
      clientHeight: 600,
      style: { cursor: "" },
      setPointerCapture: () => {},
      hasPointerCapture: () => false,
      releasePointerCapture: () => {},
    });
    const navReceiver = {
      renderer: { domElement: canvas },
      camera: new THREE.PerspectiveCamera(),
      gizmos: { root: { visible: false } },
      store,
      geometryInfo: null,
      aim: () => {},
      pointAt: () => null,
      selectAt: () => {},
      settleCaps: () => {},
      requestRender: () => {},
    };
    const { nav } = proto.wireInput.call(navReceiver);
    try {
      const pressE = () =>
        winTarget.dispatchEvent(Object.assign(new globalThis.Event("keydown"), { key: "e", ctrlKey: false, altKey: false, metaKey: false, repeat: false }));
      pressE();
      assert.equal(store.get().visibility.edges, false, "E toggles edges when surfaces are on");
      pressE();
      assert.equal(store.get().visibility.edges, true);
      btn("mesh-opacity-off").click();
      assert.equal(store.get().visibility.opacity, 0);
      pressE();
      assert.equal(store.get().visibility.edges, true, "E is a no-op while surface opacity is 0 (off)");
    } finally {
      nav.dispose();
      (globalThis as { window?: unknown }).window = prevWin;
    }
  } finally {
    unmount();
    (globalThis as { document?: unknown }).document = prevDoc;
  }
});

