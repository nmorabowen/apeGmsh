// Line-diagram Slice E2b + Performance & Scale tests:
// 1. E2b wiring: `buildLineDiagram3D`, `Effects.syncLineDiagram`, `LineDiagramLayer.sync`,
//    and `renderLineDiagram` invoke `reconstructSpanDiagramsForStep` (`state/spanLoads.ts`)
//    for beam force/moment components (`N`, `Vy`, `Vz`, `T`, `My`, `Mz`), replacing 2-point
//    linear station interpolation with exact analytical curves and surfacing `.line-diagram-span-badge`.
// 2. Perf 1: `buildLineDiagram3D` runs once per state change across `Effects` and `LineDiagramLayer`.
// 3. Perf 2: Complete cache keys in `Effects` and `LineDiagramLayer` include `visibleLineDigest`,
//    `stage`, `stepTime`, and `definitions` so `isolate` and stage element activation rebuild.
// 4. Perf 3: `LineDiagramLayer.onCameraChange()` coalesces screen-plane extrusion rebuilds to at
//    most once per animation frame (`flushCameraChange`).
// 5. Perf 4: `LadrunoFile.readLineStations` and `ResultsHost.scan` (`source: "line"`) read step
//    chunks by hyperslab (`stepRange: [k0, k1]`) instead of allocating the full stage slab up front.

import assert from "node:assert/strict";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import * as h5wasm from "h5wasm/node";
import * as THREE from "three";
import { ResultsHost } from "../../src/main/results.ts";
import { renderLineDiagram } from "../../src/panels/lineDiagram.ts";
import { openLadruno } from "../../src/reader/ladruno.ts";
import type { ResultsH5 } from "../../src/reader/results.ts";
import { LineDiagramLayer } from "../../src/renderer/lineDiagram.ts";
import {
  buildLineDiagram3D,
  getLineDiagram3DBuildCount,
  joinLadrunoElements,
  lineDiagramClimOf,
  lineDiagramKeysOf,
  resetLineDiagram3DBuildCount,
  visibleLineDigest,
} from "../../src/state/lineDiagram.ts";
import { resultsPairing } from "../../src/state/selectors.ts";
import type { Decl, DeclPath, Definitions, SequenceZone } from "../../src/state/types.ts";
import { opened, until } from "../contour/harness.ts";

const SS_UDL_MODEL = fileURLToPath(new URL("../../fixtures/line_oracle_ss_udl.h5", import.meta.url));
const SS_UDL_LADRUNO = fileURLToPath(new URL("../../fixtures/line_oracle_ss_udl.ladruno", import.meta.url));
const FBC_LEGENDRE_MODEL = fileURLToPath(new URL("../../fixtures/line_oracle_fbc_legendre.h5", import.meta.url));
const FBC_LEGENDRE_LADRUNO = fileURLToPath(new URL("../../fixtures/line_oracle_fbc_legendre.ladruno", import.meta.url));
const FRAME3D_MODEL = fileURLToPath(new URL("../../fixtures/ladruno_frame3d.h5", import.meta.url));
const FRAME3D_LADRUNO = fileURLToPath(new URL("../../fixtures/ladruno_frame3d.ladruno", import.meta.url));

class FakeEl {
  children: FakeEl[] = [];
  parentElement: FakeEl | null = null;
  dataset: Record<string, string | undefined> = {};
  className = "";
  title = "";
  label = "";
  disabled = false;
  hidden = false;
  type = "";
  value = "";
  step = "";
  min = "";
  selected = false;
  validity = "";
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
  append(...xs: (FakeEl | string)[]): void {
    for (const x of xs) {
      if (typeof x === "string") {
        const n = new FakeEl("#text");
        n.text = x;
        n.parentElement = this;
        this.children.push(n);
      } else {
        x.parentElement = this;
        this.children.push(x);
      }
    }
  }
  replaceChildren(...xs: (FakeEl | string)[]): void {
    this.text = "";
    this.children = [];
    this.append(...xs);
  }
  setAttribute(_k: string, _v: string): void {}
  addEventListener(ev: string, fn: () => void): void {
    const arr = this.listeners.get(ev) ?? [];
    arr.push(fn);
    this.listeners.set(ev, arr);
  }
  removeEventListener(ev: string, fn: () => void): void {
    const arr = this.listeners.get(ev) ?? [];
    this.listeners.set(
      ev,
      arr.filter((f) => f !== fn),
    );
  }
  setCustomValidity(msg: string): void {
    this.validity = msg;
  }
  find(cls: string): FakeEl | undefined {
    for (const c of this.children) {
      if (c.className.split(" ").includes(cls)) return c;
      const found = c.find(cls);
      if (found) return found;
    }
    return undefined;
  }
}

function makeSingleSpanUdlModel(wy: number): {
  decls: Record<DeclPath, Decl>;
  definitions: Definitions;
  sequence: SequenceZone;
} {
  const decls: Record<DeclPath, Decl> = {
    "opensees/geomTransf/tr1": {
      path: "opensees/geomTransf/tr1",
      kind: "geomTransf",
      type: "Linear",
      name: "tr1",
      nameSource: "/opensees/names",
      h5: "/opensees/transforms/Linear_1",
      tag: 1,
      params: [],
      refs: {},
      links: {},
      problems: [],
      fields: [],
    },
    "mesh/element/1": {
      path: "mesh/element/1",
      kind: "element",
      type: "elasticBeamColumn",
      name: "1",
      nameSource: "/elements/line2",
      h5: "/elements/line2",
      tag: null,
      params: [],
      refs: { transfTag: "opensees/geomTransf/tr1" },
      links: {},
      problems: [],
      fields: [],
      element: {
        femId: 1,
        cell: { alias: "line2", block: 0, row: 0 },
        inlineNodes: [1, 2],
        groups: [],
        metas: [
          {
            type: "elasticBeamColumn",
            tag: 1,
            h5: "/opensees/element_meta/elasticBeamColumn",
            row: 0,
            args: [1, 1],
          },
        ],
      },
    },
  };

  const definitions: Definitions = {
    decls: null,
    analysis: {
      path: "/opensees/analysis",
      attrs: { analysis: "Static", integrator: "LoadControl" },
    },
    timeSeries: [
      {
        path: "/opensees/time_series/Linear_1",
        groupName: "Linear_1",
        type: "Linear",
        tag: 1,
        params: [1.0],
        attrs: {},
        tables: {},
      },
    ],
    patterns: [
      {
        path: "/opensees/patterns/Plain_1",
        groupName: "Plain_1",
        type: "Plain",
        tag: 1,
        params: [1],
        attrs: { series_ref: "/opensees/time_series/Linear_1" },
        tables: {
          element_loads: {
            path: "/opensees/patterns/Plain_1/element_loads",
            columns: ["[0]", "[1]", "[2]", "[3]", "[4]", "[5]"],
            rows: [["-ele", "1", "-type", "-beamUniform", String(wy), "0"]],
          },
        },
      },
    ],
    recorders: [],
    bcs: { fix: null, mass: null },
    stages: [],
    embeddedRebar: null,
    program: null,
  };

  return { decls, definitions, sequence: { status: "absent" } };
}

test("E2b: buildLineDiagram3D reconstructs exact parabolic Mz(x) on single-span 6 m UDL beam (wy=-10000 -> Mz(0)=+45000) and falls back on closure mismatch", () => {
  const { decls, definitions, sequence } = makeSingleSpanUdlModel(-10000);
  const join = joinLadrunoElements(decls, { state: "paired", by: "model_hash" });
  assert.equal(join.ok, true);
  if (!join.ok) return;

  const meshElements: DeclPath[] = ["mesh/element/1"];
  const lineElement = new Int32Array([0]);
  const drawnLinePositions = new Float32Array([0, 0, 0, 6, 0, 0]);
  const lineNodes = new Float64Array([1, 2]);

  const vyStep = {
    component: "Vy" as const,
    time: 1.0,
    elementIndex: new Float64Array([1, 1]),
    stationNaturalCoord: new Float64Array([-1, 1]),
    values: new Float64Array([30000, -30000]),
    localAxesQuaternion: null,
    localAxesZRec: null,
  };
  const mzStep = {
    component: "Mz" as const,
    time: 1.0,
    elementIndex: new Float64Array([1, 1]),
    stationNaturalCoord: new Float64Array([-1, 1]),
    values: new Float64Array([0, 0]),
    localAxesQuaternion: null,
    localAxesZRec: null,
    companionStep: vyStep,
  };

  // 1. Without `definitions`, 2-point interpolation between Mz(-1)=0 and Mz(+1)=0 stays 0 everywhere:
  const interpGeom = buildLineDiagram3D({
    decls,
    meshElements,
    lineElement,
    drawnLinePositions,
    lineNodes,
    stepData: mzStep,
    join,
    ndm: 2,
  });
  assert.equal(interpGeom.exactSpanCount, 0);
  assert.equal(interpGeom.interpolatedSpanCount, 1);
  assert.equal(interpGeom.peakAbsValue, 0);
  assert.equal(interpGeom.beams[0]?.spanMode, "interpolated");
  assert.match(interpGeom.beams[0]?.spanReason ?? "", /no \/opensees load definitions/);

  // 2. With `definitions`, E2b reconstructs the exact parabola with peak Mz(xi=0) = +45,000 N*m:
  const exactGeom = buildLineDiagram3D({
    decls,
    definitions,
    sequence,
    meshElements,
    lineElement,
    drawnLinePositions,
    lineNodes,
    stepData: mzStep,
    join,
    ndm: 2,
  });
  assert.equal(exactGeom.exactSpanCount, 1);
  assert.equal(exactGeom.interpolatedSpanCount, 0);
  assert.equal(exactGeom.beams[0]?.spanMode, "exact");
  assert.equal(exactGeom.beams[0]?.spanReason, null);
  assert.ok(exactGeom.beams[0]!.xi.length > 2, "expected dense reconstructed stations");
  assert.ok(Math.abs(exactGeom.peakAbsValue - 45000) < 1e-6, `expected peak Mz = 45000, got ${exactGeom.peakAbsValue}`);
  assert.ok(Math.abs(exactGeom.clim[1] - 45000) < 1e-6, `expected clim[1] = 45000, got ${exactGeom.clim[1]}`);

  // 3. When `companionStep` (`Vy`) is absent on a loaded span, `buildLineDiagram3D` fails closed to "interpolated"
  //    rather than inferring `Vy(0)` from `Mz(0)` and `Mz(L)` (which would make `Mz(L) === Mz_J` a tautology):
  const mzStepWithoutVy = { ...mzStep, companionStep: null };
  const noVyGeom = buildLineDiagram3D({
    decls,
    definitions,
    sequence,
    meshElements,
    lineElement,
    drawnLinePositions,
    lineNodes,
    stepData: mzStepWithoutVy,
    join,
    ndm: 2,
  });
  assert.equal(noVyGeom.exactSpanCount, 0);
  assert.equal(noVyGeom.interpolatedSpanCount, 1);
  assert.equal(noVyGeom.beams[0]?.spanMode, "interpolated");
  assert.match(noVyGeom.beams[0]?.spanReason ?? "", /requires companion shear Vy end stations/);

  // 4. Perturbing Mz(L) so node-j equilibrium fails triggers fallback to "interpolated" with closure reason:
  const badMzStep = {
    ...mzStep,
    values: new Float64Array([0, 5000]),
  };
  const fallbackGeom = buildLineDiagram3D({
    decls,
    definitions,
    sequence,
    meshElements,
    lineElement,
    drawnLinePositions,
    lineNodes,
    stepData: badMzStep,
    join,
    ndm: 2,
  });
  assert.equal(fallbackGeom.exactSpanCount, 0);
  assert.equal(fallbackGeom.interpolatedSpanCount, 1);
  assert.equal(fallbackGeom.beams[0]?.spanMode, "interpolated");
  assert.match(fallbackGeom.beams[0]?.spanReason ?? "", /closure failed/i);

  // 5. Symmetric companion moment closure when viewing `Vy`:
  //    - With valid companion `Mz` attached, `Vy` is reconstructed as `exact`.
  //    - With perturbed companion `Mz(L) = 5000`, `Vy` fails moment closure and falls back to `interpolated`.
  //    - With missing companion `Mz` on a loaded span, `Vy` fails closed naming companion moment `Mz`.
  const vyWithValidMz = buildLineDiagram3D({
    decls,
    definitions,
    sequence,
    meshElements,
    lineElement,
    drawnLinePositions,
    lineNodes,
    stepData: { ...vyStep, companionStep: mzStep },
    join,
    ndm: 2,
  });
  assert.equal(vyWithValidMz.exactSpanCount, 1);
  assert.equal(vyWithValidMz.beams[0]?.spanMode, "exact");

  const vyWithBadMz = buildLineDiagram3D({
    decls,
    definitions,
    sequence,
    meshElements,
    lineElement,
    drawnLinePositions,
    lineNodes,
    stepData: { ...vyStep, companionStep: badMzStep },
    join,
    ndm: 2,
  });
  assert.equal(vyWithBadMz.exactSpanCount, 0);
  assert.equal(vyWithBadMz.interpolatedSpanCount, 1);
  assert.equal(vyWithBadMz.beams[0]?.spanMode, "interpolated");
  assert.match(vyWithBadMz.beams[0]?.spanReason ?? "", /closure failed/i);

  const vyWithoutMz = buildLineDiagram3D({
    decls,
    definitions,
    sequence,
    meshElements,
    lineElement,
    drawnLinePositions,
    lineNodes,
    stepData: { ...vyStep, companionStep: null },
    join,
    ndm: 2,
  });
  assert.equal(vyWithoutMz.exactSpanCount, 0);
  assert.equal(vyWithoutMz.interpolatedSpanCount, 1);
  assert.equal(vyWithoutMz.beams[0]?.spanMode, "interpolated");
  assert.match(vyWithoutMz.beams[0]?.spanReason ?? "", /requires companion moment Mz end stations/);

  // 6. Changing `stepData.localAxesSources` invalidates `buildLineDiagram3D`'s cache key:
  resetLineDiagram3DBuildCount();
  buildLineDiagram3D({
    decls,
    definitions,
    sequence,
    meshElements,
    lineElement,
    drawnLinePositions,
    lineNodes,
    stepData: { ...mzStep, localAxesSources: ["step_local_axes", "step_local_axes"] },
    join,
    ndm: 2,
  });
  assert.equal(getLineDiagram3DBuildCount(), 1);
  buildLineDiagram3D({
    decls,
    definitions,
    sequence,
    meshElements,
    lineElement,
    drawnLinePositions,
    lineNodes,
    stepData: { ...mzStep, localAxesSources: ["model_local_axes", "model_local_axes"] },
    join,
    ndm: 2,
  });
  assert.equal(getLineDiagram3DBuildCount(), 2, "localAxesSources change must invalidate buildLineDiagram3D cache");
});

test("E2b + Perf 1: line_oracle_ss_udl end-to-end reconstructs exact parabolic Mz in Effects + LineDiagramLayer + panel badge, building 3-D geometry only once per step", async () => {
  const h = await opened(SS_UDL_MODEL, SS_UDL_LADRUNO);
  const prevDoc = (globalThis as { document?: unknown }).document;
  (globalThis as { document?: unknown }).document = {
    documentElement: { dataset: { theme: "dark" } },
    createElement: (tag: string) => new FakeEl(tag),
    createElementNS: (_ns: string, tag: string) => new FakeEl(tag),
  };
  try {
    assert.equal(resultsPairing(h.store.get()).state, "paired");
    const camera = new THREE.PerspectiveCamera(35, 1, 0.1, 1000);
    camera.position.set(3, -10, 2);
    camera.lookAt(3, 0, 0);
    camera.updateMatrixWorld();
    const layer = new LineDiagramLayer(camera);

    // Subscribe `layer.sync` to store updates just like `Viewport.onState` does:
    const unsubLayer = h.store.subscribe((s) => {
      layer.sync(s, h.blobs, h.effects.currentLineDiagram(), h.effects.currentDisplacement());
    });

    resetLineDiagram3DBuildCount();
    h.store.dispatch({ type: "setPhase", at: { kind: "results", stage: 0, step: 1 } });
    h.store.dispatch({ type: "setLineDiagram", on: true });

    await until(() => {
      const cur = h.store.get();
      const ldKeys = lineDiagramKeysOf(cur);
      return (
        ldKeys !== null &&
        h.effects.currentLineDiagram()?.key === ldKeys.step &&
        cur.lineDiagram.stepRange?.key === ldKeys.step &&
        cur.lineDiagram.drawnBeams.length === 4
      );
    }, "line_oracle_ss_udl step 1 Mz ready");

    // Perf 1: across Effects.syncLineDiagram AND LineDiagramLayer.sync (before and after lineDiagramStepRange),
    // buildLineDiagram3D executed its geometry build body ONLY ONCE!
    assert.equal(
      getLineDiagram3DBuildCount(),
      1,
      `expected exactly 1 buildLineDiagram3D execution for step 1, got ${getLineDiagram3DBuildCount()}`,
    );

    // Step 1 is t = 1.0 (w = -10 on L = 6 m, 4 elements of 1.5 m: E3 [0,1.5], E4 [1.5,3.0], E5 [3.0,4.5], E6 [4.5,6.0]):
    // At element mid-spans (xi = 0):
    // - E3 (x = 0.75 m): exact parabolic Mz(0.75) = 0.5 * 10 * 0.75 * 5.25 = +19.6875 (vs. linear midpoint +16.875)
    // - E4 (x = 2.25 m): exact parabolic Mz(2.25) = 0.5 * 10 * 2.25 * 3.75 = +42.1875 (vs. linear midpoint +39.375)
    const s = h.store.get();
    assert.ok(
      Math.abs((s.lineDiagram.stepRange?.max ?? 0) - 45) < 1e-6,
      `expected exact parabolic peak 45 in stepRange.max, got ${s.lineDiagram.stepRange?.max}`,
    );
    for (const b of s.lineDiagram.drawnBeams) {
      assert.equal(b.spanMode, "exact");
      assert.equal(b.spanReason, null);
      assert.ok(b.xi.length > 2, `expected > 2 stations on beam E${b.elementId}, got ${b.xi.length}`);
    }
    const beam3 = s.lineDiagram.drawnBeams.find((b) => b.elementId === 3)!;
    const beam4 = s.lineDiagram.drawnBeams.find((b) => b.elementId === 4)!;
    const beam3MidVal = beam3.values[beam3.xi.findIndex((x) => Math.abs(x) < 1e-9)]!;
    const beam4MidVal = beam4.values[beam4.xi.findIndex((x) => Math.abs(x) < 1e-9)]!;
    assert.ok(
      Math.abs(beam3MidVal - 19.6875) < 1e-6,
      `expected E3 mid-span Mz(xi=0) = 19.6875 (parabolic, not linear 16.875), got ${beam3MidVal}`,
    );
    assert.ok(
      Math.abs(beam4MidVal - 42.1875) < 1e-6,
      `expected E4 mid-span Mz(xi=0) = 42.1875 (parabolic, not linear 39.375), got ${beam4MidVal}`,
    );

    // Select all 4 beams and verify `s.lineDiagram.chains` and `.line-diagram-span-badge` in `renderLineDiagram`:
    h.store.dispatch({ type: "selectDecls", decls: s.mesh!.elements });
    const sSelected = h.store.get();
    assert.equal(sSelected.lineDiagram.chains.length, 1);
    assert.equal(sSelected.lineDiagram.chains[0]?.spans.length, 4);
    const chainSpan3 = sSelected.lineDiagram.chains[0]!.spans.find((sp) => sp.elementId === 3)!;
    const chainSpan4 = sSelected.lineDiagram.chains[0]!.spans.find((sp) => sp.elementId === 4)!;
    const midStation3 = chainSpan3.stations.find((st) => Math.abs(st.xi) < 1e-9)!;
    const midStation4 = chainSpan4.stations.find((st) => Math.abs(st.xi) < 1e-9)!;
    assert.ok(Math.abs(midStation3.value - 19.6875) < 1e-6, `expected chain E3 mid-span 19.6875, got ${midStation3.value}`);
    assert.ok(Math.abs(midStation4.value - 42.1875) < 1e-6, `expected chain E4 mid-span 42.1875, got ${midStation4.value}`);
    const chainPeak = Math.max(
      ...sSelected.lineDiagram.chains[0]!.spans.flatMap((sp) => sp.stations.map((st) => st.value)),
    );
    assert.ok(Math.abs(chainPeak - 45) < 1e-6, `expected 2-D chain plot peak 45, got ${chainPeak}`);

    const dom = renderLineDiagram(sSelected) as unknown as FakeEl | null;
    assert.ok(dom !== null);
    const badge = dom.find("line-diagram-span-badge");
    assert.ok(badge !== undefined, "expected .line-diagram-span-badge in panel");
    assert.equal(badge.dataset["spanMode"], "exact");
    assert.equal(badge.dataset["exactCount"], "4");
    assert.equal(badge.dataset["interpolatedCount"], "0");
    assert.match(badge.textContent, /Span: exact \(4 beams\)/);

    unsubLayer();
    layer.dispose();
  } finally {
    (globalThis as { document?: unknown }).document = prevDoc;
    h.effects.dispose();
  }
});

test("E2b fallback badge: line_oracle_fbc_legendre surfaces interpolated mode and reason in drawnBeams and .line-diagram-span-badge", async () => {
  const h = await opened(FBC_LEGENDRE_MODEL, FBC_LEGENDRE_LADRUNO);
  const prevDoc = (globalThis as { document?: unknown }).document;
  (globalThis as { document?: unknown }).document = {
    documentElement: { dataset: { theme: "dark" } },
    createElement: (tag: string) => new FakeEl(tag),
    createElementNS: (_ns: string, tag: string) => new FakeEl(tag),
  };
  try {
    h.store.dispatch({ type: "setPhase", at: { kind: "results", stage: 0, step: 0 } });
    h.store.dispatch({ type: "setLineDiagram", on: true });

    await until(() => {
      const cur = h.store.get();
      const ldKeys = lineDiagramKeysOf(cur);
      return (
        ldKeys !== null &&
        cur.lineDiagram.stepRange?.key === ldKeys.step &&
        cur.lineDiagram.drawnBeams.length > 0
      );
    }, "line_oracle_fbc_legendre step 0 Mz ready");

    const s = h.store.get();
    for (const b of s.lineDiagram.drawnBeams) {
      assert.equal(b.spanMode, "interpolated");
      assert.ok((b.spanReason ?? "").length > 0);
    }

    const dom = renderLineDiagram(s) as unknown as FakeEl | null;
    assert.ok(dom !== null);
    const badge = dom.find("line-diagram-span-badge");
    assert.ok(badge !== undefined, "expected .line-diagram-span-badge on Legendre FBC model");
    assert.equal(badge.dataset["spanMode"], "interpolated");
    assert.equal(badge.dataset["exactCount"], "0");
    assert.match(badge.textContent, /Span: interpolated/);
    assert.match(badge.textContent, /interior|station/i);
  } finally {
    (globalThis as { document?: unknown }).document = prevDoc;
    h.effects.dispose();
  }
});

test("Perf 2 & Perf 3: visibleLineMap digest invalidates LineDiagramLayer cache on element activation/isolate, and onCameraChange coalesces screen-plane rebuilds to once per frame", async () => {
  const h = await opened(FRAME3D_MODEL, FRAME3D_LADRUNO);
  try {
    h.store.dispatch({ type: "setPhase", at: { kind: "results", stage: 0, step: 2 } });
    h.store.dispatch({ type: "setLineDiagram", on: true });

    await until(() => {
      const cur = h.store.get();
      const ldKeys = lineDiagramKeysOf(cur);
      return ldKeys !== null && h.effects.currentLineDiagram()?.key === ldKeys.step && cur.lineDiagram.stepRange?.key === ldKeys.step;
    }, "ladruno_frame3d step 2 ready");

    const camera = new THREE.PerspectiveCamera(35, 1, 0.1, 1000);
    camera.position.set(4, -12, 3);
    camera.lookAt(4, 0, 3);
    camera.updateMatrixWorld();
    const layer = new LineDiagramLayer(camera);

    let s = h.store.get();
    const stepData = h.effects.currentLineDiagram()!;
    assert.equal(layer.sync(s, h.blobs, stepData), true);
    const allRiserCount = (layer.riserLines.geometry.getAttribute("position") as THREE.BufferAttribute).count;
    assert.equal(allRiserCount, 10 * 3 * 2, "all 10 beams drawn initially");

    // Perf 2a: Passing a restricted `visibleLineMap` (simulating stage element activation `elementMaskOf`
    // while `s.visibility.hidden` is still empty `[]`) must invalidate `LineDiagramLayer.sync`'s cache key:
    const subsetMap = new Int32Array([0, 1, 2]);
    assert.notEqual(visibleLineDigest(subsetMap, 10), visibleLineDigest(undefined, 10));
    assert.equal(layer.sync(s, h.blobs, stepData, null, false, subsetMap), true);
    const subsetRiserCount = (layer.riserLines.geometry.getAttribute("position") as THREE.BufferAttribute).count;
    assert.equal(subsetRiserCount, 3 * 3 * 2, "only the 3 active beams drawn when visibleLineMap changes");

    // Calling again with the same `subsetMap` is a cache hit (returns false):
    assert.equal(layer.sync(s, h.blobs, stepData, null, false, new Int32Array([0, 1, 2])), false);

    // Perf 2b: Dispatching `isolate` on a single beam declaration with `layer.sync` subscribed
    // updates `Effects.syncLineDiagram`, `drawnBeams`, and `LineDiagramLayer` in a single `buildLineDiagram3D` build:
    const unsubLayer = h.store.subscribe((cur) => {
      layer.sync(cur, h.blobs, h.effects.currentLineDiagram(), h.effects.currentDisplacement());
    });
    const firstBeam = s.mesh!.elements[0]!;
    resetLineDiagram3DBuildCount();
    h.store.dispatch({ type: "isolate", decls: [firstBeam] });
    await until(() => h.store.get().lineDiagram.drawnBeams.length === 1, "isolate updates drawnBeams to 1 beam");
    assert.equal(h.store.get().lineDiagram.drawnBeams[0]?.decl, firstBeam);
    assert.equal(
      getLineDiagram3DBuildCount(),
      1,
      `expected exactly 1 buildLineDiagram3D execution across isolate in Effects + LineDiagramLayer, got ${getLineDiagram3DBuildCount()}`,
    );

    resetLineDiagram3DBuildCount();
    h.store.dispatch({ type: "showAll" });
    await until(() => h.store.get().lineDiagram.drawnBeams.length === 10, "showAll restores all 10 beams");
    assert.equal(
      getLineDiagram3DBuildCount(),
      1,
      `expected exactly 1 buildLineDiagram3D execution across showAll in Effects + LineDiagramLayer, got ${getLineDiagram3DBuildCount()}`,
    );

    // Perf 2c (Finding 3): When `range === "history"` and `scan` is still pending/running (`clim.mode === "step"`),
    // dispatching `isolate` on the same step must NOT pass stale `[oldMin, oldMax]` into `buildLineDiagram3D`
    // and must execute `buildLineDiagram3D` only once across `Effects.syncLineDiagram` and `LineDiagramLayer.sync`:
    const ldKeysNow = lineDiagramKeysOf(h.store.get())!;
    h.store.dispatch({ type: "setLineDiagramRange", range: "history" });
    h.store.dispatch({
      type: "lineDiagramScan",
      scan: { key: ldKeysNow.scan, status: "running", done: 1, total: 10, min: null, max: null, error: null },
    });
    assert.equal(lineDiagramClimOf(h.store.get()).mode, "step");
    resetLineDiagram3DBuildCount();
    h.store.dispatch({ type: "isolate", decls: [firstBeam] });
    assert.equal(
      getLineDiagram3DBuildCount(),
      1,
      `expected 1 buildLineDiagram3D execution across Effects + LineDiagramLayer on isolate while history scan is pending, got ${getLineDiagram3DBuildCount()}`,
    );
    await until(() => h.store.get().lineDiagram.drawnBeams.length === 1, "isolate drawnBeams updated");
    h.store.dispatch({ type: "showAll" });
    await until(() => h.store.get().lineDiagram.drawnBeams.length === 10, "showAll restores 10 beams");

    // Perf 2d (Finding 3): When history scan finishes with raw-station extrema narrower than the current step's
    // E2b mid-span parabola peak (`stepRange`), `lineDiagramClimOf` widens `[clim.min, clim.max]` to include `stepRange`:
    const curStepKey = lineDiagramKeysOf(h.store.get())!.step;
    h.store.dispatch({ type: "lineDiagramStepRange", key: curStepKey, min: -10, max: 45 });
    h.store.dispatch({
      type: "lineDiagramScan",
      scan: { key: ldKeysNow.scan, status: "done", done: 10, total: 10, min: -5, max: 42.1875, error: null },
    });
    const widenedClim = lineDiagramClimOf(h.store.get());
    assert.equal(widenedClim.mode, "history");
    assert.equal(widenedClim.min, -10);
    assert.equal(widenedClim.max, 45);

    h.store.dispatch({ type: "setLineDiagramRange", range: "step" });
    await until(() => lineDiagramClimOf(h.store.get()).mode === "step", "back to step range");
    const stepAutoScale = h.store.get().lineDiagram.drawnBeams[0]!.autoScale;
    assert.ok(stepAutoScale > 0);

    // Perf 2e: Switching `range` to `"fixed"` with a 10x larger peak updates `drawnBeams[i].autoScale` in the store
    // and executes `buildLineDiagram3D` only once across `Effects.syncLineDiagram` and `LineDiagramLayer.sync`:
    const stepPeak = Math.max(
      Math.abs(h.store.get().lineDiagram.stepRange?.min ?? 1),
      Math.abs(h.store.get().lineDiagram.stepRange?.max ?? 1),
    );
    resetLineDiagram3DBuildCount();
    h.store.dispatch({
      type: "setLineDiagramRange",
      range: "fixed",
      fixed: { min: -10 * stepPeak, max: 10 * stepPeak },
    });
    await until(
      () => Math.abs(h.store.get().lineDiagram.drawnBeams[0]!.autoScale - stepAutoScale / 10) < 1e-12,
      "drawnBeams autoScale updated to match fixed clim",
    );
    assert.equal(
      getLineDiagram3DBuildCount(),
      1,
      `expected 1 buildLineDiagram3D execution when switching to fixed range, got ${getLineDiagram3DBuildCount()}`,
    );
    h.store.dispatch({ type: "setLineDiagramRange", range: "step" });
    await until(
      () => Math.abs(h.store.get().lineDiagram.drawnBeams[0]!.autoScale - stepAutoScale) < 1e-12,
      "drawnBeams autoScale restored on step range",
    );
    unsubLayer();

    // Perf 3: Switch to screen plane and orbit camera 10 times within a single frame:
    h.store.dispatch({ type: "setLineDiagramPlane", plane: "screen" });
    await until(() => h.store.get().lineDiagram.plane === "screen", "screen plane set");
    s = h.store.get();
    assert.equal(layer.sync(s, h.blobs, stepData), true);

    resetLineDiagram3DBuildCount();
    for (let i = 1; i <= 10; i++) {
      camera.position.set(4 + i, -12 + i * 0.5, 3 + i * 0.25);
      camera.lookAt(4, 0, 3);
      camera.updateMatrixWorld();
      assert.equal(layer.onCameraChange(), true);
    }
    // Before the animation frame flushes, 0 synchronous builds occurred during the 10 pointer events:
    assert.equal(getLineDiagram3DBuildCount(), 0, "onCameraChange defers geometry rebuild until animation frame flush");

    // Flushing the animation frame executes a single rebuild for the final camera orientation:
    assert.equal(layer.flushCameraChange(), true);
    assert.equal(getLineDiagram3DBuildCount(), 1, "flushCameraChange rebuilds exactly once per frame");
    assert.equal(layer.flushCameraChange(), false, "second flush with no camera movement is a no-op");

    // Orbiting away and returning to the exact same view direction before flush cancels the dirty flag:
    resetLineDiagram3DBuildCount();
    camera.position.set(20, -5, 8);
    camera.lookAt(4, 0, 3);
    camera.updateMatrixWorld();
    assert.equal(layer.onCameraChange(), true);
    camera.position.set(14, -7, 5.5);
    camera.lookAt(4, 0, 3);
    camera.updateMatrixWorld();
    assert.equal(layer.onCameraChange(), false, "returning to lastViewDirKey clears cameraDirty");
    assert.equal(layer.flushCameraChange(), false, "no rebuild needed when camera returned to initial direction");
    assert.equal(getLineDiagram3DBuildCount(), 0);

    // Switching back to "local" plane while cameraDirty is pending clears cameraDirty on sync:
    camera.position.set(20, -5, 8);
    camera.lookAt(4, 0, 3);
    camera.updateMatrixWorld();
    assert.equal(layer.onCameraChange(), true);
    h.store.dispatch({ type: "setLineDiagramPlane", plane: "local" });
    s = h.store.get();
    layer.sync(s, h.blobs, stepData);
    assert.equal(layer.flushCameraChange(), false, "sync clears pending cameraDirty when switching to local plane");

    layer.dispose();
  } finally {
    h.effects.dispose();
  }
});

test("Perf 4: LadrunoFile.readLineStations supports bounded stepRange hyperslab reads and ResultsHost.scan stops reading subsequent chunks when cancelled", async () => {
  await h5wasm.ready;
  const H5 = h5wasm as unknown as ResultsH5;
  const lf = openLadruno(H5, FRAME3D_LADRUNO);
  try {
    const full = lf.readLineStations("stage_0", "Mz");
    assert.ok(full !== null);
    assert.equal(full.nSteps, 3);
    const nRows = full.elementIndex.length;
    assert.equal(nRows, 30);

    // Read only step [1, 2) via hyperslab:
    const slice1 = lf.readLineStations("stage_0", "Mz", { stepRange: [1, 2] });
    assert.ok(slice1 !== null);
    assert.equal(slice1.nSteps, 1);
    assert.equal(slice1.values.length, nRows);
    assert.deepEqual(Array.from(slice1.values), Array.from(full.values.subarray(nRows, 2 * nRows)));

    // Verify companionStep ("Vy" <-> "Mz" and "Vz" <-> "My") is automatically attached symmetrically by readLineStationStep:
    const step2 = lf.readLineStationStep("stage_0", "Mz", 2);
    assert.ok(step2 !== null);
    assert.ok(step2.companionStep !== null && step2.companionStep !== undefined, "expected Vy companionStep on Mz step");
    assert.equal(step2.companionStep.component, "Vy");

    const step2Vy = lf.readLineStationStep("stage_0", "Vy", 2);
    assert.ok(step2Vy !== null);
    assert.ok(step2Vy.companionStep !== null && step2Vy.companionStep !== undefined, "expected Mz companionStep on Vy step");
    assert.equal(step2Vy.companionStep.component, "Mz");

    const step2My = lf.readLineStationStep("stage_0", "My", 2);
    assert.ok(step2My !== null);
    assert.ok(step2My.companionStep !== null && step2My.companionStep !== undefined, "expected Vz companionStep on My step");
    assert.equal(step2My.companionStep.component, "Vz");

    const step2Vz = lf.readLineStationStep("stage_0", "Vz", 2);
    assert.ok(step2Vz !== null);
    assert.ok(step2Vz.companionStep !== null && step2Vz.companionStep !== undefined, "expected My companionStep on Vz step");
    assert.equal(step2Vz.companionStep.component, "My");
  } finally {
    lf.close();
  }

  // Verify ResultsHost.scan (`source: "line"`, `chunkSteps: 1`) reads 1-step hyperslabs and stops on cancel:
  const progressCalls: { done: number; total: number }[] = [];
  const host = new ResultsHost(H5, h5wasm.ready);
  try {
    const openedRes = await host.open(FRAME3D_LADRUNO);
    assert.equal(openedRes.ok, true);

    const res = await host.scan(
      {
        path: FRAME3D_LADRUNO,
        stage: "stage_0",
        components: ["Mz"],
        magnitude: false,
        scanId: 99,
        source: "line",
      },
      (p) => {
        progressCalls.push({ done: p.done, total: p.total });
        if (p.done === 1) {
          host.cancelScan(p.scanId);
        }
      },
      { chunkSteps: 1, progressMs: 0 },
    );
    assert.equal(res.ok, false);
    if (!res.ok) {
      assert.equal(res.cancelled, true);
    }
    assert.equal(progressCalls.length, 1, "scan stopped after first 1-step chunk was cancelled");
  } finally {
    host.close();
  }
});
