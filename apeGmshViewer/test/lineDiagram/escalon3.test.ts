// Escalón 3 tests:
// - `.ladruno` sibling discovery in `src/main/pairing.ts`
// - `LineDiagramState` reducer transitions and selectors (`src/state/reduce.ts`, `src/state/lineDiagram.ts`)
// - `.ladruno` `<RESULT>/<label>` nodal field grouping & magnitude in `src/state/contour.ts`
// - `ResultsHost` + `Effects` integration on `fixtures/ladruno_frame3d.{h5,ladruno}`
// - Simultaneous `Contour` (on 2-D/3-D slabs & beam chords) + `Line diagram` (3-D ribbons on beams)
// - `LineDiagramLayer` (`src/renderer/lineDiagram.ts`): 3-D ribbon mesh, profile, risers,
//   fill toggle, selection dimming, deformed baseline, screen-plane camera update, clipping planes
// - Standalone `#line-diagram` panel (`mountLineDiagram`, `renderLineDiagram` in `src/panels/lineDiagram.ts`)

import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import * as THREE from "three";
import { pairSet } from "../../src/main/pairing.ts";
import { mountLineDiagram, renderLineDiagram } from "../../src/panels/lineDiagram.ts";
import { contourColours } from "../../src/renderer/contour.ts";
import { LineDiagramLayer } from "../../src/renderer/lineDiagram.ts";
import { CMAP_NAMES, MAGNITUDE, nodalFields } from "../../src/state/contour.ts";
import { displacementAxes } from "../../src/state/deform.ts";
import {
  colorForValue,
  defaultLineDiagramComponent,
  effectiveFlipSign,
  LINE_DIAGRAM_CMAP_NAMES,
  lineDiagramClimOf,
  lineDiagramKeysOf,
  signRampCss,
} from "../../src/state/lineDiagram.ts";
import { initialState, reduce } from "../../src/state/reduce.ts";
import { contourView, resultsPairing } from "../../src/state/selectors.ts";
import type { Event, ResultsStage, State } from "../../src/state/types.ts";
import { DARK, hexToRgb, LIGHT } from "../../src/theme/tokens.ts";
import { isHeadingToggleTarget, toggleCardCollapse } from "../../src/ui/collapse.ts";
import { opened, until } from "../contour/harness.ts";

const FRAME3D_MODEL = fileURLToPath(new URL("../../fixtures/ladruno_frame3d.h5", import.meta.url));
const FRAME3D_LADRUNO = fileURLToPath(new URL("../../fixtures/ladruno_frame3d.ladruno", import.meta.url));

// -- 1. Sibling auto-discovery (.h5 <-> .ladruno) -----------------------------

test("pairSet pairs model.h5 with its .ladruno sibling in both directions", () => {
  const fromModel = pairSet(FRAME3D_MODEL, existsSync);
  assert.equal(fromModel.model, FRAME3D_MODEL);
  assert.equal(fromModel.results, FRAME3D_LADRUNO);

  const fromLadruno = pairSet(FRAME3D_LADRUNO, existsSync);
  assert.equal(fromLadruno.model, FRAME3D_MODEL);
  assert.equal(fromLadruno.results, FRAME3D_LADRUNO);
});

// -- 2. Pure state & reducer transitions --------------------------------------

test("defaultLineDiagramComponent prefers Mz > My > Vy > Vz > N > T", () => {
  assert.equal(defaultLineDiagramComponent(["N", "Vy", "Mz"]), "Mz");
  assert.equal(defaultLineDiagramComponent(["N", "Vy", "My"]), "My");
  assert.equal(defaultLineDiagramComponent(["N", "Vy"]), "Vy");
  assert.equal(defaultLineDiagramComponent([]), null);
});

test("effectiveFlipSign defaults to true for bending moments (My, Mz) and false otherwise, with user override", () => {
  assert.equal(effectiveFlipSign("Mz", null), true);
  assert.equal(effectiveFlipSign("My", null), true);
  assert.equal(effectiveFlipSign("Vy", null), false);
  assert.equal(effectiveFlipSign("N", null), false);
  assert.equal(effectiveFlipSign("Mz", false), false);
  assert.equal(effectiveFlipSign("Vy", true), true);
  assert.equal(effectiveFlipSign({ component: "Mz", flipSign: null }), true);
  assert.equal(effectiveFlipSign({ component: "Mz", flipSign: false }), false);
});

test("signRampCss and colorForValue use active theme tokens (DARK vs LIGHT) and LINE_DIAGRAM_CMAP_NAMES covers sign + all 13 CMAP_NAMES", () => {
  assert.deepEqual(LINE_DIAGRAM_CMAP_NAMES, ["sign", ...CMAP_NAMES]);
  assert.equal(LINE_DIAGRAM_CMAP_NAMES.length, 14);
  const darkCss = signRampCss("dark");
  assert.ok(darkCss.includes(DARK.warn), `expected ${DARK.warn} in ${darkCss}`);
  assert.ok(darkCss.includes(DARK.ok), `expected ${DARK.ok} in ${darkCss}`);
  const lightCss = signRampCss("light");
  assert.ok(lightCss.includes(LIGHT.warn), `expected ${LIGHT.warn} in ${lightCss}`);
  assert.ok(lightCss.includes(LIGHT.ok), `expected ${LIGHT.ok} in ${lightCss}`);
  assert.deepEqual(colorForValue(0, "sign", -1, 1, 0, "dark"), hexToRgb(DARK.muted));
  assert.deepEqual(colorForValue(0, "sign", -1, 1, 0, "light"), hexToRgb(LIGHT.muted));
});

test("nodalFields groups .ladruno <RESULT>/<label> components and offers magnitude for translational families only", () => {
  const fields = nodalFields([
    "DISPLACEMENT/Ux",
    "DISPLACEMENT/Uy",
    "DISPLACEMENT/Uz",
    "ROTATION/Rx",
    "ROTATION/Ry",
    "ROTATION/Rz",
    "REACTION_FORCE/Rx",
    "REACTION_FORCE/Rz",
  ]);
  assert.deepEqual(
    fields.map((f) => [f.name, f.components.map((c) => `${c.label}=${c.datasets.join("+")}`)]),
    [
      [
        "DISPLACEMENT",
        [
          "magnitude=DISPLACEMENT/Ux+DISPLACEMENT/Uy+DISPLACEMENT/Uz",
          "x=DISPLACEMENT/Ux",
          "y=DISPLACEMENT/Uy",
          "z=DISPLACEMENT/Uz",
        ],
      ],
      ["ROTATION", ["x=ROTATION/Rx", "y=ROTATION/Ry", "z=ROTATION/Rz"]],
      [
        "REACTION_FORCE",
        ["magnitude=REACTION_FORCE/Rx+REACTION_FORCE/Rz", "x=REACTION_FORCE/Rx", "z=REACTION_FORCE/Rz"],
      ],
    ],
  );
  assert.equal(fields[0]!.components[0]!.key, MAGNITUDE);

  // Refused translational .ladruno axis withdraws magnitude:
  const refused = nodalFields(
    ["DISPLACEMENT/Ux", "DISPLACEMENT/Uy"],
    [{ component: "DISPLACEMENT/Uz", reason: "corrupt column" }],
  );
  assert.equal(refused[0]!.components.some((c) => c.magnitude), false);
  assert.match(refused[0]!.magnitudeRefused ?? "", /DISPLACEMENT\/Uz not read/);
});

test("reducer validates and updates LineDiagramState across all 10 events and resets flipSign on component change", () => {
  let s: State = initialState;
  s = reduce(s, { type: "setLineDiagram", on: true });
  assert.equal(s.lineDiagram.on, true);

  s = reduce(s, { type: "setLineDiagramComponent", component: "Mz" });
  assert.equal(s.lineDiagram.component, "Mz");
  assert.equal(s.lineDiagram.flipSign, null);

  s = reduce(s, { type: "setLineDiagramFlipSign", flipSign: false });
  assert.equal(s.lineDiagram.flipSign, false);

  // Switching component resets flipSign to null so the new component uses its natural convention:
  s = reduce(s, { type: "setLineDiagramComponent", component: "Vy" });
  assert.equal(s.lineDiagram.component, "Vy");
  assert.equal(s.lineDiagram.flipSign, null);

  s = reduce(s, { type: "setLineDiagramScale", scale: 2.5 });
  assert.equal(s.lineDiagram.scale, 2.5);
  assert.throws(() => reduce(s, { type: "setLineDiagramScale", scale: 0 }), RangeError);
  assert.throws(() => reduce(s, { type: "setLineDiagramScale", scale: -1 }), RangeError);
  assert.throws(() => reduce(s, { type: "setLineDiagramScale", scale: Number.NaN }), RangeError);

  s = reduce(s, { type: "setLineDiagramPlane", plane: "screen" });
  assert.equal(s.lineDiagram.plane, "screen");

  s = reduce(s, { type: "setLineDiagramCmap", cmap: "coolwarm" });
  assert.equal(s.lineDiagram.cmap, "coolwarm");

  s = reduce(s, { type: "setLineDiagramFill", fill: false });
  assert.equal(s.lineDiagram.fill, false);

  s = reduce(s, { type: "setLineDiagramFlipSign", flipSign: true });
  assert.equal(s.lineDiagram.flipSign, true);

  s = reduce(s, { type: "setLineDiagramRange", range: "fixed", fixed: { min: -50, max: 50 } });
  assert.equal(s.lineDiagram.range, "fixed");
  assert.deepEqual(s.lineDiagram.fixed, { min: -50, max: 50 });
  assert.throws(
    () => reduce(s, { type: "setLineDiagramRange", range: "fixed", fixed: { min: 10, max: -10 } }),
    RangeError,
  );
});

test("displacementAxes respects stage.refusedNodal for .ladruno DISPLACEMENT datasets", () => {
  const stage: ResultsStage = {
    id: "MODEL_STAGE[1]",
    name: "gravity",
    kind: "static",
    time: [0, 1],
    nodal: [],
    refusedNodal: [{ component: "DISPLACEMENT", reason: "TIME length 1 != 2" }],
    gauss: [],
    unread: [],
    elementIndexSpace: null,
    lineStations: [],
  };
  const verdict = displacementAxes(stage, { ndm: 3, source: "/meta/ndm" });
  assert.equal(verdict.ok, false);
  if (!verdict.ok) {
    assert.equal(verdict.kind, "refused");
    assert.match(verdict.reason, /DISPLACEMENT not read/);
  }
});

// -- 3. End-to-end ResultsHost + Effects + LineDiagramLayer + simultaneous Contour --

test("ladruno_frame3d: Effects loads 2-bay/2-story .ladruno with floor slabs, runs Contour and LineDiagram simultaneously, and LineDiagramLayer builds 3-D geometry", async () => {
  const h = await opened(FRAME3D_MODEL, FRAME3D_LADRUNO);
  try {
    let s = h.store.get();
    assert.equal(resultsPairing(s).state, "paired");
    assert.equal(s.results?.sourceType, "ladruno");
    assert.equal(s.results?.elementIndexSpace, null);
    assert.deepEqual(s.results?.stages[0]?.lineStations, ["N", "Vy", "Vz", "T", "My", "Mz"]);
    // #2128: the fixture follows fork schema §7.1.1, translation and rotation in two groups.
    assert.deepEqual(s.results?.stages[0]?.nodal, ["DISPLACEMENT/Ux", "DISPLACEMENT/Uy", "DISPLACEMENT/Uz", "ROTATION/Rx", "ROTATION/Ry", "ROTATION/Rz"]);
    // 2-bay / 2-story frame has 10 line2 beams/columns and 4 quad4 floor slabs:
    assert.equal(s.mesh?.counts.lineCells, 10);
    assert.equal(s.mesh?.counts.faceCells, 4);
    // Auto-picked default component is Mz:
    assert.equal(s.lineDiagram.component, "Mz");

    // Switch to results stage 0, step 2 and enable BOTH Contour AND Line diagram simultaneously:
    h.store.dispatch({ type: "setPhase", at: { kind: "results", stage: 0, step: 2 } });
    h.store.dispatch({ type: "setColourBy", by: "contour" });
    h.store.dispatch({ type: "setLineDiagram", on: true });

    await until(() => {
      const cur = h.store.get();
      const ldKeys = lineDiagramKeysOf(cur);
      const cv = contourView(cur);
      return (
        ldKeys !== null &&
        h.effects.currentLineDiagram()?.key === ldKeys.step &&
        cur.lineDiagram.stepRange?.key === ldKeys.step &&
        cur.lineDiagram.scan?.key === ldKeys.scan &&
        cur.lineDiagram.scan.status === "done" &&
        cv.state === "ready" &&
        cv.stepMeasured &&
        cur.contour.scan?.status === "done"
      );
    }, "simultaneous DISPLACEMENT contour and Mz lineDiagram step 2 + history scans");

    s = h.store.get();
    const cv = contourView(s);
    assert.equal(cv.state, "ready");
    if (cv.state === "ready") {
      assert.equal(cv.field, "DISPLACEMENT");
      assert.equal(cv.component.key, MAGNITUDE);
      assert.deepEqual(cv.component.datasets, ["DISPLACEMENT/Ux", "DISPLACEMENT/Uy", "DISPLACEMENT/Uz"], "the magnitude is of the translation only");
      assert.equal(cv.clim.mode, "history");
      assert.ok((cv.clim.max ?? 0) > 0, "DISPLACEMENT magnitude history max > 0");
      const lineMap = Int32Array.from({ length: s.mesh!.linePositions.shape[0]! }, (_, i) => i);
      const triMap = Int32Array.from({ length: s.mesh!.triPositions.shape[0]! }, (_, i) => i);
      const cc = contourColours(s, cv, s.mesh!, h.blobs, h.effects.currentStep(), lineMap, triMap);
      assert.ok(cc !== null, "contourColours computed on quad4 slabs and line2 beams simultaneously with lineDiagram");
      assert.equal(cc.tri.length, s.mesh!.triPositions.shape[0]! * 3 * 3, "8 triangles x 3 vertices x RGB");
      assert.equal(cc.line.length, s.mesh!.linePositions.shape[0]! * 2 * 3, "10 lines x 2 vertices x RGB");
    }

    const stepData = h.effects.currentLineDiagram();
    assert.ok(stepData !== null);
    assert.equal(stepData.step.component, "Mz");
    assert.equal(stepData.step.elementIndex.length, 30, "10 beams x 3 stations");

    // At step 2 (load factor 1.0), peak |Mz| is 60, so stepRange is [-45, 60] and history scan is [-45, 60]:
    assert.equal(s.lineDiagram.stepRange?.min, -45);
    assert.equal(s.lineDiagram.stepRange?.max, 60);
    assert.equal(s.lineDiagram.scan?.min, -45);
    assert.equal(s.lineDiagram.scan?.max, 60);
    assert.equal(s.lineDiagram.notice, null);

    // Hiding the Girder group recomputes stepRange over only the drawn Columns stations ([-45, 45]):
    h.store.dispatch({ type: "setHidden", decls: ["mesh/physical_group/Girder"], hidden: true });
    await until(() => h.store.get().lineDiagram.stepRange?.max === 45, "stepRange recomputed for visible Columns only");
    assert.equal(h.store.get().lineDiagram.stepRange?.min, -45);
    assert.equal(h.store.get().lineDiagram.stepRange?.max, 45);
    h.store.dispatch({ type: "setHidden", decls: ["mesh/physical_group/Girder"], hidden: false });
    await until(() => h.store.get().lineDiagram.stepRange?.max === 60, "stepRange restored when Girder shown");

    s = h.store.get();
    const clim = lineDiagramClimOf(s);
    assert.equal(clim.min, -45);
    assert.equal(clim.max, 60);

    // Sync LineDiagramLayer:
    const camera = new THREE.PerspectiveCamera(35, 1, 0.1, 1000);
    camera.position.set(4, -12, 3);
    camera.lookAt(4, 0, 3);
    camera.updateMatrixWorld();

    const layer = new LineDiagramLayer(camera);
    assert.equal((layer.mesh.material as THREE.MeshBasicMaterial).opacity, 0.55);
    const changed = layer.sync(s, h.blobs, stepData, h.effects.currentDisplacement());
    assert.equal(changed, true);
    assert.equal(layer.root.visible, true);
    assert.equal(layer.mesh.visible, true);

    const posAttr = layer.mesh.geometry.getAttribute("position") as THREE.BufferAttribute;
    const colAttr = layer.mesh.geometry.getAttribute("color") as THREE.BufferAttribute;
    assert.ok(posAttr.count > 0, "ribbon mesh has triangles");
    assert.equal(posAttr.count, colAttr.count);

    const profPos = layer.profileLines.geometry.getAttribute("position") as THREE.BufferAttribute;
    const riserPos = layer.riserLines.geometry.getAttribute("position") as THREE.BufferAttribute;
    assert.equal(profPos.count, 10 * 2 * 2, "10 beams x 2 spans x 2 vertices per segment");
    assert.equal(riserPos.count, 10 * 3 * 2, "10 beams x 3 stations x 2 vertices per riser");

    // Calling sync again with identical inputs is a no-op:
    assert.equal(layer.sync(s, h.blobs, stepData, h.effects.currentDisplacement()), false);

    // Toggling fill hides the ribbon mesh while keeping profile and risers visible:
    h.store.dispatch({ type: "setLineDiagramFill", fill: false });
    s = h.store.get();
    assert.equal(layer.sync(s, h.blobs, stepData, h.effects.currentDisplacement()), true);
    assert.equal(layer.mesh.visible, false);
    assert.equal(layer.root.visible, true);

    // Selecting one beam dims unselected beams in vertex colors:
    const unselectedColors = new Float32Array(
      (layer.mesh.geometry.getAttribute("color") as THREE.BufferAttribute).array as Float32Array,
    );
    const firstBeamDecl = s.mesh!.elements[0]!;
    h.store.dispatch({ type: "select", pick: { decl: firstBeamDecl, at: null, nodes: null } });
    s = h.store.get();
    assert.equal(layer.sync(s, h.blobs, stepData, h.effects.currentDisplacement()), true);
    const dimmedColors = (layer.mesh.geometry.getAttribute("color") as THREE.BufferAttribute).array as Float32Array;
    const lastIdx = dimmedColors.length - 3;
    assert.ok(
      Math.abs(dimmedColors[lastIdx]! - unselectedColors[lastIdx]! * 0.07) < 1e-5,
      `expected unselected beam vertex to be dimmed by 0.07 (${dimmedColors[lastIdx]} vs ${unselectedColors[lastIdx]})`,
    );
    h.store.dispatch({ type: "clearSelection" });

    // Enabling deformed shape shifts the diagram baseline onto the deformed chord:
    h.store.dispatch({ type: "setDeform", on: true });
    h.store.dispatch({ type: "setDeformScale", scale: 10 });
    await until(() => h.effects.currentDisplacement() !== null, "displacement computed");
    s = h.store.get();
    const riserBefore = new Float32Array(
      (layer.riserLines.geometry.getAttribute("position") as THREE.BufferAttribute).array as Float32Array,
    );
    assert.equal(layer.sync(s, h.blobs, stepData, h.effects.currentDisplacement()), true);
    const riserAfter = (layer.riserLines.geometry.getAttribute("position") as THREE.BufferAttribute).array as Float32Array;
    let maxDiff = 0;
    for (let i = 0; i < riserBefore.length; i++) {
      maxDiff = Math.max(maxDiff, Math.abs(riserAfter[i]! - riserBefore[i]!));
    }
    assert.ok(maxDiff > 0.1, `deformed baseline moved riser coordinates by ${maxDiff}`);

    // Screen-plane mode updates when camera direction changes:
    h.store.dispatch({ type: "setLineDiagramPlane", plane: "screen" });
    s = h.store.get();
    assert.equal(layer.sync(s, h.blobs, stepData, h.effects.currentDisplacement()), true);
    camera.position.set(10, 0, 2);
    camera.lookAt(3, 0, 2);
    camera.updateMatrixWorld();
    assert.equal(layer.onCameraChange(), true);

    // Theme switch rebuilds geometry when theme changes:
    layer.setTheme("light");
    assert.equal(layer.sync(s, h.blobs, stepData, h.effects.currentDisplacement()), true);
    assert.equal(layer.sync(s, h.blobs, stepData, h.effects.currentDisplacement()), false);
    layer.setTheme("dark");

    // Clipping planes forward to all 3 materials:
    const clipPlanes = [new THREE.Plane(new THREE.Vector3(0, 0, -1), 2)];
    layer.setClippingPlanes(clipPlanes);
    assert.equal((layer.mesh.material as THREE.Material).clippingPlanes, clipPlanes);
    assert.equal((layer.profileLines.material as THREE.Material).clippingPlanes, clipPlanes);
    assert.equal((layer.riserLines.material as THREE.Material).clippingPlanes, clipPlanes);

    // Switching component to Vy fetches Vy step and scans Vy history while keeping Contour ready:
    h.store.dispatch({ type: "setLineDiagramComponent", component: "Vy" });
    await until(() => {
      const cur = h.store.get();
      const keys = lineDiagramKeysOf(cur);
      return (
        keys !== null &&
        h.effects.currentLineDiagram()?.key === keys.step &&
        cur.lineDiagram.scan?.key === keys.scan &&
        cur.lineDiagram.scan.status === "done"
      );
    }, "ladruno_frame3d Vy step 2 and history scan");
    s = h.store.get();
    assert.equal(h.effects.currentLineDiagram()?.step.component, "Vy");
    assert.equal(s.lineDiagram.scan?.min, -30);
    assert.equal(s.lineDiagram.scan?.max, 30);
    assert.equal(contourView(s).state, "ready", "Contour remains ready when Line diagram switches component");

    layer.dispose();
  } finally {
    h.effects.dispose();
  }
});

// -- 4. Standalone #line-diagram panel (mountLineDiagram, renderLineDiagram) --

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
  get firstElementChild(): FakeEl | null {
    return this.children.find((c) => c.tagName !== "#text") ?? null;
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
  closest(selector: string): FakeEl | null {
    const tags = new Set(selector.split(",").map((s) => s.trim().toLowerCase()));
    let cur: FakeEl | null = this;
    while (cur) {
      if (tags.has(cur.tagName.toLowerCase())) return cur;
      cur = cur.parentElement;
    }
    return null;
  }
  contains(other: FakeEl | null): boolean {
    let cur = other;
    while (cur) {
      if (cur === this) return true;
      cur = cur.parentElement;
    }
    return false;
  }
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
  trigger(ev: string): void {
    for (const fn of this.listeners.get(ev) ?? []) fn();
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

test("mountLineDiagram and renderLineDiagram render standalone #line-diagram panel and dispatch store events", async () => {
  const h = await opened(FRAME3D_MODEL, FRAME3D_LADRUNO);
  const prevDoc = (globalThis as { document?: unknown }).document;
  const aside = new FakeEl("aside");
  const docEl = { dataset: { theme: "dark" } as Record<string, string | undefined> };
  (globalThis as { document?: unknown }).document = {
    documentElement: docEl,
    createElement: (tag: string) => new FakeEl(tag),
    getElementById: (id: string) => (id === "line-diagram" ? aside : null),
  };
  let unmount: (() => void) | null = null;
  try {
    unmount = mountLineDiagram(h.store);
    // On mesh phase, #line-diagram panel is hidden and renderLineDiagram returns null:
    assert.equal(aside.hidden, true);
    assert.equal(renderLineDiagram(h.store.get()), null);

    // On results phase with lineStations, #line-diagram becomes visible and its first child is <h3> (collapsible):
    h.store.dispatch({ type: "setPhase", at: { kind: "results", stage: 0, step: 1 } });
    assert.equal(aside.hidden, false);
    const headerH3 = aside.find("line-diagram-header");
    assert.equal(headerH3?.textContent, "Line diagram");
    assert.equal(aside.firstElementChild, headerH3);
    assert.equal(
      isHeadingToggleTarget(headerH3 as unknown as Element, aside as unknown as HTMLElement),
      true,
    );
    assert.equal(toggleCardCollapse(aside as unknown as HTMLElement), true);
    assert.equal(aside.dataset["collapsed"], "true");
    assert.equal(toggleCardCollapse(aside as unknown as HTMLElement), false);
    assert.equal(aside.dataset["collapsed"], undefined);

    const dispatched: Event[] = [];
    const unlisten: (() => void)[] = [];
    let dom = renderLineDiagram(
      h.store.get(),
      (e) => {
        dispatched.push(e);
        h.store.dispatch(e);
      },
      unlisten,
    ) as unknown as FakeEl | null;
    assert.ok(dom !== null);
    assert.equal(dom.find("line-diagram-header")?.textContent, "Line diagram");

    // Click toggle button -> turns lineDiagram on:
    const toggleBtn = dom.find("line-diagram-toggle");
    assert.ok(toggleBtn !== undefined);
    assert.equal(toggleBtn.textContent, "diagram");
    toggleBtn.trigger("click");
    assert.deepEqual(dispatched.at(-1), { type: "setLineDiagram", on: true });

    // Re-render with lineDiagram.on === true (component is Mz, where default flipSign is true):
    for (const u of unlisten.splice(0)) u();
    dom = renderLineDiagram(
      h.store.get(),
      (e) => {
        dispatched.push(e);
        h.store.dispatch(e);
      },
      unlisten,
    ) as unknown as FakeEl | null;
    assert.ok(dom !== null);
    assert.equal(dom.find("line-diagram-title")?.textContent, "section_force · Mz (bending_moment_z)");

    // Flip sign button (for Mz, flipped is true by default, so clicking it dispatches flipSign: false):
    const flipBtn = dom.find("line-diagram-flip");
    assert.ok(flipBtn !== undefined);
    flipBtn.trigger("click");
    assert.deepEqual(dispatched.at(-1), { type: "setLineDiagramFlipSign", flipSign: false });

    // Field & Component selects:
    const fieldSelect = dom.find("line-diagram-field");
    assert.ok(fieldSelect !== undefined);
    assert.equal(fieldSelect.tagName, "select");
    assert.equal(fieldSelect.children[0]?.value, "section_force");

    const compSelect = dom.find("line-diagram-component");
    assert.ok(compSelect !== undefined);
    assert.equal(compSelect.tagName, "select");
    assert.equal(compSelect.children[0]?.tagName, "optgroup");
    assert.equal(compSelect.children[0]?.label, "section_force");
    compSelect.value = "Vy";
    compSelect.trigger("change");
    assert.deepEqual(dispatched.at(-1), { type: "setLineDiagramComponent", component: "Vy" });

    // Plane buttons: click screen plane:
    const screenBtn = dom.find("line-diagram-plane-screen");
    assert.ok(screenBtn !== undefined);
    assert.equal(screenBtn.textContent, "screen plane");
    screenBtn.trigger("click");
    assert.deepEqual(dispatched.at(-1), { type: "setLineDiagramPlane", plane: "screen" });

    // Scale input and scaleLabel:
    const scaleInput = dom.find("line-diagram-scale");
    assert.ok(scaleInput !== undefined);
    assert.equal(dom.find("line-diagram-scale-label")?.textContent, "true scale (×1)");
    scaleInput.value = "3.5";
    scaleInput.trigger("change");
    assert.deepEqual(dispatched.at(-1), { type: "setLineDiagramScale", scale: 3.5 });

    // Colormap select & options ("sign" + 13 CMAP_NAMES):
    const cmapSelect = dom.find("line-diagram-cmap");
    assert.ok(cmapSelect !== undefined);
    assert.equal(cmapSelect.tagName, "select");
    assert.equal(cmapSelect.children.length, LINE_DIAGRAM_CMAP_NAMES.length);
    const viridisBtn = dom.find("line-diagram-cmap-viridis");
    assert.ok(viridisBtn !== undefined);
    viridisBtn.trigger("click");
    assert.deepEqual(dispatched.at(-1), { type: "setLineDiagramCmap", cmap: "viridis" });
    cmapSelect.value = "Spectral";
    cmapSelect.trigger("change");
    assert.deepEqual(dispatched.at(-1), { type: "setLineDiagramCmap", cmap: "Spectral" });

    // Notice rendering when s.lineDiagram.notice is populated:
    h.store.dispatch({
      type: "lineDiagramStepRange",
      key: lineDiagramKeysOf(h.store.get())!.step,
      min: -45,
      max: 60,
      notice: "1 3-D beam skipped: Mz in local plane requires LOCAL_AXES or geomTransf vecxz (never guessed in 3-D).",
    });
    for (const u of unlisten.splice(0)) u();
    dom = renderLineDiagram(
      h.store.get(),
      (e) => {
        dispatched.push(e);
        h.store.dispatch(e);
      },
      unlisten,
    ) as unknown as FakeEl | null;
    assert.ok(dom !== null);
    assert.match(dom.find("line-diagram-notice")?.textContent ?? "", /1 3-D beam skipped/);

    // Switching to mesh phase (no lineStations) clears stale stepRange, scan, and notice, and hides #line-diagram:
    h.store.dispatch({ type: "setPhase", at: { kind: "mesh" } });
    assert.equal(h.store.get().lineDiagram.notice, null);
    assert.equal(aside.hidden, true);

    for (const u of unlisten) u();
  } finally {
    unmount?.();
    (globalThis as { document?: unknown }).document = prevDoc;
    h.effects.dispose();
  }
});

test("Escalón 3: style.css prevents #banner/#line-diagram overlap and component select truncation", () => {
  const css = readFileSync(new URL("../../src/renderer/style.css", import.meta.url), "utf8");
  assert.match(css, /#banner:not\(\[hidden\]\)\s*~\s*#line-diagram/);
  assert.match(css, /\.line-diagram-pick\s+\.line-diagram-field\s*\{\s*flex:\s*0\s+0\s+auto;/);
});

