// Renderer entry: wires the store, the BlobStore, the effects, the viewport
// and the panels, then runs the mode main asked for (view, measure or
// capture; ADR 0128's render and describe run in ./agentRender.ts).

import * as THREE from "three";
import { Effects, userCombinationColumn, type Bridge } from "../effects.ts";
import { mountBanner } from "../panels/banner.ts";
import { mountEmpty } from "../panels/empty.ts";
import { mountHeader } from "../panels/header.ts";
import { mountInspector } from "../panels/inspector.ts";
import { mountLegend } from "../panels/legend.ts";
import { mountLineDiagram } from "../panels/lineDiagram.ts";
import { mountLoads } from "../panels/loads.ts";
import { mountPrincipal } from "../panels/principal.ts";
import { mountReactions } from "../panels/reactions.ts";
import { mountModalTable } from "../panels/modalTable.ts";
import { mountOfferBadge, mountOffers } from "../panels/offers.ts";
import { mountPhase } from "../panels/phase.ts";
import { mountPlanes } from "../panels/planes.ts";
import { mountSources } from "../panels/sources.ts";
import { mountSpotlight } from "../panels/spotlight.ts";
import { mountWindows } from "../panels/windows/frame.ts";
import { BlobStore } from "../state/blobs.ts";
import { offsetRange } from "../state/clip.ts";
import { contourKeysOf, gaussFields } from "../state/contour.ts";
import { chainOf, contourFieldsOf, contourView, deformOf, deformView, elementVerdictOf } from "../state/selectors.ts";
import { userEventOf, userPathOf, userReadable } from "../state/reduce.ts";
import { Store } from "../state/store.ts";
import { WINDOW_KINDS, type DeclPath, type GaussMode, type UserReadIn } from "../state/types.ts";
import { DEFAULT_PANEL_IDS, mountCollapsibleChrome } from "../ui/collapse.ts";
import { OfferRunner, snapshotManifest } from "../spec/snapshot.ts";
import { APP_LAYERS, type AppLayer } from "../spec/spec.ts";
import type { OfferIn } from "../state/types.ts";
import { onScreenOf, runAgentMode, type Layout, type RenderBridge } from "./agentRender.ts";
import { CombinationFeed, contourKey, contourShownKey, elementContourKey } from "./contour.ts";
import { tipPoint } from "./gizmo.ts";
import { zoomToward } from "./navigation.ts";
import { mountViewCube } from "./viewcube.ts";
import { Viewport } from "./viewport.ts";

const bridge = (window as unknown as { viewer: Bridge }).viewer;

const store = new Store();
const blobs = new BlobStore();
const effects = new Effects(store, blobs, bridge);
// VA1: the theme is the store's; the effect styles the page with it (and does so now).
effects.setThemeTarget(document);
const viewport = new Viewport(document.getElementById("viewport")!, store, blobs);
effects.setFrameTarget(viewport);
// V4c1: the contour colours from the step the effects fetched (never kept in the store).
viewport.setStepSource(() => effects.currentStep());
// V4d: the deformed shape moves the drawn vertices by the displacement the effects computed for the step shown.
viewport.setDisplacementSource(() => effects.currentDisplacement());
// 3-D line diagrams: ribbon + profile + risers from the step the effects fetched.
viewport.setLineDiagramSource(() => effects.currentLineDiagram());
viewport.setStationsSource(() => effects.stationsOfStep());
// VU7b-2 (ADR 0130 D4): the User Combination shown, read through main's `user:readCase` and combined by VU4's
// evaluator (effects.ts `userCombinationColumn`); its values stay in the feed, as a step's stay in the effects.
const combinations = new CombinationFeed(store, (req) => userCombinationColumn(bridge, req));
viewport.setCombinationSource(() => combinations.held);
const panels = [
  combinations.start(),
  // VA5 (ADR 0128 D5): first, so its box opens the right rail above the inspector.
  mountOffers(store),
  // ADR 0130 D6 (VU7a): the .user.h5 modal table, below the offers, while a mode is shown.
  mountModalTable(store),
  mountHeader(store),
  mountBanner(store),
  mountLineDiagram(store, (fn) => viewport.onViewChange(fn)),
  mountLoads(store, (fn) => viewport.onViewChange(fn)),
  mountReactions(store, (fn) => viewport.onViewChange(fn)),
  mountPrincipal(store),
  mountLegend(store, (fn) => viewport.onViewChange(fn)),
  mountInspector(store),
  mountEmpty(store),
  mountPhase(store),
  mountSources(store),
  mountWindows(store),
  mountSpotlight(store),
];
// V4e1: the section planes panel; its slider box and the view direction come from the viewport.
panels.push(mountPlanes(store, { box: () => viewport.sliderBox(), viewNormal: () => viewport.viewNormal(), sliding: (on) => viewport.setSliding(on) }));
panels.push(
  mountViewCube({
    visible: () => {
      const s = store.get();
      return s.mesh !== null || s.geometry !== null;
    },
    heading: () => viewport.nav.heading,
    projection: () => viewport.projection,
    setProjection: (mode) => viewport.setProjection(mode),
    // VA1: the theme and every ViewCube click, drag and button go through the store (setTheme, setCamera).
    theme: () => store.get().theme,
    setTheme: (mode) => store.dispatch({ type: "setTheme", theme: mode }),
    snapTo: (h) => store.dispatch({ type: "setCamera", to: { yaw: h.yaw, tilt: h.tilt } }),
    orbit: (dYaw, dTilt) => {
      // A turntable orbit about the pivot is a turn to the heading it reaches (navigation.ts orbitAbout).
      const h = viewport.nav.heading;
      store.dispatch({ type: "setCamera", to: { yaw: h.yaw + dYaw, tilt: h.tilt + dTilt } });
    },
    home: () => store.dispatch({ type: "setCamera", to: "home" }),
    subscribe: (fn) => viewport.onViewChange(fn),
  }),
);
const chrome = mountCollapsibleChrome({
  panelIds: [...DEFAULT_PANEL_IDS, "line-diagram", "loads", "reactions", "principal", "offers", "modal-table"],
  visible: () => {
    const s = store.get();
    return s.mesh !== null || s.provenance.length > 0;
  },
  onRailToggle: (collapsed) => viewport.setRailCollapsed(collapsed),
});
// VA5: a collapsed rail hides the offers panel; its toggle carries the count (the toggle exists once the chrome is mounted).
panels.push(mountOfferBadge(store));
const unsubChrome = store.subscribe(() => chrome.syncVisibility());
panels.push(() => {
  unsubChrome();
  chrome.dispose();
});
window.addEventListener("beforeunload", () => {
  for (const d of panels) d();
  effects.dispose();
  viewport.dispose();
});

const nextFrame = () => new Promise<number>((r) => requestAnimationFrame(r));
/** Resolves once the frame after the current DOM change has been painted. */
const painted = () => new Promise<void>((r) => requestAnimationFrame(() => setTimeout(r, 0)));

const BEAM_TYPES = new Set(["dispBeamColumn", "forceBeamColumn", "elasticBeamColumn"]);

/**
 * The element a scripted click targets: a beam-column if any; else an element
 * whose chain resolves with at least one link; else the middle drawn element.
 * A bounded sample (from the middle outwards) keeps this cheap on big models.
 * With `pickType`, every drawn element is a candidate (not only the sample),
 * and the first of that OpenSees type whose chain resolves wins; none found
 * raises, naming the type.
 */
function demoTarget(pickType: string | null = null): DeclPath | null {
  const s = store.get();
  if (!s.mesh) return null;
  const drawn = s.mesh.elements;
  if (pickType !== null) {
    for (const p of drawn) {
      if (s.decls[p]?.type !== pickType) continue;
      const c = chainOf(s, p);
      if (c && c.problems.length === 0) return p;
    }
    throw new Error(`--pick=${pickType}: no drawn element of that type with a resolved chain`);
  }
  const step = Math.max(1, Math.floor(drawn.length / 400));
  const candidates: DeclPath[] = [];
  for (let i = Math.floor(drawn.length / 2); i < drawn.length; i += step) candidates.push(drawn[i]!);
  for (let i = 0; i < Math.floor(drawn.length / 2); i += step) candidates.push(drawn[i]!);
  const chains = candidates.map((p) => ({ p, c: chainOf(s, p)! }));
  const beam = chains.find(({ c }) => BEAM_TYPES.has(c.root.type) && c.problems.length === 0);
  const linked = chains.find(({ c }) => c.root.children.length > 0 && c.problems.length === 0);
  return (beam ?? linked)?.p ?? candidates[0] ?? null;
}

/** Click at the target's projected midpoint through the same pick path a mouse uses. */
async function scriptedClick(target: DeclPath) {
  const pt = viewport.screenPointOf(target);
  if (!pt) throw new Error("scripted click: target is not drawn");
  const t = performance.now();
  const pick = viewport.pickAt(pt.x, pt.y);
  if (!pick) throw new Error("scripted click: pick returned nothing");
  const tPick = performance.now();
  store.dispatch({ type: "select", pick });
  const tState = performance.now();
  await painted();
  const end = performance.now();
  return {
    ms: end - t,
    split: { pickMs: tPick - t, stateAndDomMs: tState - tPick, paintMs: end - tState },
    pickedTarget: pick.decl === target,
    decl: pick.decl,
  };
}

async function measure(t0: number, startup: { appReadyMs: number; configMs: number }) {
  await nextFrame();
  viewport.renderNow();
  await painted();
  const firstFrameMs = Date.now() - t0;
  const s = store.get();
  const model = s.artifacts.model!;
  const mesh = s.mesh!;

  // Scripted orbit: one turn about the vertical (Z) axis through the fitted
  // centre, 6 s, through the same turntable code a right-drag uses.
  const ORBIT_MS = 6000;
  const target = viewport.nav.target.clone();
  const deltas: number[] = [];
  const renderMs: number[] = [];
  viewport.continuous = true;
  const start = await nextFrame();
  let last = start;
  let turned = 0;
  for (;;) {
    const now = await nextFrame();
    deltas.push(now - last);
    last = now;
    const ang = Math.min(1, (now - start) / ORBIT_MS) * Math.PI * 2;
    viewport.nav.orbit(target, ang - turned, 0);
    turned = ang;
    viewport.renderNow();
    renderMs.push(viewport.lastRenderMs);
    if (now - start >= ORBIT_MS) break;
  }
  viewport.continuous = false;
  viewport.renderNow();
  await painted();

  const median = (a: number[]) => {
    const b = [...a].sort((x, y) => x - y);
    return b.length ? b[Math.floor(b.length / 2)]! : NaN;
  };
  const tgt = demoTarget();
  const click = tgt ? await scriptedClick(tgt) : null;
  const chain = click ? chainOf(store.get(), click.decl) : null;
  // A second, shorter orbit with the selection (and its halo pass) active.
  const SELECTED_ORBIT_MS = 2000;
  const selDeltas: number[] = [];
  let orbitSelected: { durationMs: number; frames: number; medianFps: number } | null = null;
  if (click) {
    viewport.continuous = true;
    const s0 = await nextFrame();
    let prev = s0, done = 0;
    for (;;) {
      const now = await nextFrame();
      selDeltas.push(now - prev);
      prev = now;
      const ang = Math.min(1, (now - s0) / SELECTED_ORBIT_MS) * Math.PI * 2;
      viewport.nav.orbit(target, ang - done, 0);
      done = ang;
      viewport.renderNow();
      if (now - s0 >= SELECTED_ORBIT_MS) break;
    }
    viewport.continuous = false;
    orbitSelected = { durationMs: prev - s0, frames: selDeltas.length, medianFps: 1000 / median(selDeltas) };
  }
  const metrics = await bridge.metrics();
  return {
    file: model.path,
    sizeBytes: model.sizeBytes,
    nodes: model.counts.nodes,
    cells: model.counts.cells,
    opsOnlyElements: mesh.counts.opsOnly,
    drawn: { segments: mesh.linePositions.shape[0], triangles: mesh.triPositions.shape[0] },
    readMs: model.readMs,
    firstFrameMs,
    startup: { electronReadyMs: startup.appReadyMs, rendererUpMs: startup.configMs },
    orbit: {
      durationMs: last - start,
      frames: deltas.length,
      medianFps: 1000 / median(deltas),
      medianRenderCpuMs: median(renderMs),
    },
    orbitSelected,
    memory: metrics,
    inspector: click
      ? {
          fillMs: click.ms,
          split: click.split,
          pickedTarget: click.pickedTarget,
          root: chain?.root.name,
          type: chain?.root.type,
          links: chain ? countLinks(chain.root) : 0,
          problems: chain?.problems ?? [],
        }
      : null,
    gpu: viewport.gpuName(),
    pixelRatio: window.devicePixelRatio,
    viewport: [viewport.renderer.domElement.width, viewport.renderer.domElement.height],
    warnings: [...model.warnings],
    three: THREE.REVISION,
  };
}

function countLinks(n: { children: { children: unknown[] }[] }): number {
  return n.children.reduce((a, c) => a + 1 + countLinks(c as typeof n), 0);
}

/** The results stills of a capture (see the call site); each waits for its step fetch to settle. */
async function captureResults(path: string, settle: (ms?: number) => Promise<unknown>) {
  if (!(await effects.openResults(path))) throw new Error(`capture: results ${path} did not open: ${store.get().artifacts.results?.error}`);
  const s = store.get();
  const stages = s.results?.stages ?? [];
  const onAxis = s.phase.axis.some((k) => k.kind === "results");
  const still = async (name: string) => {
    for (let i = 0; i < 100 && store.get().resultStep?.status === "loading"; i++) await settle(20);
    await settle();
    viewport.renderNow();
    // Two painted frames: a hidden window's first capture after a burst of DOM changes can be the frame before them.
    await painted();
    await painted();
    await bridge.captureStill(name);
  };
  if (!onAxis) {
    // Not paired: one still of the banner that says why.
    await still("results-unpaired");
    return;
  }
  const safe = (n: string) => n.replace(/[^A-Za-z0-9_.-]+/g, "_") || "unnamed";
  for (const [i, st] of stages.entries()) {
    store.dispatch({ type: "setPhase", at: { kind: "results", stage: i, step: 0 } });
    if (st.time.length > 1) store.dispatch({ type: "setResultStep", step: st.time.length - 1 });
    await still(`results-${i}-${safe(st.name)}`);
  }
  // The first stage again at a 1280 px window: the scrubber row must still clear the header.
  if (typeof bridge.captureResize === "function" && stages.length) {
    await bridge.captureResize(1280, window.innerHeight);
    store.dispatch({ type: "setPhase", at: { kind: "results", stage: 0, step: 0 } });
    if (stages[0]!.time.length > 1) store.dispatch({ type: "setResultStep", step: stages[0]!.time.length - 1 });
    await still(`results-0-${safe(stages[0]!.name)}.1280`);
  }
}

/**
 * 3-D line diagram stills on a `.ladruno` stage with `lineStations`:
 * (1) `line-diagram-1-mz-sign`: Mz with ghosted slabs (`opacity 0.35`) and two-tone sign colouring,
 * (2) `line-diagram-2-mz-deformed-selected`: Mz on deformed shape (`scale 10`, reference ghost) with two contiguous beams selected (`select` + `selectAdd`) and surfaces `off` (`opacity 0`),
 * (3) `line-diagram-3-with-contour`: simultaneous Contour (`DISPLACEMENT` on quad4 floor slabs) + Line diagram (`Mz` on 1-D frame) + Deformed shape (`scale 10`).
 */
async function captureLineDiagrams(settle: (ms?: number) => Promise<unknown>, width: number, height: number) {
  const s0 = store.get();
  if (!s0.results || !s0.phase.axis.some((k) => k.kind === "results")) return;
  const stages = s0.results.stages;
  const stageIdx = stages.findIndex((st) => (st.lineStations?.length ?? 0) > 0);
  if (stageIdx < 0) return;
  if (typeof bridge.captureResize === "function") await bridge.captureResize(width, height);
  const st = stages[stageIdx]!;
  const last = Math.max(0, st.time.length - 1);
  store.dispatch({ type: "clearSelection" });
  store.dispatch({ type: "setDeform", on: false });
  store.dispatch({ type: "setDeformScale", scale: 1 });
  store.dispatch({ type: "setReferenceGhost", on: false });
  store.dispatch({ type: "setColourBy", by: "group" });
  store.dispatch({ type: "setOpacity", value: 0.35 });
  store.dispatch({ type: "setPhase", at: { kind: "results", stage: stageIdx, step: last } });
  const comp = st.lineStations!.includes("Mz") ? "Mz" : st.lineStations![0]!;
  store.dispatch({ type: "setLineDiagramComponent", component: comp });
  store.dispatch({ type: "setLineDiagramRange", range: "step" });
  store.dispatch({ type: "setLineDiagramCmap", cmap: "sign" });
  store.dispatch({ type: "setLineDiagram", on: true });
  await until(() => store.get().resultStep?.status === "ready" && store.get().lineDiagram.stepRange !== null, 15000);
  await settle(150);
  viewport.renderNow();
  await painted();
  await painted();
  await bridge.captureStill("line-diagram-1-mz-sign");

  // Second still: deformed shape + surfaces off + two beams selected (`select` + `selectAdd`)
  store.dispatch({ type: "setOpacity", value: 0 });
  if (deformOf(store.get()).ok) {
    store.dispatch({ type: "setDeform", on: true });
    store.dispatch({ type: "setDeformScale", scale: 10 });
    store.dispatch({ type: "setReferenceGhost", on: true });
  }
  const beams = (store.get().mesh?.elements ?? []).filter((d) => store.get().decls[d]?.element?.cell?.alias.startsWith("line"));
  if (beams.length >= 2) {
    store.dispatch({ type: "select", pick: { decl: beams[0]!, at: null, nodes: null } });
    store.dispatch({ type: "selectAdd", pick: { decl: beams[1]!, at: null, nodes: null } });
  }
  await settle(150);
  viewport.renderNow();
  await painted();
  await painted();
  await bridge.captureStill("line-diagram-2-mz-deformed-selected");

  // Third still: simultaneous Contour (DISPLACEMENT on quad4 floor slabs) + Line diagram (Mz on 1-D frame) + Deformed shape (scale 10)
  store.dispatch({ type: "clearSelection" });
  store.dispatch({ type: "setOpacity", value: 0.75 });
  store.dispatch({ type: "setColourBy", by: "contour" });
  store.dispatch({ type: "setContourRange", range: "history" });
  const nf = contourFieldsOf(store.get())?.fields[0];
  if (nf && nf.components[0]) {
    store.dispatch({ type: "setContourField", source: "nodal", field: nf.name, component: nf.components[0].key });
  }
  await until(
    () =>
      contourShown() &&
      store.get().contour.scan?.status === "done" &&
      deformShown() &&
      store.get().lineDiagram.stepRange !== null,
    30000,
  );
  await settle(150);
  viewport.renderNow();
  await painted();
  await painted();
  await bridge.captureStill("line-diagram-3-with-contour");

  // Restore state for subsequent capture steps
  store.dispatch({ type: "setLineDiagram", on: false });
  store.dispatch({ type: "setLineDiagramRange", range: "history" });
  store.dispatch({ type: "setDeform", on: false });
  store.dispatch({ type: "setDeformScale", scale: 1 });
  store.dispatch({ type: "setReferenceGhost", on: false });
  store.dispatch({ type: "setColourBy", by: "group" });
  store.dispatch({ type: "setOpacity", value: 1 });
  store.dispatch({ type: "clearSelection" });
}

/**
 * 3-D frame extrusion + rigid joint-offset results stills on models with `hasExtrudableFrames`:
 * (1) `extrude-results-1-deform-ghost`: extruded 3-D frame + rigid joint offsets + deformed shape (`scale 25`, reference ghost),
 * (2) `extrude-results-2-deform-contour`: extruded 3-D frame + rigid joint offsets + deformed shape + nodal contour (`|u|`),
 * (3) `extrude-results-3-line-diagram`: extruded 3-D frame (auto-dimmed to 0.25 opacity) + rigid joint offsets + 3-D moment diagram (`My`/`Mz`),
 * (4) `extrude-results-4-deform-contour-diagram`: simultaneous extruded 3-D frame + rigid joint offsets + deformed shape + nodal contour + 3-D moment diagram.
 */
async function captureExtrudeResults(settle: (ms?: number) => Promise<unknown>, width: number, height: number) {
  const s0 = store.get();
  if (!s0.mesh?.frameExtrusion?.hasExtrudableFrames || !s0.results || !s0.phase.axis.some((k) => k.kind === "results")) return;
  const stages = s0.results.stages;
  let stageIdx = stages.findIndex((st) => (st.lineStations?.length ?? 0) > 0);
  if (stageIdx < 0) stageIdx = 0;
  const st = stages[stageIdx];
  if (!st) return;
  if (typeof bridge.captureResize === "function") await bridge.captureResize(width, height);
  const last = Math.max(0, st.time.length - 1);

  store.dispatch({ type: "clearSelection" });
  store.dispatch({ type: "setLineDiagram", on: false });
  store.dispatch({ type: "setOpacity", value: 1 });
  store.dispatch({ type: "setColourBy", by: "group" });
  store.dispatch({ type: "setPhase", at: { kind: "results", stage: stageIdx, step: last } });
  if (!store.get().visibility.extrudeFrames) store.dispatch({ type: "toggleExtrudeFrames" });
  if (s0.mesh.frameExtrusion.hasJointOffsets && !store.get().visibility.showOffsets) {
    store.dispatch({ type: "toggleShowOffsets" });
  }

  // (1) Deformed shape + reference ghost on 3-D extruded frame + offsets
  if (deformOf(store.get()).ok) {
    store.dispatch({ type: "setDeform", on: true });
    store.dispatch({ type: "setDeformScale", scale: 25 });
    store.dispatch({ type: "setReferenceGhost", on: true });
  }
  await until(() => deformShown(), 15000);
  await settle(150);
  viewport.renderNow();
  await painted();
  await painted();
  await bridge.captureStill("extrude-results-1-deform-ghost");

  // (2) Deformed shape + nodal contour on 3-D extruded frame + offsets
  store.dispatch({ type: "setColourBy", by: "contour" });
  store.dispatch({ type: "setContourRange", range: "history" });
  const nf = contourFieldsOf(store.get())?.fields[0];
  if (nf && nf.components[0]) {
    store.dispatch({ type: "setContourField", source: "nodal", field: nf.name, component: nf.components[0].key });
  }
  await until(() => contourShown() && store.get().contour.scan?.status === "done" && deformShown(), 30000);
  await settle(150);
  viewport.renderNow();
  await painted();
  await painted();
  await bridge.captureStill("extrude-results-2-deform-contour");

  // (3) Undeformed 3-D extruded frame (auto-dimmed to 0.25) + offsets + 3-D moment diagram
  if ((st.lineStations?.length ?? 0) > 0) {
    store.dispatch({ type: "setDeform", on: false });
    store.dispatch({ type: "setReferenceGhost", on: false });
    store.dispatch({ type: "setColourBy", by: "group" });
    const comp = st.lineStations!.includes("Mz") ? "Mz" : st.lineStations!.includes("My") ? "My" : st.lineStations![0]!;
    store.dispatch({ type: "setLineDiagramComponent", component: comp });
    store.dispatch({ type: "setLineDiagramRange", range: "step" });
    store.dispatch({ type: "setLineDiagramCmap", cmap: "sign" });
    store.dispatch({ type: "setLineDiagram", on: true });
    await until(() => store.get().resultStep?.status === "ready" && store.get().lineDiagram.stepRange !== null, 15000);
    await settle(150);
    viewport.renderNow();
    await painted();
    await painted();
    await bridge.captureStill("extrude-results-3-line-diagram");

    // (4) Simultaneous 3-D extruded frame + offsets + deformed shape + nodal contour + 3-D moment diagram
    if (deformOf(store.get()).ok) {
      store.dispatch({ type: "setDeform", on: true });
      store.dispatch({ type: "setDeformScale", scale: 25 });
      store.dispatch({ type: "setReferenceGhost", on: true });
    }
    store.dispatch({ type: "setColourBy", by: "contour" });
    await until(() => contourShown() && deformShown() && store.get().lineDiagram.stepRange !== null, 30000);
    await settle(150);
    viewport.renderNow();
    await painted();
    await painted();
    await bridge.captureStill("extrude-results-4-deform-contour-diagram");

    // (5) Collapse #line-diagram and verify #loads immediately shifts up with no overlap
    const ldPanel = document.getElementById("line-diagram")!;
    const loadsPanel = document.getElementById("loads")!;
    const legendPanel = document.getElementById("legend")!;
    // Every left-rail card starts collapsed: open these three, so the overlap and collapse checks below see them open.
    for (const panel of [ldPanel, loadsPanel, legendPanel]) {
      if (!panel.hidden && panel.dataset["collapsed"] === "true") panel.querySelector("h3")?.click();
    }
    await painted();
    await painted();
    if (!ldPanel.hidden && !loadsPanel.hidden && !legendPanel.hidden) {
      const ldBefore = ldPanel.getBoundingClientRect();
      const loadsBefore = loadsPanel.getBoundingClientRect();
      const legendBefore = legendPanel.getBoundingClientRect();
      if (ldBefore.bottom > loadsBefore.top + 1) {
        throw new Error(`capture: #line-diagram (${ldBefore.bottom}) overlaps #loads (${loadsBefore.top})`);
      }
      if (loadsBefore.bottom > legendBefore.top + 1) {
        throw new Error(`capture: #loads (${loadsBefore.bottom}) overlaps #legend (${legendBefore.top})`);
      }
      ldPanel.querySelector("h3")?.click();
      await painted();
      await painted();
      const loadsAfter = loadsPanel.getBoundingClientRect();
      if (!(loadsAfter.top < loadsBefore.top - 40)) {
        throw new Error(`capture: collapsing #line-diagram did not shift #loads up (${loadsBefore.top} -> ${loadsAfter.top})`);
      }
      await bridge.captureStill("extrude-results-5-collapsed-line-diagram");
      ldPanel.querySelector("h3")?.click();
      await painted();

      // (6) Switch global units in #header to kN-mm-tonne-s and verify #header, #line-diagram, #loads, and #legend all convert
      const prevTargetUnits = store.get().contour.targetUnits;
      store.dispatch({ type: "setContourUnits", target: "kN_m_s", targetUnits: "kN_m_s" });
      await painted();
      await painted();
      await bridge.captureStill("extrude-results-6-global-units-kN-m");
      store.dispatch({ type: "setContourUnits", target: prevTargetUnits, targetUnits: prevTargetUnits });
      await painted();
    }
  }

  // Restore state for subsequent capture steps
  if (store.get().visibility.extrudeFrames) store.dispatch({ type: "toggleExtrudeFrames" });
  store.dispatch({ type: "setLineDiagram", on: false });
  store.dispatch({ type: "setLineDiagramRange", range: "history" });
  store.dispatch({ type: "setDeform", on: false });
  store.dispatch({ type: "setDeformScale", scale: 1 });
  store.dispatch({ type: "setReferenceGhost", on: false });
  store.dispatch({ type: "setColourBy", by: "group" });
  store.dispatch({ type: "setOpacity", value: 1 });
  store.dispatch({ type: "clearSelection" });
}

/** Poll the store (every 20 ms, at most `ms`) until `ok` holds; false on timeout. */
async function until(ok: () => boolean, ms = 15000): Promise<boolean> {
  for (let t = 0; t < ms; t += 20) {
    if (ok()) return true;
    await new Promise((r) => setTimeout(r, 20));
  }
  return ok();
}

/** The contour colours on screen are the step the state shows, at its range (renderer/contour.ts keys); V4c2a: an element field's too. */
function contourShown(): boolean {
  const v = contourView(store.get());
  if (v.state === "ready") return v.stepMeasured && viewport.contourApplied === contourKey(v);
  if (v.state === "element") return v.stepMeasured && viewport.contourApplied === elementContourKey(v);
  return false;
}

/** V4c2a: the mesh shows a refused or not-yet element contour as no data (its key names the reason). */
function refusalShown(): boolean {
  const v = contourView(store.get());
  return (v.state === "refused" || v.state === "not-yet") && viewport.contourApplied === contourShownKey(v);
}

/**
 * The V4c2a element-contour stills (D7), by a fixed rule so no flag is
 * needed: on the results stage with the most steps (the last of a tie) that
 * has an element field, its first element field's first component over the
 * whole-history range at the last step: `elements-1-<topology>` (or
 * `elements-1-refused-<code>` when the file is refused, with the banner);
 * then, drawable and deformable, the deformed shape at scale 10 with the
 * reference ghost (`elements-2-deformed-scale10`); V4c2b: then each
 * node-located draw mode (`elements-3-cell_averaged`, `-node`,
 * `-node_discrete`); then the first still again at 1280 px.
 * `<out>.elements-<n>-<what>.png`.
 */
async function captureElements(settle: (ms?: number) => Promise<unknown>, width: number, height: number) {
  const s0 = store.get();
  if (!s0.results || !s0.phase.axis.some((k) => k.kind === "results")) return;
  const stages = s0.results.stages;
  let best = -1;
  stages.forEach((st, i) => {
    if (gaussFields(st.gauss).length && (best < 0 || st.time.length >= stages[best]!.time.length)) best = i;
  });
  if (best < 0) return;
  if (typeof bridge.captureResize === "function") await bridge.captureResize(width, height);
  store.dispatch({ type: "setDeform", on: false });
  store.dispatch({ type: "clearSelection" });
  store.dispatch({ type: "setPhase", at: { kind: "results", stage: best, step: 0 } });
  const last = stages[best]!.time.length - 1;
  if (last > 0) store.dispatch({ type: "setResultStep", step: last });
  const f = gaussFields(stages[best]!.gauss)[0]!;
  store.dispatch({ type: "setColourBy", by: "contour" });
  store.dispatch({ type: "setContourRange", range: "history" });
  store.dispatch({ type: "setContourField", field: f.name, component: f.components[0]!.key, source: "element" });
  const ready = () => {
    const v = contourView(store.get());
    if (v.state === "element") return contourShown() && store.get().contour.scan?.status === "done" && deformShown();
    return refusalShown();
  };
  const still = async (name: string) => {
    if (!(await until(ready, 30000))) throw new Error(`capture ${name}: the element contour did not show (${contourView(store.get()).state})`);
    await settle(100);
    viewport.renderNow();
    await painted();
    await painted();
    await bridge.captureStill(name);
  };
  await until(() => contourView(store.get()).state !== "unavailable", 15000);
  const v = contourView(store.get());
  const first = v.state === "element" ? `elements-1-${v.topology}` : v.state === "refused" ? `elements-1-refused-${v.code}` : `elements-1-${v.state}`;
  await still(first);
  if (v.state === "element" && deformOf(store.get()).ok) {
    store.dispatch({ type: "setDeform", on: true });
    store.dispatch({ type: "setReferenceGhost", on: true });
    store.dispatch({ type: "setDeformScale", scale: 10 });
    await still("elements-2-deformed-scale10");
    store.dispatch({ type: "setDeform", on: false });
  }
  // V4c2b: the node-located draw modes, one still each (`elements-3-<mode>`), then back to `cell`.
  for (const mode of ["cell_averaged", "node", "node_discrete"] as const) {
    store.dispatch({ type: "setContourGauss", mode });
    await still(`elements-3-${mode}`);
  }
  store.dispatch({ type: "setContourGauss", mode: "cell" });
  if (typeof bridge.captureResize === "function") {
    await bridge.captureResize(1280, height);
    await settle();
    // Nothing is selected: frame the whole model in the narrower window.
    store.dispatch({ type: "frameSelection" });
    await settle();
    await still(`${first}.1280`);
  }
}

/**
 * The V4c1 contour stills (D7), by a fixed rule so no flag is needed: on the
 * results stage with the most steps (the last of a tie), coloured by its
 * first nodal field's first component (the magnitude when offered), the
 * legend while the history scan runs (main paces a capture's scan, one step
 * per 2 s), then after it at the first and the last step, then the first
 * axis component at the last step, and that last still again at 1280 px.
 * `<out>.contour-<n>-<what>.png`.
 */
async function captureContours(settle: (ms?: number) => Promise<unknown>, width: number, height: number) {
  const s0 = store.get();
  if (!s0.results || !s0.phase.axis.some((k) => k.kind === "results")) return;
  const stages = s0.results.stages;
  let best = -1;
  stages.forEach((st, i) => {
    if (st.nodal.length && (best < 0 || st.time.length >= stages[best]!.time.length)) best = i;
  });
  if (best < 0) return;
  if (typeof bridge.captureResize === "function") await bridge.captureResize(width, height);
  const still = async (name: string) => {
    if (!(await until(contourShown))) throw new Error(`capture ${name}: the contour did not show the step on screen`);
    await settle(100);
    viewport.renderNow();
    await painted();
    await painted();
    await bridge.captureStill(name);
  };
  store.dispatch({ type: "setPhase", at: { kind: "results", stage: best, step: 0 } });
  store.dispatch({ type: "setColourBy", by: "contour" });
  const scan = () => store.get().contour.scan;
  if (!(await until(() => scan()?.status === "running" && scan()!.done > 0))) throw new Error(`capture: the history scan did not report progress (${JSON.stringify(scan())})`);
  await still("contour-1-scan-running");
  // The still must show a scan in progress: one that finished first would show "whole history" under this name.
  if (scan()?.status !== "running") throw new Error(`capture: the history scan finished before its still was taken (${JSON.stringify(scan())})`);
  if (!(await until(() => scan()?.status === "done", 30000))) throw new Error(`capture: the history scan did not finish (${JSON.stringify(scan())})`);
  await still("contour-2-history-first-step");
  const last = stages[best]!.time.length - 1;
  store.dispatch({ type: "setResultStep", step: last });
  await still("contour-3-history-last-step");
  const v = contourView(store.get());
  const axis = v.state === "ready" ? contourFieldsOf(store.get())?.fields.find((f) => f.name === v.field)?.components.find((c) => !c.magnitude) : undefined;
  if (axis && v.state === "ready") {
    store.dispatch({ type: "setContourField", field: v.field, component: axis.key });
    await until(() => scan()?.status === "done" && scan()!.key.endsWith(`|${axis.key}`), 30000);
    await still(`contour-4-component-${axis.label}`);
  }
  if (typeof bridge.captureResize === "function") {
    await bridge.captureResize(1280, height);
    await settle();
    await still("contour-5-last-step.1280");
  }
}

/** V4d: the drawn positions are the deformed shape of the step the state shows, at its scale (renderer/deform.ts keys); or the shape cannot be drawn. */
function deformShown(): boolean {
  const v = deformView(store.get());
  if (v.state === "off" || v.state === "unavailable" || v.state === "refused") return true;
  if (v.state === "loading") return false;
  return viewport.deformApplied === `${effects.currentDisplacement()?.key}|scale ${v.scale}` && effects.currentDisplacement()?.key === v.keys.disp;
}

/** Resolves once the contour (and, V4d, the deformed shape) shows the step the state names (checked after every store change; the viewport listens first). */
function contourShowsNow(): Promise<void> {
  const shown = () => contourShown() && deformShown();
  return new Promise((resolve) => {
    if (shown()) return resolve();
    const off = store.subscribe(() => {
      if (!shown()) return;
      off();
      resolve();
    });
  });
}

/**
 * The V4d deformed-shape stills (D7), by a fixed rule so no flag is needed:
 * on the results stage with the most steps (the last of a tie) that has a
 * usable displacement, at its last step, coloured by group: deformed at true
 * scale with the reference ghost, then at scale 10, then coloured by
 * contour; then two staged views (the first two results stages at their
 * last step, or the first and the last step of a single stage), and the
 * scale-10 still again at 1280 px. A `partial` stage shows its banner and
 * markers in every still. `<out>.deform-<n>-<what>.png`.
 */
async function captureDeform(settle: (ms?: number) => Promise<unknown>, width: number, height: number) {
  const s0 = store.get();
  if (!s0.results || !s0.phase.axis.some((k) => k.kind === "results")) return;
  const stages = s0.results.stages;
  const usable = (i: number) => {
    store.dispatch({ type: "setPhase", at: { kind: "results", stage: i, step: 0 } });
    return deformOf(store.get()).ok;
  };
  let best = -1;
  stages.forEach((st, i) => {
    if (usable(i) && (best < 0 || st.time.length >= stages[best]!.time.length)) best = i;
  });
  if (best < 0) return;
  if (typeof bridge.captureResize === "function") await bridge.captureResize(width, height);
  store.dispatch({ type: "setColourBy", by: "group" });
  store.dispatch({ type: "clearSelection" });
  const show = (stage: number, step: number) => {
    store.dispatch({ type: "setPhase", at: { kind: "results", stage, step: 0 } });
    if (step > 0) store.dispatch({ type: "setResultStep", step });
  };
  const still = async (name: string) => {
    if (!(await until(() => deformShown() && (store.get().visibility.colourBy !== "contour" || contourShown())))) throw new Error(`capture ${name}: the deformed shape did not show the step on screen (${deformView(store.get()).state})`);
    await settle(100);
    viewport.renderNow();
    await painted();
    await painted();
    await bridge.captureStill(name);
  };
  const last = (i: number) => stages[i]!.time.length - 1;
  show(best, last(best));
  store.dispatch({ type: "setDeform", on: true });
  store.dispatch({ type: "setReferenceGhost", on: true });
  store.dispatch({ type: "setDeformScale", scale: 1 });
  await still("deform-1-scale1-ghost");
  store.dispatch({ type: "setDeformScale", scale: 10 });
  await still("deform-2-scale10-ghost");
  store.dispatch({ type: "setColourBy", by: "contour" });
  await still("deform-3-scale10-contour");
  store.dispatch({ type: "setColourBy", by: "group" });
  const safe = (n: string) => n.replace(/[^A-Za-z0-9_.-]+/g, "_") || "unnamed";
  const views: [number, number][] = stages.length >= 2 ? [[0, last(0)], [1, last(1)]] : [[0, 0], [0, last(0)]];
  for (const [i, k] of views) {
    show(i, k);
    await still(`deform-4-stage${i}-${safe(stages[i]!.name)}-step${k + 1}`);
  }
  if (typeof bridge.captureResize === "function") {
    await bridge.captureResize(1280, height);
    show(best, last(best));
    await settle();
    await still("deform-5-scale10-ghost.1280");
  }
  store.dispatch({ type: "setDeform", on: false });
}

/** V4e3: the deformation scale of the capped scrub (see `measureContour`). */
const CAPPED_SCALE = 1e-6;

/**
 * The V4c1 contour scrub (MEASUREMENTS.md, the contour rows): on the results
 * stage with the most steps, coloured by its first field's first component
 * (the magnitude when offered) over the whole-history range, every step in
 * turn, end to end: `setResultStep`, the step read in main, the IPC, the
 * step range, the vertex colours, the upload and a draw with `gl.finish()`.
 * The history scan's time comes first; the machine's CPU load is sampled
 * before (1 s idle) and during the scrub, and the working sets every 100 ms.
 */
async function measureContour(path: string, deform = false, element = false, gauss: GaussMode = "cell", planes = false, capped = false) {
  store.dispatch({ type: "clearSelection" });
  // V4d: the same scrub with the deformed shape on (true scale): each step also waits for the displaced positions.
  store.dispatch({ type: "setDeform", on: deform });
  if (!(await effects.openResults(path))) throw new Error(`measure: results ${path} did not open: ${store.get().artifacts.results?.error}`);
  const s = store.get();
  if (!s.phase.axis.some((k) => k.kind === "results")) throw new Error(`measure: ${path} does not pair with the model, so there is nothing to scrub`);
  const stages = s.results!.stages;
  // V4c2a: the element scrub colours by the first drawable element field of a verified file (none: no row).
  const elementField = (i: number) => gaussFields(stages[i]!.gauss).find((f) => f.refused === null);
  if (element && !elementVerdictOf(s.results!).verified) return null;
  let best = -1;
  stages.forEach((st, i) => {
    const has = element ? elementField(i) !== undefined : st.nodal.length > 0;
    if (has && (best < 0 || st.time.length > stages[best]!.time.length)) best = i;
  });
  if (best < 0 && element) return null;
  if (best < 0) throw new Error(`measure: ${path} has no stage with nodal results`);
  const cpu = bridge.measureCpu!;
  const memory = bridge.measureMemory!;
  await cpu();
  await new Promise((r) => setTimeout(r, 1000));
  const cpuBefore = await cpu();
  const peak = { main: 0, renderer: 0 };
  const sample = async () => {
    const m = await memory();
    if (m.main + m.renderer > peak.main + peak.renderer) Object.assign(peak, m);
  };
  const sampler = setInterval(() => void sample(), 100);
  // V4e3: the caps toggle and the deformation scale as they were, put back after the scrub.
  const cutFaceBefore = store.get().clip.cutFace;
  const scaleBefore = store.get().deform.scale;
  try {
    const t0 = performance.now();
    store.dispatch({ type: "setPhase", at: { kind: "results", stage: best, step: 0 } });
    store.dispatch({ type: "setColourBy", by: "contour" });
    store.dispatch({ type: "setContourRange", range: "history" });
    if (element) {
      const f = elementField(best)!;
      store.dispatch({ type: "setContourField", field: f.name, component: f.components[0]!.key, source: "element" });
    }
    // V4c2b: the element scrub in a node-located draw mode (`cell` leaves the V4c2a rows as they were).
    store.dispatch({ type: "setContourGauss", mode: element ? gauss : "cell" });
    // V4c2b: a node-located mode waits for its own scan (its key names the mode), never an earlier mode's.
    const scanDone = () =>
      store.get().contour.scan?.status === "done" && (!element || store.get().contour.scan!.key.includes("|elem:")) && (gauss === "cell" || store.get().contour.scan!.key === contourKeysOf(store.get())?.scan);
    if (!(await until(scanDone, 600000))) throw new Error(`measure: the history scan did not finish: ${JSON.stringify(store.get().contour.scan)}`);
    const scanMs = performance.now() - t0;
    await contourShowsNow();
    const v = contourView(store.get());
    const d = deformView(store.get());
    if (deform && d.state !== "ready" && d.state !== "partial") throw new Error(`measure: the deformed shape is ${d.state}`);
    if (element && v.state !== "element") throw new Error(`measure: the element contour is ${v.state}`);
    const what = v.state === "ready" ? `${v.field} ${v.component.label}` : v.state === "element" ? `element ${v.field} ${v.component.label}, ${v.topology}` : "?";
    // V4e3: the cut-face caps on (the `caps` row) or off (the `clip` row, comparable with the V4e1 and V4e2 rows).
    // The measured files' displacement is synthetic (the node id plus a ramp: thousands of model lengths), so at
    // true scale the deformed solid leaves every plane and no cap is cut; the capped row deforms it at ×1e-6.
    store.dispatch({ type: "setCutFace", on: capped });
    if (capped) store.dispatch({ type: "setDeformScale", scale: CAPPED_SCALE });
    // V4e1: six section planes cutting through the scrub (each 20 % inside the reference box along ±X, ±Y, ±Z).
    if (planes) addSixPlanes();
    if (planes && viewport.clipCount !== 6) throw new Error(`measure: ${viewport.clipCount} section planes cut, not 6`);
    // V4e2: and their six gizmos drawn (shown by default), so the scrub renders them every step.
    if (planes && viewport.gizmoCount !== 6) throw new Error(`measure: ${viewport.gizmoCount} plane gizmos drawn, not 6`);
    // V4e3: and, capped, a cap on each of the six planes, sliced again at every step (the deformed solid, the step's field).
    if (capped && !(await until(() => viewport.capsSettled() && viewport.capCount === 6))) throw new Error(`measure: ${viewport.capCount} caps drawn, not 6`);
    const slices0 = viewport.caps.slices;
    const field = what + (deform ? `, deformed ×${capped ? CAPPED_SCALE : 1}` : "") + (planes ? ", 6 section planes + gizmos" : "") + (capped ? " + 6 caps" : "");
    const T = stages[best]!.time.length;
    const gl = viewport.renderer.getContext();
    const stepMs: number[] = [];
    await cpu();
    const start = performance.now();
    for (let k = 1; k < T; k++) {
      const t = performance.now();
      store.dispatch({ type: "setResultStep", step: k });
      await contourShowsNow();
      if (capped && !viewport.capsSettled()) throw new Error(`measure: the caps do not show step ${k} with its contour`);
      viewport.renderNow();
      gl.finish();
      stepMs.push(performance.now() - t);
    }
    const total = performance.now() - start;
    const cpuDuring = await cpu();
    await sample();
    const sorted = [...stepMs].sort((a, b) => a - b);
    // V4e3: one slice per plane per step, never more (F-SETTLED at scale).
    const slices = viewport.caps.slices - slices0;
    if (capped && (slices !== 6 * (T - 1) || viewport.capCount !== 6)) throw new Error(`measure: ${slices} slices over ${T - 1} steps (want ${6 * (T - 1)}), ${viewport.capCount} caps`);
    return {
      file: path,
      field: `${field} (stage ${stages[best]!.name || stages[best]!.id})`,
      steps: T,
      ...(capped ? { capSlices: slices, capTriangles: viewport.caps.drawn.reduce((a, c) => a + c.slice.cell.length, 0) } : {}),
      scanMs,
      scrubStepsPerSec: ((T - 1) * 1000) / total,
      stepMs: { median: sorted[Math.floor((sorted.length - 1) / 2)] ?? NaN, max: sorted[sorted.length - 1] ?? NaN },
      peakMB: peak,
      cpu: { before: cpuBefore, during: cpuDuring },
    };
  } finally {
    clearInterval(sampler);
    if (planes) removePlanes();
    store.dispatch({ type: "setCutFace", on: cutFaceBefore });
    store.dispatch({ type: "setDeformScale", scale: scaleBefore });
  }
}

/** V4e1: six active section planes, one per face of the reference box, each 20 % of its extent inside it. */
function addSixPlanes() {
  const box = viewport.sliderBox();
  if (!box) throw new Error("section planes: no mesh");
  const normals: [number, number, number][] = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]];
  for (const n of normals) {
    store.dispatch({ type: "addClipPlane", normal: n });
    const p = store.get().clip.planes.at(-1)!;
    const r = offsetRange(p.normal, box);
    store.dispatch({ type: "setClipOffset", id: p.id, offset: r.min + 0.2 * (r.max - r.min) });
  }
}

function removePlanes() {
  for (const p of store.get().clip.planes) store.dispatch({ type: "removeClipPlane", id: p.id });
}

/** The id of the plane added last. */
const lastPlane = (): string => store.get().clip.planes.at(-1)!.id;

/** Put plane `id` at fraction `t` of the reference box along its normal (0: the low end). */
function placeAt(id: string, t: number) {
  const box = viewport.sliderBox()!;
  const p = store.get().clip.planes.find((q) => q.id === id)!;
  const r = offsetRange(p.normal, box);
  store.dispatch({ type: "setClipOffset", id, offset: r.min + t * (r.max - r.min) });
}

/** Add a plane of normal `n` at fraction `t` of the reference box along it; its id. */
function addAt(n: readonly number[], t: number): string {
  store.dispatch({ type: "addClipPlane", normal: n });
  const id = lastPlane();
  placeAt(id, t);
  return id;
}

/**
 * The V4e1 section-plane stills (D7), by a fixed rule so no flag is needed:
 * on the results stage with the most steps (the last of a tie) whose
 * deformed shape can be drawn, at its last step, coloured by group,
 * undeformed: the model before any plane; one Z plane, flipped to keep the
 * lower 40 %, so the solid shows hollow from above (no caps until V4e3); an
 * X plane at 30 % added (a corner cut); the Z plane flipped back; its offset moved to
 * 75 %; the panel with the X plane's Custom normal fields open on (1, 1, 0);
 * six planes, each 10 % inside a face of the box; a 7th refused with its
 * notice. Then a click that misses cut-away geometry: a plane along the view
 * cuts away the nearer half (on a planar model shown in plan, which lies in
 * that plane, the first axis plane through the centre that cuts something
 * away), and a scripted click on an element of that half is a miss (the ring
 * marks where it clicked); the same click with "Apply cuts" off selects it.
 * Then the deformed (scale 10, ghost on), contoured
 * model cut by a Y plane (the half nearer the camera gone) and an X plane at
 * 45 %, at the first and the last step of the scrub (the shear carries more
 * of it through the X plane), and that last still at 1280 px. Plane offsets
 * avoid the mesh's own face planes, where a cut coincides with a face.
 * `<out>.clip-<n>-<what>.png`.
 */
async function captureClip(settle: (ms?: number) => Promise<unknown>, width: number, height: number) {
  const s0 = store.get();
  if (!s0.results || !s0.mesh || !s0.phase.axis.some((k) => k.kind === "results")) return;
  const stages = s0.results.stages;
  const usable = (i: number) => {
    store.dispatch({ type: "setPhase", at: { kind: "results", stage: i, step: 0 } });
    return deformOf(store.get()).ok;
  };
  let best = -1;
  stages.forEach((st, i) => {
    if (usable(i) && (best < 0 || st.time.length >= stages[best]!.time.length)) best = i;
  });
  if (best < 0) return;
  if (typeof bridge.captureResize === "function") await bridge.captureResize(width, height);
  const last = stages[best]!.time.length - 1;
  removePlanes();
  store.dispatch({ type: "setApplyCuts", on: true });
  // V4e3: these are the V4e1 stills, caps off (the hollow cut they show is what the caps fill: captureCaps).
  store.dispatch({ type: "setCutFace", on: false });
  store.dispatch({ type: "setDeform", on: false });
  store.dispatch({ type: "setColourBy", by: "group" });
  store.dispatch({ type: "clearSelection" });
  store.dispatch({ type: "setPhase", at: { kind: "results", stage: best, step: 0 } });
  if (last > 0) store.dispatch({ type: "setResultStep", step: last });
  store.dispatch({ type: "frameSelection" });
  const still = async (name: string) => {
    if (!(await until(() => deformShown() && (store.get().visibility.colourBy !== "contour" || contourShown())))) throw new Error(`capture ${name}: the step did not show`);
    await settle(150);
    viewport.renderNow();
    await painted();
    await painted();
    await bridge.captureStill(name);
  };
  await still("clip-1-before");
  const z = addAt([0, 0, 1], 0.4);
  store.dispatch({ type: "setClipFlipped", id: z, flipped: true });
  await still("clip-2-z-plane-hollow");
  const x = addAt([1, 0, 0], 0.3);
  await still("clip-3-corner-two-planes");
  store.dispatch({ type: "setClipFlipped", id: z, flipped: false });
  await still("clip-4-z-flipped-back");
  placeAt(z, 0.75);
  await still("clip-5-z-offset-75");
  store.dispatch({ type: "setClipNormal", id: x, normal: [1, 1, 0] });
  await still("clip-6-panel-custom-normal");
  removePlanes();
  for (const n of [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]] as const) addAt(n, 0.1);
  await still("clip-7-six-planes");
  store.dispatch({ type: "addClipPlane", normal: [1, 0, 1] });
  if (store.get().clip.notice === null) throw new Error("capture: a 7th plane was not refused");
  await still("clip-8-seventh-refused");

  // The pick that misses cut-away geometry. The view normal through the centre cuts away the near half of a
  // body; a planar model shown in plan lies in that plane, where nothing is cut away (#2707), so the axis
  // normals through the centre are tried next, until one cuts away something a click can reach.
  let target: ReturnType<typeof clipMissTarget> = null;
  const normals: [number, number, number][] = [[...viewport.viewNormal()] as [number, number, number], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]];
  for (const normal of normals) {
    removePlanes();
    store.dispatch({ type: "addClipPlane", normal });
    target = clipMissTarget();
    if (target) break;
  }
  if (!target) throw new Error("capture: no element of the cut-away half answers a click with the cuts off");
  const marker = document.createElement("div");
  marker.style.cssText =
    `position: fixed; left: ${target.x - 9}px; top: ${target.y - 9}px; width: 18px; height: 18px; border-radius: 50%;` +
    "border: 3px solid var(--accent); pointer-events: none; z-index: 10;";
  const label = document.createElement("div");
  label.style.cssText = `position: fixed; left: ${target.x + 16}px; top: ${target.y - 10}px; padding: 2px 6px; border-radius: 5px; font: 600 12px var(--font); color: var(--text); background: var(--panel); border: 1px solid var(--line); pointer-events: none; z-index: 10;`;
  document.body.append(marker, label);
  const click = () => {
    const pick = viewport.pickAt(target.x, target.y);
    store.dispatch(pick ? { type: "select", pick } : { type: "clearSelection" });
    return pick;
  };
  const miss = click();
  if (miss) throw new Error(`capture: the click on cut-away geometry picked ${miss.decl}`);
  label.textContent = "click: miss (the face hit first is cut away; no pick-through)";
  await still("clip-9-pick-miss-cut-away");
  store.dispatch({ type: "setApplyCuts", on: false });
  const hit = click();
  if (hit?.decl !== target.decl) throw new Error(`capture: with the cuts off the click picked ${hit?.decl ?? "nothing"}, not ${target.decl}`);
  label.textContent = "the same click, Apply cuts off: picked";
  await settle(900);
  await still("clip-10-pick-cuts-off");
  marker.remove();
  label.remove();
  store.dispatch({ type: "clearSelection" });
  store.dispatch({ type: "setApplyCuts", on: true });
  removePlanes();

  // The deformed, contoured (and stage-activated) model cut while scrubbing.
  store.dispatch({ type: "setPhase", at: { kind: "results", stage: best, step: 0 } });
  store.dispatch({ type: "setDeform", on: true });
  store.dispatch({ type: "setReferenceGhost", on: true });
  store.dispatch({ type: "setDeformScale", scale: 10 });
  store.dispatch({ type: "setColourBy", by: "contour" });
  addAt([0, 1, 0], 0.5);
  addAt([1, 0, 0], 0.45);
  await still("clip-11-deformed-contour-step1");
  if (last > 0) store.dispatch({ type: "setResultStep", step: last });
  await still(`clip-12-deformed-contour-step${last + 1}`);
  if (typeof bridge.captureResize === "function") {
    await bridge.captureResize(1280, height);
    await settle();
    store.dispatch({ type: "frameSelection" });
    await still(`clip-13-deformed-contour-step${last + 1}.1280`);
  }
  removePlanes();
  store.dispatch({ type: "setDeform", on: false });
  store.dispatch({ type: "setColourBy", by: "group" });
  store.dispatch({ type: "setCutFace", on: true });
}

/**
 * The V4e3 cut-face stills (D7), by a fixed rule. On a model with volumetric
 * cells, on the results stage with the most steps whose deformed shape can be
 * drawn: one Z plane keeping the lower 40 % (the cut V4e1's clip-2 showed
 * hollow), its cap a flat fill (coloured by group); the same cap carrying the
 * contour; an X plane at 30 % added, the two caps trimming each other; the
 * deformed (scale 10) contoured model at the last step, cut by a Y plane at 50
 * % and an X plane at 45 % (V4e1's clip-12, now filled); and "Contour on cut
 * face" off, the same cut hollow again. On a model with no volumetric cell
 * (a shell, a frame): one Z plane at 50 % and the planes panel saying the
 * mesh gets no cap. Each still waits until the caps show what the state asks
 * for and checks their count. `<out>.caps-<n>-<what>.png`.
 */
async function captureCaps(settle: (ms?: number) => Promise<unknown>, width: number, height: number) {
  const s0 = store.get();
  if (!s0.mesh) return;
  if (typeof bridge.captureResize === "function") await bridge.captureResize(width, height);
  removePlanes();
  store.dispatch({ type: "setApplyCuts", on: true });
  store.dispatch({ type: "setCutFace", on: true });
  store.dispatch({ type: "setShowGizmos", on: true });
  store.dispatch({ type: "setDeform", on: false });
  store.dispatch({ type: "setColourBy", by: "group" });
  store.dispatch({ type: "clearSelection" });
  const still = async (name: string, caps: number) => {
    const ok = () => deformShown() && (store.get().visibility.colourBy !== "contour" || contourShown()) && viewport.capsSettled() && viewport.capCount === caps;
    if (!(await until(ok))) throw new Error(`capture ${name}: ${viewport.capCount} caps (want ${caps}), settled ${viewport.capsSettled()}`);
    await settle(150);
    viewport.renderNow();
    await painted();
    await painted();
    await bridge.captureStill(name);
  };
  if (s0.mesh.counts.solidCells === 0) {
    store.dispatch({ type: "frameSelection" });
    addAt([0, 0, 1], 0.5);
    const note = document.querySelector("#planes .planes-capnote");
    if (!note?.textContent) throw new Error("capture: the planes panel does not say this mesh gets no cap");
    await still("caps-0-no-volumetric-cells", 0);
    removePlanes();
    return;
  }
  const stages = s0.results?.stages ?? [];
  let best = -1;
  stages.forEach((st, i) => {
    store.dispatch({ type: "setPhase", at: { kind: "results", stage: i, step: 0 } });
    if (deformOf(store.get()).ok && (best < 0 || st.time.length >= stages[best]!.time.length)) best = i;
  });
  if (best < 0) return;
  const last = stages[best]!.time.length - 1;
  store.dispatch({ type: "setPhase", at: { kind: "results", stage: best, step: 0 } });
  if (last > 0) store.dispatch({ type: "setResultStep", step: last });
  store.dispatch({ type: "frameSelection" });
  const z = addAt([0, 0, 1], 0.4);
  store.dispatch({ type: "setClipFlipped", id: z, flipped: true });
  await still("caps-1-flat-fill", 1);
  store.dispatch({ type: "setColourBy", by: "contour" });
  await still("caps-2-contour", 1);
  addAt([1, 0, 0], 0.3);
  await still("caps-3-two-planes-trimmed", 2);
  removePlanes();
  store.dispatch({ type: "setDeform", on: true });
  store.dispatch({ type: "setReferenceGhost", on: true });
  store.dispatch({ type: "setDeformScale", scale: 10 });
  addAt([0, 1, 0], 0.5);
  addAt([1, 0, 0], 0.45);
  await still(`caps-4-deformed-contour-step${last + 1}`, 2);
  store.dispatch({ type: "setCutFace", on: false });
  await still("caps-5-cut-face-off", 0);
  store.dispatch({ type: "setCutFace", on: true });
  removePlanes();
  store.dispatch({ type: "setDeform", on: false });
  store.dispatch({ type: "setColourBy", by: "group" });
}

/**
 * The click target of the miss still: of the drawn elements whose own screen
 * point picks them with the cuts off and is cut away with them on, the one
 * nearest the middle of the model on screen.
 */
function clipMissTarget(): { decl: DeclPath; x: number; y: number } | null {
  const s = store.get();
  const c = new THREE.Vector3(...s.mesh!.center).project(viewport.camera);
  const rect = viewport.renderer.domElement.getBoundingClientRect();
  const cx = rect.left + ((c.x + 1) / 2) * rect.width, cy = rect.top + ((1 - c.y) / 2) * rect.height;
  let best: { decl: DeclPath; x: number; y: number; d: number } | null = null;
  for (const decl of s.mesh!.elements) {
    const pt = viewport.screenPointOf(decl);
    if (!pt) continue;
    const d = Math.hypot(pt.x - cx, pt.y - cy);
    if (best && d >= best.d) continue;
    store.dispatch({ type: "setApplyCuts", on: false });
    const off = viewport.pickAt(pt.x, pt.y);
    store.dispatch({ type: "setApplyCuts", on: true });
    if (off?.decl !== decl) continue;
    if (viewport.pickAt(pt.x, pt.y) === null) best = { decl, ...pt, d };
  }
  return best && { decl: best.decl, x: best.x, y: best.y };
}

/** The client point of a world point. */
function screenOf(p: readonly number[]): { x: number; y: number } {
  const v = new THREE.Vector3(p[0], p[1], p[2]).project(viewport.camera);
  const rect = viewport.renderer.domElement.getBoundingClientRect();
  return { x: rect.left + ((v.x + 1) / 2) * rect.width, y: rect.top + ((1 - v.y) / 2) * rect.height };
}

/** A real pointer event at a client point, sent where the browser would send it (the topmost element there). */
function pointerAt(type: string, at: { x: number; y: number }, target: Element = document.elementFromPoint(at.x, at.y) ?? viewport.renderer.domElement): Element {
  target.dispatchEvent(new PointerEvent(type, { clientX: at.x, clientY: at.y, button: 0, buttons: type === "pointerup" ? 0 : 1, pointerId: 1, bubbles: true, cancelable: true }));
  return target;
}

/** A ring and a label on the page at a client point (capture only). */
function marker(at: { x: number; y: number }, text: string): () => void {
  const ring = document.createElement("div");
  ring.style.cssText =
    `position: fixed; left: ${at.x - 9}px; top: ${at.y - 9}px; width: 18px; height: 18px; border-radius: 50%;` +
    "border: 3px solid var(--accent); pointer-events: none; z-index: 10;";
  const label = document.createElement("div");
  label.style.cssText = `position: fixed; left: ${at.x + 16}px; top: ${at.y - 10}px; padding: 2px 6px; border-radius: 5px; font: 600 12px var(--font); color: var(--text); background: var(--panel); border: 1px solid var(--line); pointer-events: none; z-index: 10;`;
  label.textContent = text;
  document.body.append(ring, label);
  return () => {
    ring.remove();
    label.remove();
  };
}

/**
 * The V4e2 plane-gizmo stills (D7), by a fixed rule, on the undeformed model
 * coloured by group, seen from the iso camera: one Z plane at 55 % with its gizmo; three planes (Z, X at
 * 40 %, Y at 60 %), three gizmos; the Z gizmo mid-drag (a real press on the
 * quad centre, moved a quarter of the arrow up the normal, still held); the Z
 * plane rotated by its tip (a real press, moved to tilt the normal about 30
 * degrees toward the camera, released); and the legend over a gizmo: the camera
 * zoomed in until the Z quad lies under the legend, a real press on the legend
 * there, which the gizmo does not take. Every gesture goes through the canvas's
 * own listeners. `<out>.gizmo-<n>-<what>.png`.
 */
async function captureGizmo(settle: (ms?: number) => Promise<unknown>, width: number, height: number) {
  if (!store.get().mesh) return;
  if (typeof bridge.captureResize === "function") await bridge.captureResize(width, height);
  removePlanes();
  store.dispatch({ type: "setApplyCuts", on: true });
  store.dispatch({ type: "setShowGizmos", on: true });
  store.dispatch({ type: "setDeform", on: false });
  store.dispatch({ type: "setColourBy", by: "group" });
  store.dispatch({ type: "clearSelection" });
  // The iso view: a planar model opens in plan, where the Z plane's quad and its arrow tip fall on one screen
  // point, so the press on the quad would land on the tip (#2707).
  store.dispatch({ type: "setCamera", to: "iso" });
  store.dispatch({ type: "frameSelection" });
  await settle(150);
  viewport.renderNow();
  await painted();
  const still = async (name: string) => {
    await settle(150);
    viewport.renderNow();
    await painted();
    await painted();
    await bridge.captureStill(name);
  };
  const plane = (id: string) => store.get().clip.planes.find((p) => p.id === id)!;
  const geom = (id: string) => viewport.gizmos.geometries().get(id)!;
  // Functions, not the getters themselves: each read is after a gesture or an event the type checker cannot see.
  const count = (): number => viewport.gizmoCount;
  const mode = (): string | null => viewport.gizmoInput.mode;
  const z = addAt([0, 0, 1], 0.55);
  if (count() !== 1) throw new Error(`capture: ${count()} gizmos for one plane`);
  await still("gizmo-1-one-plane");
  const x = addAt([1, 0, 0], 0.4);
  const y = addAt([0, 1, 0], 0.6);
  if (count() !== 3) throw new Error(`capture: ${count()} gizmos for three planes`);
  await still("gizmo-2-three-planes");
  store.dispatch({ type: "removeClipPlane", id: x });
  store.dispatch({ type: "removeClipPlane", id: y });

  // Mid-drag: a press on the quad centre, a move up the normal, the button still held.
  const g = geom(z);
  const off0 = plane(z).offset;
  const at0 = screenOf(g.centre);
  pointerAt("pointerdown", at0, viewport.renderer.domElement);
  if (mode() !== "translate") throw new Error(`capture: the press on the Z quad is ${mode()}, not a translate`);
  const up = [g.centre[0] + 0.25 * g.arrowLen * g.normal[0], g.centre[1] + 0.25 * g.arrowLen * g.normal[1], g.centre[2] + 0.25 * g.arrowLen * g.normal[2]];
  for (const t of [0.25, 0.5, 0.75, 1]) {
    const p = g.centre.map((c, k) => c + t * (up[k]! - c));
    pointerAt("pointermove", screenOf(p), viewport.renderer.domElement);
  }
  const moved = plane(z).offset - off0;
  if (!(Math.abs(moved - 0.25 * g.arrowLen) < 1e-3 * g.arrowLen)) throw new Error(`capture: the drag moved the plane ${moved}, not ${0.25 * g.arrowLen}`);
  const done = marker(screenOf(up), "mid-drag: the Z plane moves with the cursor along its normal");
  await still("gizmo-3-mid-drag");
  done();
  pointerAt("pointerup", screenOf(up), viewport.renderer.domElement);

  // Rotated: a press on the tip, moved to tilt the normal toward the camera, released.
  const r = geom(z);
  pointerAt("pointerdown", screenOf(tipPoint(r)), viewport.renderer.domElement);
  if (mode() !== "rotate") throw new Error(`capture: the press on the Z tip is ${mode()}, not a rotate`);
  const back = new THREE.Vector3(0, 0, 1).applyQuaternion(viewport.camera.quaternion); // toward the camera
  const n0 = new THREE.Vector3(...r.direction);
  const tilt = back.clone().sub(n0.clone().multiplyScalar(back.dot(n0))).normalize();
  const want = n0.clone().multiplyScalar(Math.cos(Math.PI / 6)).addScaledVector(tilt, Math.sin(Math.PI / 6));
  for (const t of [0.33, 0.66, 1]) {
    const d = n0.clone().lerp(want, t).normalize();
    pointerAt("pointermove", screenOf([r.centre[0] + r.arrowLen * d.x, r.centre[1] + r.arrowLen * d.y, r.centre[2] + r.arrowLen * d.z]), viewport.renderer.domElement);
  }
  pointerAt("pointerup", screenOf(tipPoint(geom(z))), viewport.renderer.domElement);
  const p = plane(z);
  const onPlane = p.normal[0] * r.centre[0] + p.normal[1] * r.centre[1] + p.normal[2] * r.centre[2] - p.offset;
  if (Math.abs(onPlane) > 1e-6 * (store.get().mesh!.radius || 1)) throw new Error(`capture: the rotated plane is ${onPlane} off its quad centre`);
  await still("gizmo-4-rotated");

  // The legend over a gizmo: zoom in until the (re-straightened) Z quad lies under the legend, then press there.
  store.dispatch({ type: "setClipPose", id: z, normal: [0, 0, 1], offset: r.centre[2] });
  const legend = document.getElementById("legend")!;
  const box = legend.getBoundingClientRect();
  const at = { x: box.left + box.width / 2, y: box.top + box.height / 2 };
  const ray = new THREE.Raycaster();
  const gizmoUnder = () => {
    const rect = viewport.renderer.domElement.getBoundingClientRect();
    ray.setFromCamera(new THREE.Vector2(((at.x - rect.left) / rect.width) * 2 - 1, -((at.y - rect.top) / rect.height) * 2 + 1), viewport.camera);
    return viewport.gizmos.hit({ origin: ray.ray.origin.toArray(), direction: ray.ray.direction.toArray() });
  };
  for (let k = 0; k < 12 && !gizmoUnder(); k++) {
    zoomToward(viewport.camera, new THREE.Vector3(...geom(z).centre), 0.75);
    viewport.nav.updated();
  }
  if (!gizmoUnder()) throw new Error("capture: no zoom put the Z gizmo under the legend");
  const target = pointerAt("pointerdown", at);
  pointerAt("pointerup", at, target);
  if (!legend.contains(target)) throw new Error(`capture: the press under the legend reached ${target.nodeName}#${target.id}, not the legend`);
  if (viewport.gizmoInput.dragging) throw new Error("capture: the gizmo took a press on the legend");
  const off = marker(at, "a press here is the legend's: the gizmo under it does not move");
  await still("gizmo-5-legend-over-gizmo");
  off();
  removePlanes();
  store.dispatch({ type: "frameSelection" });
}

/**
 * VA1 (D7): `<out>.camera-plus-x.png`, the `+x` named view through the store;
 * `<out>.camera-plus-x-reread.png`, the same after the model file is read
 * again (a re-read keeps the camera); and `<out>.camera-plus-x-again.png`,
 * the view asked for again after a turn away. Each must show the same frame
 * (pixel-identical, and the same pose to the bit). The camera is then put
 * back at the pose it had, through an explicit `setCamera`, for the stills
 * after it.
 */
async function captureCamera(settle: (ms?: number) => Promise<unknown>, file: string) {
  viewport.renderNow();
  const before = store.get().camera.pose;
  if (!before) throw new Error("capture: the store holds no camera after the model was framed");
  const show = async (name: string) => {
    await settle();
    viewport.renderNow();
    await bridge.captureStill(name);
    return store.get().camera.pose;
  };
  store.dispatch({ type: "setCamera", to: "+x" });
  const first = await show("camera-plus-x");
  const same = (what: string, pose: unknown) => {
    if (JSON.stringify(first) !== JSON.stringify(pose)) throw new Error(`capture: ${what} changed the camera: ${JSON.stringify(first)} vs ${JSON.stringify(pose)}`);
  };
  if (!(await effects.open(file))) throw new Error(`capture: the re-read of ${file} failed`);
  same("a re-read of the model", await show("camera-plus-x-reread"));
  store.dispatch({ type: "setCamera", to: { yaw: 0.7, tilt: 1.1 } });
  viewport.renderNow();
  store.dispatch({ type: "setCamera", to: "+x" });
  same("asking for +x again", await show("camera-plus-x-again"));
  store.dispatch({ type: "setCamera", to: before });
  viewport.renderNow();
}

/**
 * ADR 0128 D9: the page elements each app layer omits from the layout. A
 * layer that is off sets `data-still-omit` on its elements (style.css:
 * `display: none`); every panel is an overlay on the fixed viewport, so the
 * viewport keeps its position and size. Model mode omits every element of
 * the page but the viewport.
 */
const PANEL_LAYERS: Readonly<Record<AppLayer, readonly string[]>> = {
  header: ["header"],
  phase: ["phase"],
  viewcube: ["viewcube"],
  // #scale-stack holds the contour's and line diagram's colour bars, which #2701 moved off their left-rail cards.
  left_rail: ["left-rail", "scale-stack"],
  rail: ["rail", "rail-toggle"],
  windows: ["windows-menu", "windows"],
  banner: ["banner"],
};

const stillLayout: Layout = {
  apply(chrome, layers) {
    document.body.dataset["stillChrome"] = chrome;
    const omitted: HTMLElement[] = [];
    for (const layer of APP_LAYERS) {
      if (chrome === "app" && typeof layers[layer] !== "boolean") throw new Error(`render: app layer ${layer} has no state`);
      for (const id of PANEL_LAYERS[layer]) {
        const node = document.getElementById(id);
        if (!node) throw new Error(`render: the ${layer} layer's element #${id} is not on the page`);
        const off = chrome === "app" && layers[layer] === false;
        node.toggleAttribute("data-still-omit", off);
        if (off) omitted.push(node);
      }
    }
    // The DOM check: an omitted panel has no layout box (never merely covered or transparent).
    const left = omitted.filter((n) => n.getClientRects().length > 0);
    if (left.length) throw new Error(`render: ${left.map((n) => `#${n.id}`).join(", ")} still in the layout after being omitted`);
  },
  inLayout() {
    return [...document.body.children]
      .filter((n) => n.id !== "viewport" && n.getClientRects().length > 0)
      .map((n) => (n.id ? `#${n.id}` : n.tagName.toLowerCase()));
  },
};

/** ADR 0130 (VU3b): main's door for the set's `.user.h5` (main/preload.ts `readUser`, `user:read`); the Effects bridge does not carry it. */
interface UserBridge {
  readUser(req: { path: string; model: string | null; results: string | null }): Promise<UserReadIn>;
  onOpen(cb: (set: { model: string | null; results: string | null; user?: string | null }) => void): () => void;
  onFileChanged(cb: (path: string) => void): () => void;
}

/** One `.user.h5` read: the file and the set's model and results files its staleness is judged against (A6). */
interface UserTarget {
  path: string;
  model: string | null;
  results: string | null;
  /** main named the file in the set (`OpenSet.user`); otherwise `path` is where one would be, and may hold nothing */
  known: boolean;
}

const userDoor = bridge as unknown as UserBridge;
/** Counts `user:read` requests: an answer that is not the latest request's is dropped. */
let userSeq = 0;

/**
 * Read the set's `.user.h5` into the store; resolves once the answer is
 * there. No file there closes it (`userEventOf`); an answer the app cannot
 * take (an unknown target, a malformed reply) is a refusal with the reason,
 * never a partial listing. The read is announced (`userOpened`, the "reading"
 * badge) only for a file the set has: main named it, or it was read before.
 * A look where no file is known says nothing until a file answers (#2737
 * delta review, item 2).
 */
async function readUser(t: UserTarget): Promise<void> {
  const seq = ++userSeq;
  const announced = t.known || store.get().user?.path === t.path;
  if (announced) store.dispatch({ type: "userOpened", path: t.path });
  let read: UserReadIn;
  try {
    read = await userDoor.readUser(t);
  } catch (err) {
    read = { ok: false, error: `user:read failed: ${err instanceof Error ? err.message : String(err)}` };
  }
  if (seq !== userSeq) return;
  const event = userEventOf(t.path, read);
  if (!announced) {
    if (event.type === "userClosed") return;
    store.dispatch({ type: "userOpened", path: t.path });
  }
  try {
    store.dispatch(event);
  } catch (err) {
    store.dispatch({ type: "userRead", path: t.path, read: { ok: false, error: `the app cannot take user:read's answer: ${err instanceof Error ? err.message : String(err)}` } });
  }
}

/**
 * ADR 0130, view mode: each open set's `.user.h5` (main's `user` slot, else
 * `<stem>.user.h5`, which may appear later), read once the model's first read
 * has landed, so the first frame never waits on it; and again whenever the
 * user file, the model or the results change, because A6's staleness depends
 * on all three (#2729). The file appearing, being replaced or being deleted
 * arrives as `fileChanged` of its path, never as a reopen.
 */
function mountUserFeed(): void {
  let target: UserTarget | null = null;
  let pending = false;
  const go = () => {
    if (!pending || !target || !userReadable(store.get(), target.model)) return;
    pending = false;
    void readUser(target);
  };
  panels.push(store.subscribe(go));
  panels.push(
    userDoor.onOpen((set) => {
      const path = set.user ?? userPathOf(set);
      const moved = path !== target?.path;
      target = path === null ? null : { path, model: set.model, results: set.results, known: set.user !== null && set.user !== undefined };
      if (target === null) {
        userSeq++;
        pending = false;
        store.dispatch({ type: "userClosed" });
        return;
      }
      // Another set's user file: the last one's families leave at once, and an answer still on its way for it is dropped
      // (#2737 review, F1). Only the read itself waits for the model's first read; the same file again keeps waiting as before.
      if (moved) {
        userSeq++;
        store.dispatch(target.known ? { type: "userOpened", path: target.path } : { type: "userClosed" });
      }
      pending = true;
      go();
    }),
  );
  panels.push(
    userDoor.onFileChanged((path) => {
      if (!target || ![target.path, target.model, target.results].includes(path)) return;
      pending = true;
      go();
    }),
  );
}

/** VA5: main's door for `--present` and `--snapshot` (main/preload.ts); the Effects bridge does not carry them. */
interface OfferBridge {
  onOffers(cb: (offers: OfferIn[]) => void): () => void;
  onSnapshotRequest(cb: (req: { id: number; image: [number, number] }) => void): () => void;
  snapshotReply(id: number, reply: unknown): Promise<unknown>;
  offerOpen(path: string, needs: readonly string[]): Promise<unknown>;
}

/**
 * ADR 0128 D5, D6, view mode only: an offer that reaches the window is listed
 * and nothing else (INV-5); the runner carries out the human's clicks. A
 * snapshot paints nothing and dispatches nothing: main has captured the page,
 * and this reads the same view back as a spec and its manifest entry.
 */
function mountOfferFlow(): void {
  const door = bridge as unknown as OfferBridge;
  const runner = new OfferRunner({
    store,
    open: (path, needs) => door.offerOpen(path, needs),
    onScreen: (s) => onScreenOf(s, viewport, effects),
    pose: () => (store.get().mesh ? viewport.currentPose() : null),
    settle: async () => {
      viewport.renderNow();
      await painted();
      await painted();
    },
  });
  panels.push(runner.start());
  panels.push(door.onOffers((offers) => runner.receive(offers)));
  panels.push(
    door.onSnapshotRequest(({ id, image }) => {
      void door.snapshotReply(id, snapshotReplyOf(runner, image));
    }),
  );
}

/** The reply to one `--snapshot`; a view the manifest cannot state is refused (INV-6), never a still without its numbers. */
function snapshotReplyOf(runner: OfferRunner, image: [number, number]) {
  const result = runner.snapshot(image);
  if (!result.ok) return result;
  try {
    const canvas = viewport.renderer.domElement.getBoundingClientRect();
    const manifest = snapshotManifest(store.get(), result.snapshot, {
      image,
      fov: viewport.camera instanceof THREE.PerspectiveCamera ? viewport.camera.fov : null,
      layout: { viewport: [canvas.x, canvas.y, canvas.width, canvas.height], panels: stillLayout.inLayout() },
      blobs,
      step: effects.currentStep(),
    });
    return { ok: true, view: result.snapshot.view, manifest };
  } catch (err) {
    return { ok: false, reason: `the manifest cannot state this view: ${err instanceof Error ? err.message : String(err)}` };
  }
}

async function main() {
  const cfg = await bridge.config();
  // ADR 0128 D4: the agent modes (render, describe) run their own flow; the config's mode union predates them.
  const mode: string = cfg.mode;
  if (mode === "render" || mode === "describe") {
    const renderBridge = bridge as unknown as RenderBridge;
    if (mode === "render") {
      // ADR 0130: the spec's set has a `.user.h5` when one is beside its model; it is read before the shots, so every still states it.
      const job = await renderBridge.renderJob();
      const path = job.mode === "render" ? userPathOf(job) : null;
      if (job.mode === "render" && path !== null) await readUser({ path, model: job.model, results: job.results, known: false });
    }
    await runAgentMode({ store, blobs, effects, viewport, bridge: renderBridge, layout: stillLayout, combination: () => combinations.held });
    return;
  }
  if (cfg.mode === "view") {
    effects.attach();
    // After the effects' own onOpen, so a set's model read has started before the user feed looks at it.
    mountUserFeed();
    mountOfferFlow();
    // With V2f's `onOpen`, main replays the open set on subscription; loading
    // `config().file` here as well would read the model twice.
    if (cfg.file && !effects.hasOpenFeed()) await effects.open(cfg.file);
    return;
  }
  if (!cfg.file || !(await effects.open(cfg.file))) {
    await bridge.fail(store.get().artifacts.model?.error ?? "no --file given");
    return;
  }
  if (cfg.mode === "measure") {
    const result = await measure(cfg.t0, cfg);
    // V4c1: with a results sibling, the contour scrub row too (main prints it).
    const contour = cfg.results ? await measureContour(cfg.results) : null;
    // V4d: and the contour scrub again with the deformed shape on.
    const deformed = cfg.results ? await measureContour(cfg.results, true) : null;
    // V4c2a: the scrub with an element contour on (a verified file with an element field), then deformed too.
    const elements = cfg.results ? await measureContour(cfg.results, false, true) : null;
    const elementsDeformed = elements && cfg.results ? await measureContour(cfg.results, true, true) : null;
    // V4c2b: the element scrub carried to the nodes, one row per node-located mode.
    const elementsAveraged = elements && cfg.results ? await measureContour(cfg.results, false, true, "cell_averaged") : null;
    const elementsNode = elements && cfg.results ? await measureContour(cfg.results, false, true, "node") : null;
    const elementsDiscrete = elements && cfg.results ? await measureContour(cfg.results, false, true, "node_discrete") : null;
    // V4e1: the nodal scrub, deformed, with six section planes cutting (in the JSON as `clip`; main prints no row for it).
    const clip = cfg.results ? await measureContour(cfg.results, true, false, "cell", true) : null;
    // V4e3: the same scrub with the six planes capped (in the JSON as `caps`; main prints no row for it). A model with no solid cell has no row.
    const caps = cfg.results && store.get().mesh!.counts.solidCells > 0 ? await measureContour(cfg.results, true, false, "cell", true, true) : null;
    await bridge.measureDone({ ...result, contour, deform: deformed, elements, elementsDeform: elementsDeformed, elementsAveraged, elementsNode, elementsDiscrete, clip, caps });
    return;
  }
  // capture: select the scripted target and let main grab the page; when the
  // chain is taller than the window, a second still shows its end; a third,
  // `<out>.framed.png`, shows the view after `F` framed the selection.
  const settle = (ms = 300) => new Promise((r) => setTimeout(r, ms));
  viewport.renderNow();
  const tgt = demoTarget(cfg.pick);
  if (tgt) store.dispatch({ type: "select", pick: viewport.pickOf(tgt, null) });
  viewport.renderNow();
  // Past the selection pulse, so the still shows the halo at rest.
  await settle(900);
  viewport.renderNow();
  await bridge.captureStill("");
  const insp = document.getElementById("inspector")!;
  if (!insp.hidden && insp.scrollHeight > insp.clientHeight + 4) {
    insp.scrollTop = insp.scrollHeight;
    await settle();
    viewport.renderNow();
    await bridge.captureStill("chain-end");
    insp.scrollTop = 0;
  }
  if (tgt) {
    window.dispatchEvent(new KeyboardEvent("keydown", { key: "f" }));
    await settle();
    viewport.renderNow();
    await bridge.captureStill("framed");
  }
  // Two palette stills with nothing selected (a selection dims the model):
  // `<out>.groups.png` coloured by physical group, `<out>.role.png` by role.
  // With nothing selected, `frameSelection` frames the whole model again
  // after `F` framed the selection above.
  store.dispatch({ type: "clearSelection" });
  store.dispatch({ type: "frameSelection" });
  await settle();
  viewport.renderNow();
  await bridge.captureStill("groups");
  await captureCamera(settle, cfg.file);
  store.dispatch({ type: "setColourBy", by: "role" });
  await settle();
  viewport.renderNow();
  await bridge.captureStill("role");
  // One still per model-definition window, `<out>.window-<kind>.png`, each
  // opened alone through the store, as the window menu opens it. A window
  // with something to select gets a second still, `<out>.window-<kind>-selected.png`
  // (V4f, D7): its first row that selects a declaration (`.window-row[data-decl]`)
  // selected through the store, or, in a window of stage cards, the last stage
  // selected (`selectStage`). The rule is fixed, so no flag is needed and the
  // stills come from committed code.
  store.dispatch({ type: "setColourBy", by: "group" });
  if (store.get().mesh?.frameExtrusion?.hasExtrudableFrames) {
    store.dispatch({ type: "toggleExtrudeFrames" });
    await settle();
    viewport.renderNow();
    await bridge.captureStill("extrude");
    if (store.get().mesh?.frameExtrusion?.hasJointOffsets) {
      store.dispatch({ type: "toggleShowOffsets" });
      await settle();
      viewport.renderNow();
      await bridge.captureStill("extrude-no-offsets");
      store.dispatch({ type: "toggleShowOffsets" });
    }
    store.dispatch({ type: "toggleExtrudeFrames" });
    await settle();
    viewport.renderNow();
  }
  for (const kind of WINDOW_KINDS) {
    store.dispatch({ type: "openWindow", window: kind });
    await settle();
    viewport.renderNow();
    await bridge.captureStill(`window-${kind}`);
    const row = document.querySelector<HTMLElement>("#windows .window-row[data-decl]");
    const decl = row?.dataset["decl"];
    const cards = document.querySelectorAll("#windows .stage-card").length;
    if (decl) store.dispatch({ type: "select", pick: { decl, at: null, nodes: null } });
    else if (cards) store.dispatch({ type: "selectStage", index: cards - 1 });
    if (decl || cards) {
      await settle();
      viewport.renderNow();
      await bridge.captureStill(`window-${kind}-selected`);
      store.dispatch({ type: "clearSelection" });
      store.dispatch({ type: "selectStage", index: null });
    }
    store.dispatch({ type: "closeWindow", window: kind });
  }
  // V4b (D7): with a results sibling (`<stem>.results.h5`, which main names
  // in `config().results`), one still per results stage,
  // `<out>.results-<i>-<stage name>.png`, at its last step: the scrubber, the
  // stage activation (inactive elements hidden), and the banner when the
  // activation is `ambiguous` or `absent`, or the results do not pair. The
  // rule is fixed, so no flag is needed.
  if (cfg.results) {
    const width = window.innerWidth, height = window.innerHeight;
    await captureResults(cfg.results, settle);
    // V4c1 (D7): the contour stills, at the capture window's own width, then 1280 px.
    await captureContours(settle, width, height);
    // V4d (D7): the deformed-shape stills, at the capture window's own width, then 1280 px.
    await captureDeform(settle, width, height);
    // V4f (D7): the 3-D line-diagram stills (including simultaneous contour + line diagram).
    await captureLineDiagrams(settle, width, height);
    await captureExtrudeResults(settle, width, height);
    // V4c2a (D7): the element-contour stills, at the capture window's own width, then 1280 px.
    await captureElements(settle, width, height);
    // V4e3 (D7): the cut-face stills (or, on a mesh with no volumetric cell, the one saying it gets no cap).
    await captureCaps(settle, width, height);
    // V4e1 (D7): the section-plane stills, at the capture window's own width, then 1280 px.
    await captureClip(settle, width, height);
    // V4e2 (D7): the plane-gizmo stills, at the capture window's own width.
    await captureGizmo(settle, width, height);
  }
  await bridge.captureDone();
}

main().catch((err) => {
  console.error(err);
  void bridge.fail(err instanceof Error ? `${err.message}\n${err.stack}` : String(err));
});
