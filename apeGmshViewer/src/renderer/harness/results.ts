// The capture harness (VH1, #2865): the results stills (D7): contours, the deformed shape, line diagrams,
// extruded frames and element contours.

import { gaussFields } from "../../state/contour.ts";
import { contourFieldsOf, contourView, deformOf, deformView } from "../../state/selectors.ts";
import { painted, until, type HarnessContext } from "./context.ts";
import { contourShown, deformShown, refusalShown } from "./shown.ts";

/** The results stills of a capture (see the call site); each waits for its step fetch to settle. */
export async function captureResults(ctx: HarnessContext, path: string, settle: (ms?: number) => Promise<unknown>) {
  const { store, effects, viewport, bridge } = ctx;
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
export async function captureLineDiagrams(ctx: HarnessContext, settle: (ms?: number) => Promise<unknown>, width: number, height: number) {
  const { store, viewport, bridge } = ctx;
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
      contourShown(ctx) &&
      store.get().contour.scan?.status === "done" &&
      deformShown(ctx) &&
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
export async function captureExtrudeResults(ctx: HarnessContext, settle: (ms?: number) => Promise<unknown>, width: number, height: number) {
  const { store, viewport, bridge } = ctx;
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
  await until(() => deformShown(ctx), 15000);
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
  await until(() => contourShown(ctx) && store.get().contour.scan?.status === "done" && deformShown(ctx), 30000);
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
    await until(() => contourShown(ctx) && deformShown(ctx) && store.get().lineDiagram.stepRange !== null, 30000);
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
export async function captureElements(ctx: HarnessContext, settle: (ms?: number) => Promise<unknown>, width: number, height: number) {
  const { store, viewport, bridge } = ctx;
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
    if (v.state === "element") return contourShown(ctx) && store.get().contour.scan?.status === "done" && deformShown(ctx);
    return refusalShown(ctx);
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
export async function captureContours(ctx: HarnessContext, settle: (ms?: number) => Promise<unknown>, width: number, height: number) {
  const { store, viewport, bridge } = ctx;
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
    if (!(await until(() => contourShown(ctx)))) throw new Error(`capture ${name}: the contour did not show the step on screen`);
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
export async function captureDeform(ctx: HarnessContext, settle: (ms?: number) => Promise<unknown>, width: number, height: number) {
  const { store, viewport, bridge } = ctx;
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
    if (!(await until(() => deformShown(ctx) && (store.get().visibility.colourBy !== "contour" || contourShown(ctx))))) throw new Error(`capture ${name}: the deformed shape did not show the step on screen (${deformView(store.get()).state})`);
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
