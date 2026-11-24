// The capture harness (VH1, #2865): `npm run capture`'s sequence of stills, in its fixed order.

import { WINDOW_KINDS } from "../../state/types.ts";
import type { HarnessConfig, HarnessContext } from "./context.ts";
import { demoTarget } from "./pointer.ts";
import { captureContours, captureDeform, captureElements, captureExtrudeResults, captureLineDiagrams, captureResults } from "./results.ts";
import { captureCaps, captureClip, captureGizmo } from "./sections.ts";
import { captureCracks } from "./cracks.ts";
import { captureHinges } from "./hinges.ts";
import { captureMaterialTest, captureMaterialTestNd } from "./materialTest.ts";
import { captureSelection } from "./selection.ts";
import { captureWorkspace } from "./workspace.ts";
import { captureCombine } from "./combine.ts";
import { resetView } from "./reset.ts";

/**
 * VA1 (D7): `<out>.camera-plus-x.png`, the `+x` named view through the store;
 * `<out>.camera-plus-x-reread.png`, the same after the model file is read
 * again (a re-read keeps the camera); and `<out>.camera-plus-x-again.png`,
 * the view asked for again after a turn away. Each must show the same frame
 * (pixel-identical, and the same pose to the bit). The camera is then put
 * back at the pose it had, through an explicit `setCamera`, for the stills
 * after it.
 */
export async function captureCamera(ctx: HarnessContext, settle: (ms?: number) => Promise<unknown>, file: string) {
  const { store, effects, viewport, bridge } = ctx;
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
 * The capture sequence (`--mode=capture`), after main opened `file`: every still, in this order, then `captureDone`.
 * Each step after the first stills starts from `resetView` (./reset.ts): the model main opened, at the size, phase,
 * colouring and camera it opened on, so no still depends on what the steps before it left.
 */
export async function runCapture(ctx0: HarnessContext, cfg: HarnessConfig, file: string): Promise<void> {
  const { store, viewport, bridge } = ctx0;
  // capture: select the scripted target and let main grab the page; when the
  // chain is taller than the window, a second still shows its end; a third,
  // `<out>.framed.png`, shows the view after `F` framed the selection.
  const settle = (ms = 300) => new Promise((r) => setTimeout(r, ms));
  viewport.renderNow();
  const opened = store.get().camera.pose;
  if (!opened) throw new Error("capture: the store holds no camera after the model was framed");
  const width = window.innerWidth, height = window.innerHeight;
  const ctx: HarnessContext = { ...ctx0, baseline: { model: store.get().artifacts.model?.path ?? file, camera: opened, width, height } };
  /** One step of the sequence, from the baseline view. */
  const step = async (run: () => Promise<unknown>) => {
    await resetView(ctx);
    await settle();
    viewport.renderNow();
    await run();
  };
  const tgt = demoTarget(ctx, cfg.pick);
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
  await step(() => captureCamera(ctx, settle, file));
  await step(() => captureSelection(ctx, settle));
  await step(async () => {
    store.dispatch({ type: "setColourBy", by: "role" });
    await settle();
    viewport.renderNow();
    await bridge.captureStill("role");
  });
  // One still per model-definition window, `<out>.window-<kind>.png`, each
  // opened alone through the store, as the window menu opens it. A window
  // with something to select gets a second still, `<out>.window-<kind>-selected.png`
  // (V4f, D7): its first row that selects a declaration (`.window-row[data-decl]`)
  // selected through the store, or, in a window of stage cards, the last stage
  // selected (`selectStage`). The rule is fixed, so no flag is needed and the
  // stills come from committed code.
  await step(async () => {
    store.dispatch({ type: "setColourBy", by: "group" });
    // Frames by their cross-section, shells and membranes by their thickness: one toggle.
    if (store.get().mesh?.frameExtrusion?.hasExtrudableFrames || store.get().mesh?.shellThickness) {
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
  });
  // ADR 0135 Amendment 1 (D7): the material tester in the materials window, only on a model with a uniaxial material (./materialTest.ts).
  await step(() => captureMaterialTest(ctx, settle));
  await step(() => captureMaterialTestNd(ctx, settle));
  // V4b (D7): with a results sibling (`<stem>.results.h5`, which main names
  // in `config().results`), one still per results stage,
  // `<out>.results-<i>-<stage name>.png`, at its last step: the scrubber, the
  // stage activation (inactive elements hidden), and the banner when the
  // activation is `ambiguous` or `absent`, or the results do not pair. The
  // rule is fixed, so no flag is needed.
  const results = cfg.results;
  if (results) {
    await step(() => captureResults(ctx, results, settle));
    // The Hinges card's stills, only on a model that declares beam hinges (./hinges.ts).
    await step(() => captureHinges(ctx, settle, width, height));
    // V4c1 (D7): the contour stills, at the capture window's own width, then 1280 px.
    await step(() => captureContours(ctx, settle, width, height));
    // V4d (D7): the deformed-shape stills, at the capture window's own width, then 1280 px.
    await step(() => captureDeform(ctx, settle, width, height));
    // V4f (D7): the 3-D line-diagram stills (including simultaneous contour + line diagram).
    await step(() => captureLineDiagrams(ctx, settle, width, height));
    await step(() => captureExtrudeResults(ctx, settle, width, height));
    // V4c2a (D7): the element-contour stills, at the capture window's own width, then 1280 px.
    await step(() => captureElements(ctx, settle, width, height));
    // V4e3 (D7): the cut-face stills (or, on a mesh with no volumetric cell, the one saying it gets no cap).
    await step(() => captureCaps(ctx, settle, width, height));
    // V4e1 (D7): the section-plane stills, at the capture window's own width, then 1280 px.
    await step(() => captureClip(ctx, settle, width, height));
    // V4e2 (D7): the plane-gizmo stills, at the capture window's own width.
    await step(() => captureGizmo(ctx, settle, width, height));
    // The Cracks card's stills, only on a stage that records a crack pattern (./cracks.ts).
    await step(() => captureCracks(ctx, settle, width, height));
  } else await step(() => captureHinges(ctx, settle, width, height));
  // VU9a (D7): the sets panel and the workspace's notices, only for a set with a <stem>.workspace.h5 (./workspace.ts).
  await step(() => captureWorkspace(ctx, settle));
  // VU9b (D7): the combination authoring panel, only for a set whose <stem>.user.h5 main read (./combine.ts).
  await step(() => captureCombine(ctx, settle));
  await bridge.captureDone();
}
