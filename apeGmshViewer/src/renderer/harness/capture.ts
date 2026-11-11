// The capture harness (VH1, #2865): `npm run capture`'s sequence of stills, in its fixed order.

import { WINDOW_KINDS } from "../../state/types.ts";
import type { HarnessConfig, HarnessContext } from "./context.ts";
import { demoTarget } from "./pointer.ts";
import { captureContours, captureDeform, captureElements, captureExtrudeResults, captureLineDiagrams, captureResults } from "./results.ts";
import { captureCaps, captureClip, captureGizmo } from "./sections.ts";

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

/** The capture sequence (`--mode=capture`), after main opened `file`: every still, in this order, then `captureDone`. */
export async function runCapture(ctx: HarnessContext, cfg: HarnessConfig, file: string): Promise<void> {
  const { store, viewport, bridge } = ctx;
  // capture: select the scripted target and let main grab the page; when the
  // chain is taller than the window, a second still shows its end; a third,
  // `<out>.framed.png`, shows the view after `F` framed the selection.
  const settle = (ms = 300) => new Promise((r) => setTimeout(r, ms));
  viewport.renderNow();
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
  await captureCamera(ctx, settle, file);
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
    await captureResults(ctx, cfg.results, settle);
    // V4c1 (D7): the contour stills, at the capture window's own width, then 1280 px.
    await captureContours(ctx, settle, width, height);
    // V4d (D7): the deformed-shape stills, at the capture window's own width, then 1280 px.
    await captureDeform(ctx, settle, width, height);
    // V4f (D7): the 3-D line-diagram stills (including simultaneous contour + line diagram).
    await captureLineDiagrams(ctx, settle, width, height);
    await captureExtrudeResults(ctx, settle, width, height);
    // V4c2a (D7): the element-contour stills, at the capture window's own width, then 1280 px.
    await captureElements(ctx, settle, width, height);
    // V4e3 (D7): the cut-face stills (or, on a mesh with no volumetric cell, the one saying it gets no cap).
    await captureCaps(ctx, settle, width, height);
    // V4e1 (D7): the section-plane stills, at the capture window's own width, then 1280 px.
    await captureClip(ctx, settle, width, height);
    // V4e2 (D7): the plane-gizmo stills, at the capture window's own width.
    await captureGizmo(ctx, settle, width, height);
  }
  await bridge.captureDone();
}
