// The Cracks card's stills (`npm run capture`), only on a results stage that records
// a crack pattern (state/cracks.ts): `<out>.cracks-<n>-<what>.png`; on layered
// shells (state/shellCracks.ts): `<out>.shellcracks-<n>-<what>.png`.

import { gaussFields } from "../../state/contour.ts";
import { contourView, crackFieldOf, shellCrackFieldOf } from "../../state/selectors.ts";
import { painted, until, type HarnessContext } from "./context.ts";

/**
 * At the last step of the first stage with a crack pattern, the faces at 0.25
 * opacity so the discs inside the solid show:
 * (1) `cracks-1-iso`: every crack that opened (min opening 0), iso view;
 * (2) `cracks-2-side`: the same from -y with the faces hidden, the discs
 *     edge-on: the view Python's `results.plot.cracks(plane="xz")` draws as segments;
 * (3) `cracks-3-min-opening`: only the openings of at least half the largest, iso;
 * (4) `cracks-4-deformed`: (2) on the deformed shape at scale 10, from iso.
 * The card is shown open. The state is put back afterwards (card off and
 * collapsed, opaque faces, undeformed).
 */
export async function captureCracks(ctx: HarnessContext, settle: (ms?: number) => Promise<unknown>, width: number, height: number) {
  const { store, viewport, bridge } = ctx;
  const s0 = store.get();
  if (!s0.results || !s0.phase.axis.some((k) => k.kind === "results")) return;
  const stageIdx = s0.results.stages.findIndex((_, i) => crackFieldOf({ ...s0, phase: { ...s0.phase, at: { kind: "results", stage: i, step: 0 } } })?.groups.length);
  if (stageIdx < 0) return captureShellCracks(ctx, settle, width, height);
  if (typeof bridge.captureResize === "function") await bridge.captureResize(width, height);
  const last = Math.max(0, s0.results.stages[stageIdx]!.time.length - 1);
  const still = async (name: string) => {
    await until(() => store.get().resultStep?.status === "ready" && viewport.cracks.drawn.discs > 0, 15000);
    await settle(150);
    viewport.renderNow();
    await painted();
    await painted();
    await bridge.captureStill(name);
  };
  store.dispatch({ type: "clearSelection" });
  store.dispatch({ type: "setColourBy", by: "group" });
  store.dispatch({ type: "setOpacity", value: 0.25 });
  store.dispatch({ type: "setPhase", at: { kind: "results", stage: stageIdx, step: last } });
  store.dispatch({ type: "setCracks", on: true, minOpening: 0 });
  store.dispatch({ type: "setCamera", to: "iso" });
  // The card starts collapsed: open it for these stills, and close it again after.
  const card = document.getElementById("cracks");
  const opened = !!card && !card.hidden && card.dataset["collapsed"] === "true";
  if (opened) card.querySelector("h3")?.click();
  await still("cracks-1-iso");
  store.dispatch({ type: "setOpacity", value: 0 });
  store.dispatch({ type: "setCamera", to: "-y" });
  await still("cracks-2-side");
  store.dispatch({ type: "setOpacity", value: 0.25 });
  store.dispatch({ type: "setCamera", to: "iso" });
  store.dispatch({ type: "setCracks", minOpening: 0.5 * viewport.cracks.drawn.maxOpening });
  await still("cracks-3-min-opening");
  store.dispatch({ type: "setDeform", on: true });
  store.dispatch({ type: "setDeformScale", scale: 10 });
  await still("cracks-4-deformed");
  store.dispatch({ type: "setDeform", on: false });
  store.dispatch({ type: "setDeformScale", scale: 1 });
  store.dispatch({ type: "setCracks", on: false, minOpening: 0 });
  store.dispatch({ type: "setOpacity", value: 1 });
  if (opened && card.dataset["collapsed"] !== "true") card.querySelector("h3")?.click();
  await settle();
  viewport.renderNow();
}

/**
 * At the last step of the first stage with a layered-shell crack pattern, opaque faces, the card open, the traced
 * crack lines (the card's default style):
 * (1) `shellcracks-1-iso`: both faces' lines (red toward the positive axis, blue the other), iso view;
 * (2) `shellcracks-2-plus-y`: from +y, the red face alone (a wall's +y side, a slab's top);
 * (3) `shellcracks-3-minus-y`: from -y, the blue face alone;
 * (4) `shellcracks-4-deformed`: (1) on the deformed shape at scale 10;
 * (5) `shellcracks-5-extruded`: (1) on the shells drawn by their thickness, when the model gives one;
 * (6) `shellcracks-6-ticks`: (2) as ticks, one per crack at each Gauss point;
 * then, with the cracks off, the damage contours of the layered shells (reader/layerVariants.ts), when recorded:
 * (7) `shelldamage-7-max`: d+ as the largest over the layers, iso;
 * (8) `shelldamage-8-plus-y`: d+ of the + face, from +y; (9) `shelldamage-9-minus-y`: d+ of the - face, from -y.
 * The state is put back afterwards.
 */
async function captureShellCracks(ctx: HarnessContext, settle: (ms?: number) => Promise<unknown>, width: number, height: number) {
  const { store, viewport, bridge } = ctx;
  const s0 = store.get();
  const stageIdx = s0.results!.stages.findIndex((_, i) => shellCrackFieldOf({ ...s0, phase: { ...s0.phase, at: { kind: "results", stage: i, step: 0 } } })?.groups.length);
  if (stageIdx < 0) return;
  if (typeof bridge.captureResize === "function") await bridge.captureResize(width, height);
  const last = Math.max(0, s0.results!.stages[stageIdx]!.time.length - 1);
  const still = async (name: string) => {
    await until(() => store.get().resultStep?.status === "ready" && (viewport.cracks.drawnShell.ticks > 0 || viewport.cracks.drawnLines.segments > 0), 15000);
    await settle(150);
    viewport.renderNow();
    await painted();
    await painted();
    await bridge.captureStill(name);
  };
  store.dispatch({ type: "clearSelection" });
  store.dispatch({ type: "setColourBy", by: "group" });
  store.dispatch({ type: "setPhase", at: { kind: "results", stage: stageIdx, step: last } });
  store.dispatch({ type: "setCracks", on: true, minOpening: 0, face: "both", style: "lines" });
  store.dispatch({ type: "setCamera", to: "iso" });
  const card = document.getElementById("cracks");
  const opened = !!card && !card.hidden && card.dataset["collapsed"] === "true";
  if (opened) card.querySelector("h3")?.click();
  await still("shellcracks-1-iso");
  store.dispatch({ type: "setCracks", face: "plus" });
  store.dispatch({ type: "setCamera", to: "+y" });
  await still("shellcracks-2-plus-y");
  store.dispatch({ type: "setCracks", face: "minus" });
  store.dispatch({ type: "setCamera", to: "-y" });
  await still("shellcracks-3-minus-y");
  store.dispatch({ type: "setCracks", face: "both" });
  store.dispatch({ type: "setCamera", to: "iso" });
  store.dispatch({ type: "setDeform", on: true });
  store.dispatch({ type: "setDeformScale", scale: 10 });
  await still("shellcracks-4-deformed");
  store.dispatch({ type: "setDeform", on: false });
  store.dispatch({ type: "setDeformScale", scale: 1 });
  if (store.get().mesh?.shellThickness) {
    store.dispatch({ type: "toggleExtrudeFrames", on: true });
    await still("shellcracks-5-extruded");
    store.dispatch({ type: "toggleExtrudeFrames", on: false });
  }
  store.dispatch({ type: "setCracks", face: "plus", style: "ticks" });
  store.dispatch({ type: "setCamera", to: "+y" });
  await still("shellcracks-6-ticks");
  store.dispatch({ type: "setCracks", on: false, minOpening: 0, face: "both", style: "lines" });
  if (opened && card.dataset["collapsed"] !== "true") card.querySelector("h3")?.click();
  const fields = gaussFields(store.get().results!.stages[stageIdx]!.gauss);
  const damage = (v: string) => fields.find((f) => f.name === `section.fiber.damage@${v}` && f.refused === null);
  if (damage("max")) {
    const contour = async (v: string, camera: "iso" | "+y" | "-y", name: string) => {
      const f = damage(v);
      if (!f) return;
      store.dispatch({ type: "setCamera", to: camera });
      store.dispatch({ type: "setContourField", field: f.name, component: f.components[0]!.key, source: "element" });
      await until(() => {
        const cv = contourView(store.get());
        return cv.state === "element" && cv.stepMeasured;
      }, 15000);
      await settle(150);
      viewport.renderNow();
      await painted();
      await painted();
      await bridge.captureStill(name);
    };
    store.dispatch({ type: "setColourBy", by: "contour" });
    store.dispatch({ type: "setContourRange", range: "step" });
    await contour("max", "iso", "shelldamage-7-max");
    await contour("+face", "+y", "shelldamage-8-plus-y");
    await contour("-face", "-y", "shelldamage-9-minus-y");
    store.dispatch({ type: "setColourBy", by: "group" });
    store.dispatch({ type: "setCamera", to: "iso" });
  }
  await settle();
  viewport.renderNow();
}
