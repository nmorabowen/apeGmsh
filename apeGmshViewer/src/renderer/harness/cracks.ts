// The Cracks card's stills (`npm run capture`), only on a results stage that records
// a crack pattern (state/cracks.ts): `<out>.cracks-<n>-<what>.png`.

import { crackFieldOf } from "../../state/selectors.ts";
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
  if (stageIdx < 0) return;
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
