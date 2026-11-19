// The Hinges card's stills (`npm run capture`), only on a model that declares
// beam hinges (state/hinges.ts): `<out>.hinges-<n>-<what>.png`.

import { hingeSitesOf, hingeViewOf } from "../../state/selectors.ts";
import { painted, until, type HarnessContext } from "./context.ts";

/**
 * With the card open:
 * (1) `hinges-1-model`: the markers on the model, no results step;
 * and, when a stage records `hingeSummary`, at its last step on the deformed shape:
 * (2) `hinges-2-rotation`: markers by rotation, the first element's curves;
 * (3) `hinges-3-capacity`: markers by moment/capacity, the last element's end j;
 * (4) `hinges-4-mid-step`: the same at the middle step, the marker partway along the curve.
 * The state is put back afterwards.
 */
export async function captureHinges(ctx: HarnessContext, settle: (ms?: number) => Promise<unknown>, width: number, height: number) {
  const { store, viewport, bridge } = ctx;
  const s0 = store.get();
  const { sites } = hingeSitesOf(s0.decls);
  if (sites.length === 0) return;
  if (typeof bridge.captureResize === "function") await bridge.captureResize(width, height);
  const still = async (name: string, ready: () => boolean) => {
    await until(() => viewport.hinges.drawn.count > 0 && ready(), 15000);
    await settle(150);
    viewport.renderNow();
    await painted();
    await painted();
    await bridge.captureStill(name);
  };
  const before = s0.phase.at;
  store.dispatch({ type: "clearSelection" });
  store.dispatch({ type: "setHinges", on: true });
  const card = document.getElementById("hinges");
  const opened = !!card && !card.hidden && card.dataset["collapsed"] === "true";
  if (opened) card.querySelector("h3")?.click();
  store.dispatch({ type: "setCamera", to: "iso" });
  store.dispatch({ type: "setPhase", at: { kind: "mesh" } });
  await still("hinges-1-model", () => true);
  const stage = s0.results?.stages.findIndex((st) => (st.hinges ?? 0) > 0) ?? -1;
  if (stage >= 0) {
    const last = Math.max(0, s0.results!.stages[stage]!.time.length - 1);
    const charted = () => store.get().resultStep?.status === "ready" && hingeViewOf(store.get()).chart !== null;
    store.dispatch({ type: "setPhase", at: { kind: "results", stage, step: last } });
    store.dispatch({ type: "setDeform", on: true });
    store.dispatch({ type: "setDeformScale", scale: 5 });
    store.dispatch({ type: "setHinges", element: sites[0]!.decl });
    await still("hinges-2-rotation", charted);
    store.dispatch({ type: "setHinges", colorBy: "capacity", element: sites[sites.length - 1]!.decl, end: "j" });
    await still("hinges-3-capacity", charted);
    store.dispatch({ type: "setPhase", at: { kind: "results", stage, step: Math.floor(last / 2) } });
    await still("hinges-4-mid-step", charted);
    store.dispatch({ type: "setDeform", on: false });
    store.dispatch({ type: "setDeformScale", scale: 1 });
  }
  store.dispatch({ type: "setHinges", on: false, colorBy: "rotation", element: null, end: "i" });
  if (before) store.dispatch({ type: "setPhase", at: before });
  if (opened && card.dataset["collapsed"] !== "true") card.querySelector("h3")?.click();
  await settle();
}
