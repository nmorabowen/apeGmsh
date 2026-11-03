// The Reactions card (`#reactions`): the reaction forces of the results step on
// screen as arrows pointing at the supports, a 45° tick across the tail
// (renderer/loads.ts `ReactionsLayer`). The resultant, or its global-axis
// components, on settings of their own: head size, scale, colour, thinning.
// Auto-hidden unless the step on screen records reaction forces.
//
// Follows ADR 0112 D6 (`panel-allow-list`): imports only `store`, `selectors`,
// `types`, and `ui/dom`.

import { loadsColourPalette, loadsUnitsOf, rampCss, reactionComponentsOf } from "../state/selectors.ts";
import type { State, Store } from "../state/store.ts";
import type { Event, LoadsColorBy, ReactionComponent } from "../state/types.ts";
import { byId, el, listen } from "../ui/dom.ts";

const cssRgb = (rgb: readonly number[]): string => `rgb(${rgb.map((c) => Math.round(c * 255)).join(",")})`;

const currentTheme = (): "dark" | "light" =>
  typeof document !== "undefined" && document.documentElement?.dataset?.["theme"] === "light" ? "light" : "dark";

const COMPONENTS: readonly (readonly [ReactionComponent, string])[] = [
  ["resultant", "resultant"],
  ["xyz", "X, Y, Z components"],
  ["x", "X component"],
  ["y", "Y component"],
  ["z", "Z component"],
];

export function mountReactions(store: Store, subscribeTheme?: (fn: () => void) => () => void): () => void {
  const panel = byId("reactions");
  let unlisten: (() => void)[] = [];
  let lastSig = "";
  const render = (s: State) => {
    const comps = reactionComponentsOf(s);
    const sig = JSON.stringify([s.reactions, comps, s.contour.sourceUnits, s.contour.targetUnits, s.contour.declaredUnits, currentTheme()]);
    if (sig === lastSig) return;
    lastSig = sig;
    for (const u of unlisten) u();
    unlisten = [];
    const content = renderReactions(s, (e) => store.dispatch(e), unlisten);
    panel.hidden = content === null;
    panel.replaceChildren(...(content ? [...content.children] : []));
  };
  render(store.get());
  const unsubscribe = store.subscribe((s) => render(s));
  const unsubTheme = subscribeTheme?.(() => render(store.get()));
  return () => {
    unsubscribe();
    unsubTheme?.();
    for (const u of unlisten) u();
  };
}

export function renderReactions(s: State, dispatch: (e: Event) => void = () => {}, unlisten: (() => void)[] = []): HTMLElement | null {
  const comps = reactionComponentsOf(s);
  if (!s.mesh || !comps) return null;
  const r = s.reactions;
  const palette = loadsColourPalette(currentTheme());
  const box = el("div", "reactions");
  box.append(el("h3", "reactions-header", "Reactions"));

  const modes = el("div", "legend-mode reactions-modes");
  const toggle = el("button", "reactions-toggle" + (r.on ? " on" : ""), "3D reactions");
  toggle.title = "Show the step's reaction forces: arrows pointing at the supports, a 45° tick across the tail";
  const which = el("select", "reactions-component");
  which.title = "the resultant at each node, or its components along the global axes";
  for (const [value, label] of COMPONENTS) {
    // A 2-D model records no z reaction: no Z arrow to offer.
    if (value === "z" && comps[2] === null) continue;
    const o = el("option", undefined, label);
    o.value = value;
    o.selected = r.component === value;
    which.append(o);
  }
  unlisten.push(
    listen(toggle, "click", () => dispatch({ type: "setReactions", on: !r.on })),
    listen(which, "change", () => dispatch({ type: "setReactions", component: which.value as ReactionComponent })),
  );
  modes.append(toggle, which);
  box.append(modes);

  const scaleRow = el("div", "deform-scale-row reactions-scale-row");
  const scaleInput = el("input", "deform-scale reactions-scale");
  scaleInput.type = "number";
  scaleInput.step = "any";
  scaleInput.min = "0";
  scaleInput.value = String(r.scale);
  scaleInput.title = "reaction arrow length multiplier";
  const headInput = el("input", "deform-scale reactions-head");
  headInput.type = "number";
  headInput.step = "1";
  headInput.min = "0";
  headInput.value = String(r.headSize);
  headInput.title = "arrowhead length on screen, in pixels (the head is twice as long as it is wide)";
  const positive = (input: HTMLInputElement, what: string, set: (v: number) => void) =>
    listen(input, "change", () => {
      const v = Number(input.value);
      if (Number.isFinite(v) && v > 0) set(v);
      else input.setCustomValidity(`the ${what} must be a number above 0`);
    });
  unlisten.push(
    positive(scaleInput, "scale", (v) => dispatch({ type: "setReactions", scale: v })),
    positive(headInput, "head size", (v) => dispatch({ type: "setReactions", headSize: v })),
  );
  scaleRow.append(el("span", undefined, "scale"), scaleInput, el("span", undefined, "head"), headInput, el("span", "deform-scale-label", "px"));
  box.append(scaleRow);

  const optsRow = el("div", "legend-mode reactions-options");
  const colorSelect = el("select", "reactions-color");
  colorSelect.title = "arrow colour";
  for (const [value, label] of [["mono", "one colour"], ["type", "reaction colour"], ["magnitude", "by magnitude (jet)"]] as const) {
    const o = el("option", undefined, label);
    o.value = value;
    o.selected = r.colorBy === value;
    colorSelect.append(o);
  }
  const thinBtn = el("button", "reactions-thin" + (r.thin ? " on" : ""), "thin out");
  thinBtn.title = "Draw one arrow where several would overlap on screen (the largest reaction); zoom in to see the rest";
  const allBtn = el("button", "reactions-all" + (r.allNodes ? " on" : ""), "all nodes");
  allBtn.title = "Also draw the reactions recorded away from the supports: a free node's residual, a constrained node's constraint force";
  unlisten.push(
    listen(colorSelect, "change", () => dispatch({ type: "setReactions", colorBy: colorSelect.value as LoadsColorBy })),
    listen(thinBtn, "click", () => dispatch({ type: "setReactions", thin: !r.thin })),
    listen(allBtn, "click", () => dispatch({ type: "setReactions", allNodes: !r.allNodes })),
  );
  optsRow.append(colorSelect, thinBtn, allBtn);
  box.append(optsRow);

  const unit = loadsUnitsOf(s).force.unit;
  const row = el("div", "legend-row reactions-kind-row");
  const chip = el("span", "chip");
  if (r.colorBy === "magnitude") chip.style.background = rampCss("jet");
  else chip.style.backgroundColor = cssRgb(r.colorBy === "type" ? palette.reaction : palette.mono);
  const axes = comps.flatMap((c, i) => (c ? ["xyz"[i]!] : [])).join(", ");
  row.append(chip, el("span", undefined, unit ? `Reaction force [${unit}]` : "Reaction force (R)"), el("span", "legend-count", `${axes} · ${r.allNodes ? "all nodes" : "supports"}`));
  box.append(row);
  return box;
}
