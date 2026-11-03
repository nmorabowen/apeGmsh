// The Springs card (`#springs`): the zero-length spring forces of the results
// step on screen, one arrow per spring along node j's global force, its tip on
// the spring (state/springs.ts `buildSprings3D`, renderer/springs.ts). The
// resultant, or its global-axis components, on settings of their own: head
// size, scale, colour, thinning. Auto-hidden unless the stage on screen
// records a zero-length spring's force (a `.ladruno` `force` bucket).
//
// Follows ADR 0112 D6 (`panel-allow-list`): imports only `store`, `selectors`,
// `types`, and `ui/dom`.

import { loadsColourPalette, loadsUnitsOf, rampCss, springCountOf } from "../state/selectors.ts";
import type { State, Store } from "../state/store.ts";
import type { Event, ReactionComponent, SpringsState } from "../state/types.ts";
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

export function mountSprings(store: Store, subscribeTheme?: (fn: () => void) => () => void): () => void {
  const panel = byId("springs");
  let unlisten: (() => void)[] = [];
  let lastSig = "";
  const render = (s: State) => {
    const sig = JSON.stringify([s.springs, springCountOf(s), s.ndm.ndm, s.contour.sourceUnits, s.contour.targetUnits, s.contour.declaredUnits, currentTheme()]);
    if (sig === lastSig) return;
    lastSig = sig;
    for (const u of unlisten) u();
    unlisten = [];
    const content = renderSprings(s, (e) => store.dispatch(e), unlisten);
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

export function renderSprings(s: State, dispatch: (e: Event) => void = () => {}, unlisten: (() => void)[] = []): HTMLElement | null {
  const count = springCountOf(s);
  if (!s.mesh || count === 0) return null;
  const r = s.springs;
  const palette = loadsColourPalette(currentTheme());
  const box = el("div", "springs");
  box.append(el("h3", "springs-header", "Springs"));

  const modes = el("div", "legend-mode springs-modes");
  const toggle = el("button", "springs-toggle" + (r.on ? " on" : ""), "3D spring forces");
  toggle.title = "Show the step's zero-length spring forces: one arrow per spring along its force, pointing at the spring";
  const which = el("select", "springs-component");
  which.title = "the resultant at each spring, or its components along the global axes";
  for (const [value, label] of COMPONENTS) {
    // A 2-D model has no z force.
    if (value === "z" && s.ndm.ndm === 2) continue;
    const o = el("option", undefined, label);
    o.value = value;
    o.selected = r.component === value;
    which.append(o);
  }
  unlisten.push(
    listen(toggle, "click", () => dispatch({ type: "setSprings", on: !r.on })),
    listen(which, "change", () => dispatch({ type: "setSprings", component: which.value as ReactionComponent })),
  );
  modes.append(toggle, which);
  box.append(modes);

  const scaleRow = el("div", "deform-scale-row springs-scale-row");
  const scaleInput = el("input", "deform-scale springs-scale");
  scaleInput.type = "number";
  scaleInput.step = "any";
  scaleInput.min = "0";
  scaleInput.value = String(r.scale);
  scaleInput.title = "spring arrow length multiplier";
  const headInput = el("input", "deform-scale springs-head");
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
    positive(scaleInput, "scale", (v) => dispatch({ type: "setSprings", scale: v })),
    positive(headInput, "head size", (v) => dispatch({ type: "setSprings", headSize: v })),
  );
  scaleRow.append(el("span", undefined, "scale"), scaleInput, el("span", undefined, "head"), headInput, el("span", "deform-scale-label", "px"));
  box.append(scaleRow);

  const optsRow = el("div", "legend-mode springs-options");
  const colorSelect = el("select", "springs-color");
  colorSelect.title = "arrow colour";
  for (const [value, label] of [["mono", "one colour"], ["magnitude", "by magnitude (jet)"]] as const) {
    const o = el("option", undefined, label);
    o.value = value;
    o.selected = r.colorBy === value;
    colorSelect.append(o);
  }
  const thinBtn = el("button", "springs-thin" + (r.thin ? " on" : ""), "thin out");
  thinBtn.title = "Draw one arrow where several would overlap on screen (the largest force); zoom in to see the rest";
  unlisten.push(
    listen(colorSelect, "change", () => dispatch({ type: "setSprings", colorBy: colorSelect.value as SpringsState["colorBy"] })),
    listen(thinBtn, "click", () => dispatch({ type: "setSprings", thin: !r.thin })),
  );
  optsRow.append(colorSelect, thinBtn);
  box.append(optsRow);

  const unit = loadsUnitsOf(s).force.unit;
  const row = el("div", "legend-row springs-kind-row");
  const chip = el("span", "chip");
  if (r.colorBy === "magnitude") chip.style.background = rampCss("jet");
  else chip.style.backgroundColor = cssRgb(palette.mono);
  row.append(chip, el("span", undefined, unit ? `Spring force [${unit}]` : "Spring force"), el("span", "legend-count", `${count} spring${count === 1 ? "" : "s"} · node j, global`));
  box.append(row);
  return box;
}
