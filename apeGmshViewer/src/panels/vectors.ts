// The Vectors card (`#vectors`): a nodal vector field of the results step on
// screen as arrows starting at the nodes, or half circles about their axes for
// a rotational field (renderer/loads.ts `VectorsLayer`). The field, the
// resultant or its global-axis components, and settings of its own: head size,
// scale, colour, thinning. Auto-hidden unless the stage on screen records a
// nodal vector field.
//
// Follows ADR 0112 D6 (`panel-allow-list`): imports only `store`, `selectors`,
// `types`, and `ui/dom`.

import { loadsColourPalette, rampCss, vectorFieldOf, vectorFieldsOf } from "../state/selectors.ts";
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

export function mountVectors(store: Store, subscribeTheme?: (fn: () => void) => () => void): () => void {
  const panel = byId("vectors");
  let unlisten: (() => void)[] = [];
  let lastSig = "";
  const render = (s: State) => {
    const sig = JSON.stringify([s.vectors, vectorFieldsOf(s), currentTheme()]);
    if (sig === lastSig) return;
    lastSig = sig;
    for (const u of unlisten) u();
    unlisten = [];
    const content = renderVectors(s, (e) => store.dispatch(e), unlisten);
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

export function renderVectors(s: State, dispatch: (e: Event) => void = () => {}, unlisten: (() => void)[] = []): HTMLElement | null {
  const fields = vectorFieldsOf(s);
  const field = vectorFieldOf(s);
  if (!s.mesh || !field) return null;
  const v = s.vectors;
  const palette = loadsColourPalette(currentTheme());
  const box = el("div", "vectors");
  box.append(el("h3", "vectors-header", "Vectors"));

  const modes = el("div", "legend-mode vectors-modes");
  const toggle = el("button", "vectors-toggle" + (v.on ? " on" : ""), "3D vectors");
  toggle.title = "Show the step's nodal vector field: arrows starting at the nodes, or half circles about their axes for a rotation";
  const fieldSelect = el("select", "vectors-field");
  fieldSelect.title = "the nodal vector field drawn";
  for (const f of fields) {
    const o = el("option", undefined, f.name);
    o.value = f.name;
    o.selected = f.name === field.name;
    fieldSelect.append(o);
  }
  const which = el("select", "vectors-component");
  which.title = "the resultant at each node, or its components along the global axes";
  for (const [value, label] of COMPONENTS) {
    // An axis the field does not record (z on a 2-D model) is not offered.
    if (value.length === 1 && !field.axes["xyz".indexOf(value)]) continue;
    const o = el("option", undefined, label);
    o.value = value;
    o.selected = v.component === value;
    which.append(o);
  }
  unlisten.push(
    listen(toggle, "click", () => dispatch({ type: "setVectors", on: !v.on })),
    listen(fieldSelect, "change", () => dispatch({ type: "setVectors", field: fieldSelect.value })),
    listen(which, "change", () => dispatch({ type: "setVectors", component: which.value as ReactionComponent })),
  );
  modes.append(toggle);
  const picks = el("div", "legend-mode vectors-picks");
  picks.append(fieldSelect, which);
  box.append(modes, picks);

  const scaleRow = el("div", "deform-scale-row vectors-scale-row");
  const scaleInput = el("input", "deform-scale vectors-scale");
  scaleInput.type = "number";
  scaleInput.step = "any";
  scaleInput.min = "0";
  scaleInput.value = String(v.scale);
  scaleInput.title = "vector arrow length multiplier";
  const headInput = el("input", "deform-scale vectors-head");
  headInput.type = "number";
  headInput.step = "1";
  headInput.min = "0";
  headInput.value = String(v.headSize);
  headInput.title = "arrowhead length on screen, in pixels (the head is twice as long as it is wide)";
  const positive = (input: HTMLInputElement, what: string, set: (x: number) => void) =>
    listen(input, "change", () => {
      const x = Number(input.value);
      if (Number.isFinite(x) && x > 0) set(x);
      else input.setCustomValidity(`the ${what} must be a number above 0`);
    });
  unlisten.push(
    positive(scaleInput, "scale", (x) => dispatch({ type: "setVectors", scale: x })),
    positive(headInput, "head size", (x) => dispatch({ type: "setVectors", headSize: x })),
  );
  scaleRow.append(el("span", undefined, "scale"), scaleInput, el("span", undefined, "head"), headInput, el("span", "deform-scale-label", "px"));
  box.append(scaleRow);

  const optsRow = el("div", "legend-mode vectors-options");
  const colorSelect = el("select", "vectors-color");
  colorSelect.title = "arrow colour";
  for (const [value, label] of [["mono", "one colour"], ["type", "by type (translation, rotation)"], ["magnitude", "by magnitude (jet)"]] as const) {
    const o = el("option", undefined, label);
    o.value = value;
    o.selected = v.colorBy === value;
    colorSelect.append(o);
  }
  const thinBtn = el("button", "vectors-thin" + (v.thin ? " on" : ""), "thin out");
  thinBtn.title = "Draw one node's vector where several would overlap on screen (the largest); zoom in to see the rest";
  unlisten.push(
    listen(colorSelect, "change", () => dispatch({ type: "setVectors", colorBy: colorSelect.value as LoadsColorBy })),
    listen(thinBtn, "click", () => dispatch({ type: "setVectors", thin: !v.thin })),
  );
  optsRow.append(colorSelect, thinBtn);
  box.append(optsRow);

  const row = el("div", "legend-row vectors-kind-row");
  const chip = el("span", "chip");
  if (v.colorBy === "magnitude") chip.style.background = rampCss("jet");
  else chip.style.backgroundColor = cssRgb(v.colorBy === "type" ? (field.rotational ? palette.moment : palette.force) : palette.mono);
  const axes = field.axes.flatMap((c, i) => (c ? ["xyz"[i]!] : [])).join(", ");
  row.append(chip, el("span", undefined, field.name), el("span", "legend-count", `${axes} · ${field.rotational ? "half circles" : "from the nodes"}`));
  box.append(row);
  return box;
}
