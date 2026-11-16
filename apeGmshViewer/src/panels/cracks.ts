// The Cracks card (`#cracks`): at each Gauss point of a smeared-crack material's
// `crackPattern` (ASDConcrete3D), a disc in each crack plane, coloured by the
// opening (renderer/cracks.ts `CracksLayer`, state/cracks.ts), the 3-D twin of
// Python's `results.plot.cracks`. Auto-hidden unless the stage on screen has a
// crack pattern a verified file places at its Gauss points.
//
// Follows ADR 0112 D6 (`panel-allow-list`): imports only `store`, `selectors`,
// `types`, and `ui/dom`.

import { CMAP_NAMES, crackFieldOf, rampCss } from "../state/selectors.ts";
import type { State, Store } from "../state/store.ts";
import type { ContourCmap, Event } from "../state/types.ts";
import { byId, el, listen } from "../ui/dom.ts";

export function mountCracks(store: Store): () => void {
  const panel = byId("cracks");
  let unlisten: (() => void)[] = [];
  let lastSig = "";
  const render = (s: State) => {
    const f = crackFieldOf(s);
    const sig = JSON.stringify([s.cracks, f && [f.groups.map((g) => g.id), f.refused]]);
    if (sig === lastSig) return;
    lastSig = sig;
    for (const u of unlisten) u();
    unlisten = [];
    const content = renderCracks(s, (e) => store.dispatch(e), unlisten);
    panel.hidden = content === null;
    panel.replaceChildren(...(content ? [...content.children] : []));
  };
  render(store.get());
  const unsubscribe = store.subscribe((s) => render(s));
  return () => {
    unsubscribe();
    for (const u of unlisten) u();
  };
}

export function renderCracks(s: State, dispatch: (e: Event) => void = () => {}, unlisten: (() => void)[] = []): HTMLElement | null {
  const field = crackFieldOf(s);
  if (!field) return null;
  const c = s.cracks;
  const box = el("div", "cracks");
  box.append(el("h3", "cracks-header", "Cracks"));

  const modes = el("div", "legend-mode cracks-modes");
  const toggle = el("button", "cracks-toggle" + (c.on ? " on" : ""), "3D cracks");
  toggle.title = "Draw a disc in each crack plane at every Gauss point, coloured by the crack's opening";
  const cmapSelect = el("select", "cracks-cmap");
  cmapSelect.title = "the colour of the opening, over the minimum opening..the largest opening on screen";
  for (const name of CMAP_NAMES) {
    const o = el("option", undefined, name);
    o.value = name;
    o.selected = c.cmap === name;
    cmapSelect.append(o);
  }
  unlisten.push(
    listen(toggle, "click", () => dispatch({ type: "setCracks", on: !c.on })),
    listen(cmapSelect, "change", () => dispatch({ type: "setCracks", cmap: cmapSelect.value as ContourCmap })),
  );
  modes.append(toggle, cmapSelect);
  box.append(modes);

  const row = el("div", "deform-scale-row cracks-row");
  const minInput = el("input", "deform-scale cracks-min");
  minInput.type = "number";
  minInput.step = "any";
  minInput.min = "0";
  minInput.value = String(c.minOpening);
  minInput.title = "a crack whose opening is below this is not drawn (the file's length unit)";
  const scaleInput = el("input", "deform-scale cracks-scale");
  scaleInput.type = "number";
  scaleInput.step = "any";
  scaleInput.min = "0";
  scaleInput.value = String(c.scale);
  scaleInput.title = "disc size multiplier: 1 is 0.4 x the Gauss point spacing across";
  unlisten.push(
    listen(minInput, "change", () => {
      const v = Number(minInput.value);
      if (minInput.value.trim() !== "" && Number.isFinite(v) && v >= 0) dispatch({ type: "setCracks", minOpening: v });
      else minInput.setCustomValidity("the minimum opening must be a number of 0 or more");
    }),
    listen(scaleInput, "change", () => {
      const v = Number(scaleInput.value);
      if (Number.isFinite(v) && v > 0) dispatch({ type: "setCracks", scale: v });
      else scaleInput.setCustomValidity("the size must be a number above 0");
    }),
  );
  row.append(el("span", undefined, "min opening"), minInput, el("span", undefined, "size"), scaleInput);
  box.append(row);

  const ramp = el("div", "legend-row cracks-ramp-row");
  const chip = el("span", "chip cracks-ramp");
  chip.style.background = rampCss(c.cmap);
  chip.title = "the colour of the opening, over the minimum opening..the largest opening of the step on screen";
  ramp.append(el("span", "cracks-range-min", c.minOpening > 0 ? `${c.minOpening}` : "0"), chip, el("span", "cracks-range-max", "largest"));
  box.append(ramp);

  box.append(el("div", "legend-note cracks-note", "inside a solid: lower the surface opacity, hide the faces or cut with a section plane to see them"));
  for (const why of field.refused) box.append(el("div", "legend-note cracks-note", `not drawn: ${why}`));
  return box;
}
