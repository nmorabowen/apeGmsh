// The Principal card (`#principal`): at each Gauss point of the stress or
// strain field on screen, an arrow along each principal direction
// (renderer/loads.ts `PrincipalLayer`, state/principal.ts), the length from
// |p| and the colour from the signed p on a diverging map, so tension and
// compression read apart. The direction is an axis, not a sense: `axis` draws
// a head each way, `arrow` one along the sign-canonical direction (Qt's
// PrincipalDirectionDiagram). Auto-hidden unless the stage on screen has a
// stress or strain tensor a verified file places at its Gauss points.
//
// Follows ADR 0112 D6 (`panel-allow-list`): imports only `store`, `selectors`,
// `types`, and `ui/dom`.

import { CMAP_NAMES, principalFieldOf, principalFieldsOf, rampCss } from "../state/selectors.ts";
import type { State, Store } from "../state/store.ts";
import type { ContourCmap, Event, PrincipalHead } from "../state/types.ts";
import { byId, el, listen } from "../ui/dom.ts";

const HEADS: readonly (readonly [PrincipalHead, string])[] = [
  ["axis", "axis"],
  ["arrow", "arrow"],
];

export function mountPrincipal(store: Store): () => void {
  const panel = byId("principal");
  let unlisten: (() => void)[] = [];
  let lastSig = "";
  const render = (s: State) => {
    const sig = JSON.stringify([s.principal, principalFieldsOf(s).map((f) => [f.name, f.groups, f.refused])]);
    if (sig === lastSig) return;
    lastSig = sig;
    for (const u of unlisten) u();
    unlisten = [];
    const content = renderPrincipal(s, (e) => store.dispatch(e), unlisten);
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

export function renderPrincipal(s: State, dispatch: (e: Event) => void = () => {}, unlisten: (() => void)[] = []): HTMLElement | null {
  const fields = principalFieldsOf(s);
  const field = principalFieldOf(s);
  if (!field) return null;
  const p = s.principal;
  const box = el("div", "principal");
  box.append(el("h3", "principal-header", "Principal directions"));

  const modes = el("div", "legend-mode principal-modes");
  const toggle = el("button", "principal-toggle" + (p.on ? " on" : ""), "3D principal");
  toggle.title = "Show an arrow along each principal direction at every Gauss point: the length from |p|, the colour from the signed value";
  const which = el("select", "principal-field");
  which.title = "the tensor whose principal directions are drawn";
  for (const f of fields) {
    const o = el("option", undefined, f.name);
    o.value = f.name;
    o.selected = f.name === field.name;
    which.append(o);
  }
  unlisten.push(
    listen(toggle, "click", () => dispatch({ type: "setPrincipal", on: !p.on })),
    listen(which, "change", () => dispatch({ type: "setPrincipal", field: which.value })),
  );
  modes.append(toggle, which);
  box.append(modes);

  // p1 >= p2 >= p3, each on its own switch (at least one stays on), and the heads.
  const values = el("div", "legend-mode principal-values");
  for (let k = 0; k < 3; k++) {
    const b = el("button", `principal-p${k + 1}` + (p.show[k] ? " on" : ""), `p${k + 1}`);
    b.title = ["the largest principal value", "the middle principal value", "the smallest principal value"][k]!;
    unlisten.push(
      listen(b, "click", () => {
        const show = [p.show[0], p.show[1], p.show[2]] as [boolean, boolean, boolean];
        show[k] = !show[k];
        if (show.some(Boolean)) dispatch({ type: "setPrincipal", show });
      }),
    );
    values.append(b);
  }

  const scaleRow = el("div", "deform-scale-row principal-scale-row");
  const scaleInput = el("input", "deform-scale principal-scale");
  scaleInput.type = "number";
  scaleInput.step = "any";
  scaleInput.min = "0";
  scaleInput.value = String(p.scale);
  scaleInput.title = "principal arrow length multiplier";
  const headInput = el("input", "deform-scale principal-head");
  headInput.type = "number";
  headInput.step = "1";
  headInput.min = "0";
  headInput.value = String(p.headSize);
  headInput.title = "arrowhead length on screen, in pixels (the head is twice as long as it is wide)";
  const positive = (input: HTMLInputElement, what: string, set: (v: number) => void) =>
    listen(input, "change", () => {
      const v = Number(input.value);
      if (Number.isFinite(v) && v > 0) set(v);
      else input.setCustomValidity(`the ${what} must be a number above 0`);
    });
  unlisten.push(
    positive(scaleInput, "scale", (v) => dispatch({ type: "setPrincipal", scale: v })),
    positive(headInput, "head size", (v) => dispatch({ type: "setPrincipal", headSize: v })),
  );
  scaleRow.append(el("span", undefined, "scale"), scaleInput, el("span", undefined, "head"), headInput, el("span", "deform-scale-label", "px"));
  box.append(scaleRow);

  const optsRow = el("div", "legend-mode principal-options");
  const headSelect = el("select", "principal-heads");
  headSelect.title = "a principal direction is an axis: a head each way, or one head along the sign-canonical direction";
  for (const [value, label] of HEADS) {
    const o = el("option", undefined, label);
    o.value = value;
    o.selected = p.head === value;
    headSelect.append(o);
  }
  const cmapSelect = el("select", "principal-cmap");
  cmapSelect.title = "the colour of the signed value, over -max..max (a diverging map reads compression against tension)";
  for (const name of CMAP_NAMES) {
    const o = el("option", undefined, name);
    o.value = name;
    o.selected = p.cmap === name;
    cmapSelect.append(o);
  }
  const thinBtn = el("button", "principal-thin" + (p.thin ? " on" : ""), "thin out");
  thinBtn.title = "Draw one point's arrows where several points would overlap on screen (the strongest); zoom in to see the rest";
  unlisten.push(
    listen(headSelect, "change", () => dispatch({ type: "setPrincipal", head: headSelect.value as PrincipalHead })),
    listen(cmapSelect, "change", () => dispatch({ type: "setPrincipal", cmap: cmapSelect.value as ContourCmap })),
    listen(thinBtn, "click", () => dispatch({ type: "setPrincipal", thin: !p.thin })),
  );
  values.append(headSelect);
  box.append(values);
  optsRow.append(cmapSelect, thinBtn);
  box.append(optsRow);

  // The map runs over -max..max of the step drawn: compression at its low end, tension at its high end.
  const ramp = el("div", "legend-row principal-ramp-row");
  const chip = el("span", "chip principal-ramp");
  chip.style.background = rampCss(p.cmap);
  chip.title = "the colour of the signed principal value, over -max..max of the step on screen";
  ramp.append(el("span", "principal-range-min", "compression"), chip, el("span", "principal-range-max", "tension"));
  box.append(ramp);

  // A 2-D group records no zz: it is taken as 0 (plane stress for a stress, plane strain for a strain).
  if (field.groups.some((g) => g.slots[2] === null)) {
    box.append(el("div", "legend-note principal-note", `zz not recorded: taken as 0 (${field.family === "stress" ? "plane stress" : "plane strain"})`));
  }
  if (field.family === "strain") box.append(el("div", "legend-note principal-note", "shear strains read as engineering (γ = 2ε), halved for the tensor"));
  for (const why of field.refused) box.append(el("div", "legend-note principal-note", `not drawn: ${why}`));
  return box;
}
