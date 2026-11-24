// The Hinges card (`#hinges`): the Ladruno beams' moment hinges. The switch draws
// a marker at each hinge `model.h5` declares (state/hinges.ts, renderer/hinges.ts),
// coloured on a results step by rotation, moment or moment/capacity from the
// `.ladruno` `hingeSummary`. For the element picked in the viewport (or chosen
// here) it plots the hinge's moment-rotation curve with the step on screen marked.
// The curve's data is prepared in state/ (`hingeViewOf`); the shared chart (`ui/chart.ts`,
// ADR 0137 H2) draws it. What the files do not hold is named.
//
// Follows ADR 0112 D6 (`panel-allow-list`): imports only `store`, `selectors`,
// `types`, and `ui/dom`.

import { hingeViewOf, NODE_SIZE_MAX, NODE_SIZE_MIN, type HingeChart } from "../state/selectors.ts";
import type { State, Store } from "../state/store.ts";
import type { Event, HingeAxis, HingeEnd, HingesState } from "../state/types.ts";
import { byId, el, listen } from "../ui/dom.ts";
import { fmt, xyPlot } from "../ui/chart.ts";

function charts(c: HingeChart): HTMLElement {
  const box = el("div", "hinges-charts");
  const caps = c.capacity === null ? [] : [c.capacity, -c.capacity];
  const k = c.step;
  const { rotationName: rot, momentName: mom, rotation, moment } = c.curve;
  box.append(
    xyPlot({ xs: rotation, ys: moment, xName: rot, yName: mom, title: `${mom} against ${rot}`, caps, mark: k }),
    el("div", "hinges-now-line", `step ${k + 1}: ${mom} ${fmt(moment[k]!)}, ${rot} ${fmt(rotation[k]!)}`),
  );
  return box;
}

function select<T extends string>(cls: string, title: string, options: readonly (readonly [T, string])[], value: T, on: (v: T) => void, unlisten: (() => void)[]): HTMLElement {
  const s = el("select", cls);
  s.title = title;
  for (const [v, label] of options) {
    const o = el("option", undefined, label);
    o.value = v;
    o.selected = v === value;
    s.append(o);
  }
  unlisten.push(listen(s, "change", () => on(s.value as T)));
  return s;
}

const COLOURS: readonly (readonly [HingesState["colorBy"], string])[] = [
  ["rotation", "colour by rotation"],
  ["moment", "colour by moment"],
  ["capacity", "colour by moment / capacity"],
];

/** The size row's value, label and auto state, without rebuilding it: a slider being dragged stays under the pointer. */
function showSize(slider: HTMLInputElement | null, label: HTMLElement | null, auto: HTMLElement | null, h: HingesState): void {
  if (slider && h.size !== null) slider.value = String(h.size);
  if (label) label.textContent = h.size === null ? "" : `${h.size} px`;
  if (auto) auto.className = "hinges-auto node-auto" + (h.auto ? " on" : "");
}

export function renderHinges(s: State, dispatch: (e: Event) => void = () => {}, unlisten: (() => void)[] = []): HTMLElement | null {
  const v = hingeViewOf(s);
  if (!s.mesh || (v.count === 0 && v.refused.length === 0)) return null;
  const h = s.hinges;
  const box = el("div", "hinges");
  box.append(el("h3", "hinges-header", "Hinges"));
  const row = el("div", "legend-mode hinges-modes");
  const toggle = el("button", "hinges-toggle" + (h.on ? " on" : ""), "Show hinges");
  toggle.title = "Draw a marker at each moment hinge the model declares, at its position along the element";
  unlisten.push(listen(toggle, "click", () => dispatch({ type: "setHinges", on: !h.on })));
  row.append(toggle, select("hinges-color", "what colours the markers on a results step", COLOURS, h.colorBy, (colorBy) => dispatch({ type: "setHinges", colorBy }), unlisten));
  box.append(row, el("div", "hinges-count", `${v.count} hinge${v.count === 1 ? "" : "s"} on ${v.elements.length} element${v.elements.length === 1 ? "" : "s"}`));
  for (const r of v.refused) box.append(el("div", "hinges-note", `Not read: ${r}`));
  if (!h.on || v.count === 0) return box;
  const sizeRow = el("div", "hinges-size-row node-size-row");
  const slider = el("input", "hinges-size node-size");
  slider.type = "range";
  slider.min = String(NODE_SIZE_MIN);
  slider.max = String(NODE_SIZE_MAX);
  slider.step = "1";
  slider.title = "hinge marker size on screen";
  const auto = el("button", "hinges-auto node-auto", "auto");
  auto.title = "let the app choose the size for this model";
  unlisten.push(listen(slider, "input", () => dispatch({ type: "setHinges", size: Number(slider.value) })), listen(auto, "click", () => dispatch({ type: "setHinges", size: null })));
  const sizeLabel = el("span", "hinges-size-label node-size-label");
  sizeRow.append(slider, sizeLabel, auto);
  box.append(sizeRow);
  showSize(slider, sizeLabel, auto, h);
  if (v.scale) box.append(el("div", "hinges-scale", v.scale));
  const pick = el("div", "hinges-pick");
  pick.append(
    select("hinges-element", "the element to plot; a click on a hinged element in the viewport picks it too", v.elements.map((e) => [e.decl, e.label] as const), v.focus ?? "", (element) => dispatch({ type: "setHinges", element }), unlisten),
    select<HingeEnd>("hinges-end", "which end's hinge", v.ends.map((e) => [e, `end ${e}`] as const), v.end ?? "i", (end) => dispatch({ type: "setHinges", end }), unlisten),
    select<HingeAxis>("hinges-axis", "which bending axis", v.axes.map((a) => [a, a === "z" ? "Mz" : "My"] as const), v.axis ?? "z", (axis) => dispatch({ type: "setHinges", axis }), unlisten),
  );
  box.append(pick);
  if (v.siteText) box.append(el("div", "hinges-site", v.siteText));
  if (v.chart) box.append(charts(v.chart));
  for (const n of v.notes) box.append(el("div", "hinges-note", n));
  return box;
}

/**
 * What the card depends on, split in two: `card` rebuilds it (never the marker size,
 * so a dragged slider is not replaced), and `step` (the step
 * on screen, which playback moves, skipping steps on a slow model) only redraws the
 * plot, so an open menu stays open while the model plays.
 */
export function hingeCardKeys(s: State): { card: string; step: number | null } {
  const { history, size: _size, auto: _auto, ...rest } = s.hinges;
  const at = s.phase.at;
  return {
    card: JSON.stringify([rest, history ? [history.stage, history.element, history.time.length] : null, at?.kind ?? null, at?.kind === "results" ? at.stage : null, s.selection.decls, s.mesh?.linePositions.id]),
    step: at?.kind === "results" ? at.step : null,
  };
}

export function mountHinges(store: Store): () => void {
  const panel = byId("hinges");
  let unlisten: (() => void)[] = [];
  let lastCard = "";
  let lastStep: number | null = null;
  const render = (s: State) => {
    const { card, step } = hingeCardKeys(s);
    if (card === lastCard) {
      showSize(panel.querySelector<HTMLInputElement>(".hinges-size"), panel.querySelector<HTMLElement>(".hinges-size-label"), panel.querySelector<HTMLElement>(".hinges-auto"), s.hinges);
      if (step === lastStep) return;
      lastStep = step;
      // With the switch off nothing on the card follows the step.
      const chart = s.hinges.on ? hingeViewOf(s).chart : null;
      if (chart) panel.querySelector(".hinges-charts")?.replaceWith(charts(chart));
      return;
    }
    lastCard = card;
    lastStep = step;
    for (const u of unlisten) u();
    unlisten = [];
    const content = renderHinges(s, (e) => store.dispatch(e), unlisten);
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
