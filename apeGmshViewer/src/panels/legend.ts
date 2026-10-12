// Legend panel: the colour mode (by physical group, by the structural role
// the file records, or, V4c1, a nodal contour) and its rows. A click on a
// group's row hides or shows it (`setHidden`); "show all" clears the hidden
// set. In contour mode it is also the field picker: the field, its
// component, the colour range mode and the colormap, and the ramp with its
// min and max, the range mode in words, and the no-data colour. It
// dispatches events and reads selectors, nothing else.

import {
  CMAP_NAMES,
  CMAPS,
  contourFieldsOf,
  contourView,
  isHidden,
  legendOf,
  NO_DATA_RGB,
  RANGE_MODES,
  roleLegendOf,
  type ContourView,
} from "../state/selectors.ts";
import type { State, Store } from "../state/store.ts";
import type { ColourBy, ContourCmap, LegendEntry } from "../state/types.ts";
import { byId, el, listen } from "../ui/dom.ts";

const MODES: readonly { by: ColourBy; label: string }[] = [
  { by: "group", label: "by group" },
  { by: "role", label: "by role" },
  { by: "contour", label: "contour" },
];

const css = (rgb: readonly number[]): string => `rgb(${rgb.map((c) => Math.round(c * 255)).join(",")})`;

/** A CSS gradient through a colormap's stops (the same table the mesh colours from). */
export const rampCss = (cmap: ContourCmap): string => `linear-gradient(to right, ${CMAPS[cmap].map(([t, r, g, b]) => `${css([r, g, b])} ${(100 * t).toFixed(1)}%`).join(", ")})`;

/** A legend number: four significant digits, exponent form outside 1e-3..1e4. */
export function fmtValue(v: number): string {
  if (v === 0) return "0";
  const a = Math.abs(v);
  return a >= 1e-3 && a < 1e4 ? String(Number(v.toPrecision(4))) : v.toExponential(3);
}

/** What the contour legend states, as plain lines (tested in Node; the DOM below shows them). */
export function contourLines(s: State, v: Extract<ContourView, { state: "ready" }>): { title: string; min: string; max: string; range: string; state: string | null } {
  const c = v.clim;
  const constant = c.min !== null && c.max !== null && c.min === c.max;
  return {
    title: `${v.field} · ${v.component.label}`,
    min: c.min === null ? "no finite value" : fmtValue(c.min),
    max: c.max === null ? "" : constant ? "(constant)" : fmtValue(c.max),
    range: c.label,
    state: !v.stepMeasured ? `step ${s.phase.at?.kind === "results" ? s.phase.at.step + 1 : "?"} loading` : null,
  };
}

export function mountLegend(store: Store): () => void {
  const legend = byId("legend");
  let unlisten: (() => void)[] = [];
  let lastContour = "";
  const render = (s: State, prev: State | null) => {
    const contour = s.visibility.colourBy === "contour";
    if (prev && s.mesh === prev.mesh && s.visibility.hidden === prev.visibility.hidden && s.visibility.colourBy === prev.visibility.colourBy) {
      if (!contour) return;
      // In contour mode the legend follows the pick, the step and the ranges; it is rebuilt only when what it says changed.
      if (contourSignature(s) === lastContour) return;
    }
    for (const u of unlisten) u();
    unlisten = [];
    legend.hidden = s.mesh === null;
    const byRole = s.visibility.colourBy === "role";
    const head = el("h3", undefined, contour ? "Contour" : byRole ? "Structural roles" : "Physical groups");
    legend.replaceChildren(head);
    if (!byRole && !contour && s.visibility.hidden.length) {
      const all = el("button", "legend-all", "show all");
      unlisten.push(listen(all, "click", () => store.dispatch({ type: "showAll" })));
      head.append(all);
    }
    const modes = el("div", "legend-mode");
    for (const m of MODES) {
      const b = el("button", s.visibility.colourBy === m.by ? "on" : "", m.label);
      unlisten.push(listen(b, "click", () => store.dispatch({ type: "setColourBy", by: m.by })));
      modes.append(b);
    }
    legend.append(modes);
    if (contour) {
      lastContour = contourSignature(s);
      renderContour(s);
      return;
    }
    let rows: LegendEntry[];
    if (byRole) {
      const r = roleLegendOf(s);
      rows = r?.legend ?? [];
      if (r && !r.fileHasRoles) {
        legend.append(el("div", "legend-note", "The file records no structural role (no role attribute on its physical groups), so nothing is coloured by role; every element is drawn as unassigned."));
      }
    } else rows = legendOf(s);
    for (const e of rows) {
      const hidden = !byRole && isHidden(s, e.decl);
      const row = el("div", "legend-row" + (hidden ? " hidden-group" : "") + (!byRole && e.decl ? " clickable" : ""));
      const chip = el("span", "chip" + (e.cue ? ` ${e.cue}` : ""));
      chip.style.backgroundColor = `rgb(${e.color.map((c) => Math.round(c * 255)).join(",")})`;
      row.append(chip, el("span", undefined, e.name), el("span", "legend-count", e.elements.toLocaleString()));
      if (!byRole && e.decl) {
        const decl = e.decl;
        row.title = "click to hide or show";
        unlisten.push(listen(row, "click", () => store.dispatch({ type: "setHidden", decls: [decl], hidden: !isHidden(store.get(), decl) })));
      }
      legend.append(row);
    }
  };

  const renderContour = (s: State) => {
    const v = contourView(s);
    const picker = contourFieldsOf(s);
    if (picker && picker.fields.length) {
      const row = el("div", "contour-pick");
      const field = el("select", "contour-field");
      field.title = "nodal field";
      if (s.contour.field === null) {
        // Nothing picked (a re-read cleared the pick): a placeholder, so picking the first field is a change.
        const o = el("option", undefined, "pick a field");
        o.disabled = true;
        o.selected = true;
        field.append(o);
      }
      for (const f of picker.fields) {
        const o = el("option", undefined, f.name);
        o.value = f.name;
        o.selected = f.name === s.contour.field;
        field.append(o);
      }
      const current = picker.fields.find((f) => f.name === s.contour.field) ?? picker.fields[0]!;
      const comp = el("select", "contour-component");
      comp.title = "component";
      for (const c of current.components) {
        const o = el("option", undefined, c.label);
        o.value = c.key;
        o.selected = c.key === s.contour.component;
        comp.append(o);
      }
      unlisten.push(
        listen(field, "change", () => {
          const f = picker.fields.find((x) => x.name === field.value)!;
          store.dispatch({ type: "setContourField", field: f.name, component: f.components[0]!.key });
        }),
        listen(comp, "change", () => store.dispatch({ type: "setContourField", field: current.name, component: comp.value })),
      );
      row.append(field, comp);
      legend.append(row);
    }
    for (const r of picker?.refused ?? []) legend.append(el("div", "legend-note refused-field", `${r.component}: not read (${r.reason})`));
    for (const f of picker?.fields ?? []) {
      if (f.magnitudeRefused) legend.append(el("div", "legend-note refused-field", `${f.name} magnitude: refused, ${f.magnitudeRefused}`));
    }
    if (v.state !== "ready") {
      legend.append(el("div", "legend-note", v.state === "unavailable" ? `Nothing to colour: ${v.reason}.` : ""));
      legend.append(noDataRow());
      return;
    }
    const lines = contourLines(s, v);
    legend.append(el("div", "contour-title", lines.title));
    const ramp = el("div", "contour-ramp");
    ramp.style.backgroundImage = rampCss(v.cmap);
    legend.append(ramp);
    const ends = el("div", "contour-ends");
    ends.append(el("span", "contour-min", lines.min), el("span", "contour-max", lines.max));
    legend.append(ends);
    legend.append(el("div", "contour-range", lines.range));
    if (lines.state) legend.append(el("div", "legend-note", lines.state));
    legend.append(el("div", "legend-note", "Units: not stored in the results file."));
    legend.append(noDataRow());

    const modes = el("div", "legend-mode contour-modes");
    for (const m of RANGE_MODES) {
      const b = el("button", s.contour.range === m.range ? "on" : "", m.label);
      unlisten.push(
        listen(b, "click", () => {
          if (m.range !== "fixed") return store.dispatch({ type: "setContourRange", range: m.range });
          // A fixed range starts from the range shown, when it has one.
          const c = contourView(store.get());
          const from = store.get().contour.fixed ?? (c.state === "ready" && c.clim.min !== null && c.clim.max !== null && c.clim.min < c.clim.max ? { min: c.clim.min, max: c.clim.max } : null);
          if (from) store.dispatch({ type: "setContourRange", range: "fixed", fixed: from });
        }),
      );
      modes.append(b);
    }
    legend.append(modes);
    if (s.contour.range === "fixed" && s.contour.fixed) {
      const box = el("div", "contour-fixed");
      const lo = el("input", "contour-fixed-min");
      const hi = el("input", "contour-fixed-max");
      for (const [i, val, t] of [[lo, s.contour.fixed.min, "min"], [hi, s.contour.fixed.max, "max"]] as const) {
        i.type = "number";
        i.value = String(val);
        i.title = t;
        i.step = "any";
      }
      const apply = () => {
        const min = Number(lo.value), max = Number(hi.value);
        if (Number.isFinite(min) && Number.isFinite(max) && min < max) store.dispatch({ type: "setContourRange", range: "fixed", fixed: { min, max } });
        else lo.setCustomValidity("min must be below max");
      };
      unlisten.push(listen(lo, "change", apply), listen(hi, "change", apply));
      box.append(lo, hi);
      legend.append(box);
    }
    const cm = el("select", "contour-cmap");
    cm.title = "colormap";
    for (const name of CMAP_NAMES) {
      const o = el("option", undefined, name);
      o.value = name;
      o.selected = name === v.cmap;
      cm.append(o);
    }
    unlisten.push(listen(cm, "change", () => store.dispatch({ type: "setContourCmap", cmap: cm.value as ContourCmap })));
    legend.append(cm);
  };

  render(store.get(), null);
  const unsubscribe = store.subscribe(render);
  return () => {
    unsubscribe();
    for (const u of unlisten) u();
  };
}

function noDataRow(): HTMLElement {
  const row = el("div", "legend-row");
  const chip = el("span", "chip");
  chip.style.backgroundColor = css(NO_DATA_RGB);
  const text = el("span", "contour-nodata", "no data");
  text.title =
    "a node the field does not record (not in its _ids, or NaN) is never zero and never taken from a neighbour: " +
    "every face or line that touches one is drawn in this grey over its whole area, including its part next to recorded nodes";
  row.append(chip, text);
  return row;
}

/** What the contour legend shows, as one string: it is rebuilt only when this changes. */
function contourSignature(s: State): string {
  const v = contourView(s);
  const picker = contourFieldsOf(s);
  return JSON.stringify([
    v.state === "ready" ? contourLines(s, v) : v,
    v.state === "ready" ? v.cmap : null,
    picker,
    s.contour.field,
    s.contour.component,
    s.contour.range,
    s.contour.fixed,
  ]);
}
