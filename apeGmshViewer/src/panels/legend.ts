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
  deformOf,
  deformView,
  GHOST_OPACITY,
  isHidden,
  legendOf,
  NO_DATA_RGB,
  RANGE_MODES,
  roleLegendOf,
  scaleLabel,
  type ContourView,
} from "../state/selectors.ts";
import type { State, Store } from "../state/store.ts";
import type { ColourBy, ContourCmap, ContourSource, GaussMode, LegendEntry } from "../state/types.ts";
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

/**
 * What the contour legend states, as plain lines (tested in Node; the DOM
 * below shows them). V4c2a: an element field also states its topology (the
 * Qt name and the averaging in words) and, when the component's name says so,
 * that its axes are local.
 */
export function contourLines(
  s: State,
  v: Extract<ContourView, { state: "ready" | "element" }>,
): { title: string; min: string; max: string; range: string; state: string | null; topology: string | null; axes: string | null } {
  const c = v.clim;
  const constant = c.min !== null && c.max !== null && c.min === c.max;
  return {
    title: `${v.field} · ${v.component.label}`,
    min: c.min === null ? "no finite value" : fmtValue(c.min),
    max: c.max === null ? "" : constant ? "(constant)" : fmtValue(c.max),
    range: c.label,
    state: !v.stepMeasured ? `step ${s.phase.at?.kind === "results" ? s.phase.at.step + 1 : "?"} loading` : null,
    topology: v.state === "element" ? v.topologyLabel : null,
    axes: v.state === "element" && v.component.local ? "Axes: local (the component's name says so)." : null,
  };
}

/** The element-field draw modes: flat per cell (V4c2a), and carried to the nodes (V4c2b). */
const GAUSS_MODES: readonly { mode: GaussMode; label: string; title: string }[] = [
  { mode: "cell", label: "per cell", title: "flat per cell: gauss_cell (one Gauss point), or the element mean of its Gauss points (an app presentation, no Qt equivalent)" },
  {
    mode: "cell_averaged",
    label: "cell, averaged at nodes",
    title: "gauss_cell_averaged: one-point cell values averaged to the nodes across elements (with several Gauss points, each element's mean: an app presentation, no Qt equivalent)",
  },
  { mode: "node", label: "at nodes", title: "gauss_node: the Gauss points extrapolated to the element corners (least squares) and averaged at the nodes across elements" },
  { mode: "node_discrete", label: "at nodes, per element", title: "gauss_node_discrete: the Gauss points extrapolated to each element's corners, no averaging (jumps between elements show)" },
];

export function mountLegend(store: Store): () => void {
  const legend = byId("legend");
  let unlisten: (() => void)[] = [];
  let lastContour = "";
  let lastDeform = "";
  const render = (s: State, prev: State | null) => {
    const contour = s.visibility.colourBy === "contour";
    if (prev && s.mesh === prev.mesh && s.visibility.hidden === prev.visibility.hidden && s.visibility.colourBy === prev.visibility.colourBy) {
      // In contour mode the legend follows the pick, the step and the ranges, and (V4d) the deformed shape's
      // state in every mode; it is rebuilt only when what it says changed.
      if (deformSignature(s) === lastDeform && (!contour || contourSignature(s) === lastContour)) return;
    }
    lastDeform = deformSignature(s);
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
      renderDeform(s);
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
    renderDeform(s);
  };

  /**
   * V4d: the deformed shape's controls and what it shows: the toggle (off
   * with the reason when there is nothing to deform by), the scale (stated
   * in words whenever it is not 1), the reference ghost, and the drawn nodes
   * left undeformed. Shown once results are on the phase axis.
   */
  const renderDeform = (s: State) => {
    if (!s.phase.axis.some((k) => k.kind === "results")) return;
    const lines = deformLines(s);
    const box = el("div", "deform");
    box.append(el("div", "deform-title", "Deformed shape"));
    const modes = el("div", "legend-mode deform-modes");
    const on = el("button", "deform-toggle" + (s.deform.on ? " on" : ""), "deformed");
    on.disabled = !lines.available && !s.deform.on;
    if (lines.why) on.title = lines.why;
    const ghost = el("button", "deform-ghost" + (s.deform.ghost ? " on" : ""), "reference ghost");
    ghost.title = `the undeformed mesh behind the deformed one, at ${GHOST_OPACITY} opacity`;
    ghost.disabled = !s.deform.on;
    unlisten.push(
      listen(on, "click", () => store.dispatch({ type: "setDeform", on: !store.get().deform.on })),
      listen(ghost, "click", () => store.dispatch({ type: "setReferenceGhost", on: !store.get().deform.ghost })),
    );
    modes.append(on, ghost);
    box.append(modes);
    const row = el("div", "deform-scale-row");
    const scale = el("input", "deform-scale");
    scale.type = "number";
    scale.step = "any";
    scale.min = "0";
    scale.value = String(s.deform.scale);
    scale.title = "deformation scale: positions are x + scale · u (1 is true scale)";
    unlisten.push(
      listen(scale, "change", () => {
        const v = Number(scale.value);
        if (Number.isFinite(v) && v > 0) store.dispatch({ type: "setDeformScale", scale: v });
        else scale.setCustomValidity("the scale must be a number above 0");
      }),
    );
    row.append(el("span", undefined, "scale"), scale, el("span", "deform-scale-label" + (s.deform.scale !== 1 ? " scaled" : ""), lines.scale));
    box.append(row);
    for (const n of lines.notes) box.append(el("div", "legend-note" + (n.refused ? " refused-field" : ""), n.text));
    if (lines.partial) {
      const m = el("div", "legend-row deform-missing");
      const chip = el("span", "deform-marker");
      chip.style.backgroundColor = css(NO_DATA_RGB);
      m.append(chip, el("span", undefined, lines.partial));
      box.append(m);
    }
    legend.append(box);
  };

  const renderContour = (s: State) => {
    const v = contourView(s);
    const picker = contourFieldsOf(s);
    // V4c2a: one list of nodal fields, then element (Gauss) fields; an option's value names its source.
    const options: { source: ContourSource; name: string; components: { key: string; label: string }[] }[] = [
      ...(picker?.fields ?? []).map((f) => ({ source: "nodal" as const, name: f.name, components: f.components })),
      ...(picker?.elements ?? []).map((f) => ({ source: "element" as const, name: f.name, components: f.components })),
    ];
    const valueOf = (source: ContourSource, name: string) => `${source}:${name}`;
    if (options.length) {
      const row = el("div", "contour-pick");
      const field = el("select", "contour-field");
      field.title = "field: nodal, or element (Gauss points)";
      if (s.contour.field === null) {
        // Nothing picked (a re-read cleared the pick): a placeholder, so picking the first field is a change.
        const o = el("option", undefined, "pick a field");
        o.disabled = true;
        o.selected = true;
        field.append(o);
      }
      for (const source of ["nodal", "element"] as const) {
        const of = options.filter((x) => x.source === source);
        if (!of.length) continue;
        const group = el("optgroup");
        group.label = source === "nodal" ? "nodal" : picker?.elementsVerified ? "element (Gauss points)" : "element (Gauss points, unverified file)";
        for (const f of of) {
          const o = el("option", undefined, f.name);
          o.value = valueOf(f.source, f.name);
          o.selected = f.source === s.contour.source && f.name === s.contour.field;
          group.append(o);
        }
        field.append(group);
      }
      const current = options.find((f) => f.source === s.contour.source && f.name === s.contour.field) ?? options[0]!;
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
          const f = options.find((x) => valueOf(x.source, x.name) === field.value)!;
          store.dispatch({ type: "setContourField", field: f.name, component: f.components[0]!.key, source: f.source });
        }),
        listen(comp, "change", () => store.dispatch({ type: "setContourField", field: current.name, component: comp.value, source: current.source })),
      );
      row.append(field, comp);
      legend.append(row);
    }
    for (const r of picker?.refused ?? []) legend.append(el("div", "legend-note refused-field", `${r.component}: not read (${r.reason})`));
    for (const f of picker?.fields ?? []) {
      if (f.magnitudeRefused) legend.append(el("div", "legend-note refused-field", `${f.name} magnitude: refused, ${f.magnitudeRefused}`));
    }
    for (const f of picker?.elements ?? []) {
      if (f.refused) legend.append(el("div", "legend-note refused-field", `element field ${f.name}: refused, ${f.refused}`));
    }
    if (s.contour.source === "element") {
      const modes = el("div", "legend-mode contour-gauss");
      for (const m of GAUSS_MODES) {
        const b = el("button", s.contour.gauss === m.mode ? "on" : "", m.label);
        b.title = m.title;
        unlisten.push(listen(b, "click", () => store.dispatch({ type: "setContourGauss", mode: m.mode })));
        modes.append(b);
      }
      legend.append(modes);
    }
    if (v.state !== "ready" && v.state !== "element") {
      const note =
        v.state === "unavailable"
          ? `Nothing to colour: ${v.reason}.`
          : v.state === "refused"
            ? `Refused: ${v.reason}.`
            : v.state === "not-yet"
              ? `Not drawn: ${v.reason}.`
              : "";
      legend.append(el("div", "legend-note" + (v.state === "refused" ? " refused-field" : ""), note));
      legend.append(noDataRow());
      return;
    }
    const lines = contourLines(s, v);
    legend.append(el("div", "contour-title", lines.title));
    if (lines.topology) legend.append(el("div", "legend-note contour-topology", lines.topology));
    if (lines.axes) legend.append(el("div", "legend-note contour-axes", lines.axes));
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
          const from = store.get().contour.fixed ?? ((c.state === "ready" || c.state === "element") && c.clim.min !== null && c.clim.max !== null && c.clim.min < c.clim.max ? { min: c.clim.min, max: c.clim.max } : null);
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
    "every face or line that touches one is drawn in this grey over its whole area, including its part next to recorded nodes. " +
    "For an element field: a drawn element no group of the field records, or one with a non-finite Gauss value, is drawn in this grey";
  row.append(chip, text);
  return row;
}

/**
 * What the deformed-shape section states (V4d), as plain data (tested in
 * Node; the DOM above shows it): whether the toggle can turn on and why not,
 * the scale in words, the notes, and the drawn nodes left undeformed.
 */
export function deformLines(s: State): {
  available: boolean;
  why: string | null;
  scale: string;
  notes: { text: string; refused: boolean }[];
  partial: string | null;
} {
  const a = deformOf(s);
  const v = deformView(s);
  const notes: { text: string; refused: boolean }[] = [];
  if (!a.ok) notes.push({ text: `${s.deform.on ? "Not drawn" : "Unavailable"}: ${a.reason}.`, refused: a.state === "refused" });
  else if (a.draw.note) notes.push({ text: a.draw.note, refused: false });
  if (v.state === "loading") notes.push({ text: `step ${s.phase.at?.kind === "results" ? s.phase.at.step + 1 : "?"} loading`, refused: false });
  return {
    available: a.ok,
    why: a.ok ? null : a.reason,
    scale: scaleLabel(s.deform.scale),
    notes,
    partial:
      v.state === "partial"
        ? `${v.missing.toLocaleString()} of ${v.drawn.toLocaleString()} drawn nodes have no displacement: drawn undeformed, marked`
        : null,
  };
}

/** What the deformed-shape section shows, as one string (the step's keys left out, so a scrub does not rebuild it). */
function deformSignature(s: State): string {
  return JSON.stringify([s.phase.axis.some((k) => k.kind === "results"), s.deform.on, s.deform.scale, s.deform.ghost, deformLines(s)]);
}

/** What the contour legend shows, as one string: it is rebuilt only when this changes. */
function contourSignature(s: State): string {
  const v = contourView(s);
  const picker = contourFieldsOf(s);
  return JSON.stringify([
    v.state === "ready" || v.state === "element" ? contourLines(s, v) : v,
    v.state === "ready" || v.state === "element" ? v.cmap : null,
    picker,
    s.contour.source,
    s.contour.gauss,
    s.contour.field,
    s.contour.component,
    s.contour.range,
    s.contour.fixed,
  ]);
}
