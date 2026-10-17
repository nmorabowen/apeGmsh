// Standalone 3-D Line Diagram panel (`#line-diagram`), positioned on the left
// above `#legend` so both `Contour` (in `#legend`) and `Line diagram` (in
// `#line-diagram`) can be controlled and viewed simultaneously without scrolling.
//
// Follows ADR 0112 D6 (`panel-allow-list`): imports only `store`, `selectors`,
// `types`, and `ui/dom`.

import {
  defaultLineDiagramComponent,
  effectiveFlipSign,
  fmtValue,
  LINE_DIAGRAM_CMAP_NAMES,
  LINE_DIAGRAM_COMPONENT_META,
  lineDiagramClimOf,
  lineDiagramFieldsOf,
  rampCss,
  RANGE_MODES,
  scaleLabel,
  signRampCss,
} from "../state/selectors.ts";
import type { State, Store } from "../state/store.ts";
import type { Event, LineDiagramCmap, LineDiagramPlane, LineStationComponent } from "../state/types.ts";
import { byId, el, listen } from "../ui/dom.ts";

const LINE_PLANES: readonly { plane: LineDiagramPlane; label: string }[] = [
  { plane: "local", label: "local axes" },
  { plane: "screen", label: "screen plane" },
];

const currentTheme = (): "dark" | "light" =>
  typeof document !== "undefined" && document.documentElement?.dataset?.["theme"] === "light" ? "light" : "dark";

export function mountLineDiagram(store: Store, subscribeTheme?: (fn: () => void) => () => void): () => void {
  const panel = byId("line-diagram");
  let unlisten: (() => void)[] = [];
  let lastSig = "";
  let lastTheme = "";

  const render = (s: State, prev: State | null) => {
    const theme = currentTheme();
    const sig = lineDiagramSignature(s);
    if (prev && theme === lastTheme && sig === lastSig) return;
    lastTheme = theme;
    lastSig = sig;
    for (const u of unlisten) u();
    unlisten = [];
    const content = renderLineDiagram(s, (e) => store.dispatch(e), unlisten);
    panel.hidden = content === null;
    panel.replaceChildren(...(content ? [...content.children] : []));
  };

  render(store.get(), null);
  const unsubscribe = store.subscribe(render);
  const unsubTheme = subscribeTheme?.(() => {
    if (currentTheme() !== lastTheme) render(store.get(), null);
  });
  return () => {
    unsubscribe();
    unsubTheme?.();
    for (const u of unlisten) u();
  };
}

/**
 * 3-D line diagram controls in `#line-diagram`, shown when the active results
 * stage has one or more line-station components (`N`, `Vy`, `Vz`, `T`, `My`,
 * `Mz`, `eps`, `kappaY`, `kappaZ`, `gammaY`, `gammaZ`, `theta`).
 */
export function renderLineDiagram(
  s: State,
  dispatch: (e: Event) => void = () => {},
  unlisten: (() => void)[] = [],
): HTMLElement | null {
  if (s.phase.at?.kind !== "results" || !s.results) return null;
  const st = s.results.stages[s.phase.at.stage];
  const comps = st?.lineStations ?? [];
  if (comps.length === 0) return null;

  const ld = s.lineDiagram;
  const box = el("div", "line-diagram");
  box.append(el("h3", "line-diagram-header", "Line diagram"));

  const modes = el("div", "legend-mode line-diagram-modes");
  const toggleBtn = el("button", "line-diagram-toggle" + (ld.on ? " on" : ""), "diagram");
  const fillBtn = el("button", "line-diagram-fill" + (ld.fill ? " on" : ""), "fill");
  fillBtn.disabled = !ld.on;
  const flipped = effectiveFlipSign(ld);
  const flipBtn = el("button", "line-diagram-flip" + (flipped ? " on" : ""), "flip sign");
  flipBtn.title = "Flip sign (draw bending moment on tension side)";
  unlisten.push(
    listen(toggleBtn, "click", () => dispatch({ type: "setLineDiagram", on: !ld.on })),
    listen(fillBtn, "click", () => dispatch({ type: "setLineDiagramFill", fill: !ld.fill })),
    listen(flipBtn, "click", () => dispatch({ type: "setLineDiagramFlipSign", flipSign: !flipped })),
  );
  modes.append(toggleBtn, fillBtn, flipBtn);
  box.append(modes);

  const planeRow = el("div", "legend-mode line-diagram-planes");
  for (const p of LINE_PLANES) {
    const b = el("button", `line-diagram-plane-${p.plane}` + (ld.plane === p.plane ? " on" : ""), p.label);
    unlisten.push(listen(b, "click", () => dispatch({ type: "setLineDiagramPlane", plane: p.plane })));
    planeRow.append(b);
  }
  box.append(planeRow);

  const fields = lineDiagramFieldsOf(comps);
  if (fields.length > 0) {
    const currentField = fields.find((f) => f.components.some((c) => c.key === ld.component)) ?? fields[0]!;
    const pickRow = el("div", "contour-pick line-diagram-pick");

    const fieldSelect = el("select", "contour-field line-diagram-field");
    fieldSelect.title = "line diagram field: section force or section deformation";
    for (const f of fields) {
      const opt = el("option", undefined, f.label);
      opt.value = f.name;
      opt.selected = f.name === currentField.name;
      fieldSelect.append(opt);
    }

    const compSelect = el("select", "contour-component line-diagram-component");
    compSelect.title = "line diagram component";
    for (const f of fields) {
      const grp = el("optgroup", undefined);
      grp.label = f.label;
      for (const c of f.components) {
        const opt = el("option", undefined, `${c.key} (${c.canonical})`);
        opt.value = c.key;
        opt.selected = ld.component === c.key;
        grp.append(opt);
      }
      compSelect.append(grp);
    }

    unlisten.push(
      listen(fieldSelect, "change", () => {
        const f = fields.find((x) => x.name === fieldSelect.value) ?? fields[0]!;
        const nextComp = defaultLineDiagramComponent(f.components.map((c) => c.key)) ?? f.components[0]!.key;
        dispatch({ type: "setLineDiagramComponent", component: nextComp });
      }),
      listen(compSelect, "change", () =>
        dispatch({ type: "setLineDiagramComponent", component: compSelect.value as LineStationComponent }),
      ),
    );
    pickRow.append(fieldSelect, compSelect);
    box.append(pickRow);
  }

  const scaleRow = el("div", "deform-scale-row line-diagram-scale-row");
  const scaleInput = el("input", "deform-scale line-diagram-scale");
  scaleInput.type = "number";
  scaleInput.step = "any";
  scaleInput.min = "0";
  scaleInput.value = String(ld.scale);
  unlisten.push(
    listen(scaleInput, "change", () => {
      const v = Number(scaleInput.value);
      if (Number.isFinite(v) && v > 0) dispatch({ type: "setLineDiagramScale", scale: v });
      else scaleInput.setCustomValidity("the scale must be a number above 0");
    }),
  );
  scaleRow.append(
    el("span", undefined, "scale"),
    scaleInput,
    el("span", "deform-scale-label line-diagram-scale-label" + (ld.scale !== 1 ? " scaled" : ""), scaleLabel(ld.scale)),
  );
  box.append(scaleRow);

  if (ld.on && ld.notice) {
    box.append(el("div", "legend-note refused-field line-diagram-notice", ld.notice));
  }

  if (ld.on) {
    const meta = ld.component ? LINE_DIAGRAM_COMPONENT_META[ld.component] : undefined;
    const titleText = meta ? `${meta.field} · ${meta.key} (${meta.canonical})` : "Line diagram";
    box.append(el("div", "contour-title line-diagram-title", titleText));

    const clim = lineDiagramClimOf(s);
    const ramp = el("div", "contour-ramp line-diagram-ramp");
    ramp.style.backgroundImage = ld.cmap === "sign" ? signRampCss(currentTheme()) : rampCss(ld.cmap);
    box.append(ramp);

    const constant = clim.min !== null && clim.max !== null && clim.min === clim.max;
    const minText = clim.min === null ? "no finite value" : fmtValue(clim.min);
    const maxText = clim.max === null ? "" : constant ? "(constant)" : fmtValue(clim.max);
    const ends = el("div", "contour-ends line-diagram-ends");
    ends.append(el("span", "contour-min", minText), el("span", "contour-max", maxText));
    box.append(ends);
    box.append(el("div", "contour-range line-diagram-range", clim.label));

    const rangeModes = el("div", "legend-mode contour-modes line-diagram-range-modes");
    for (const m of RANGE_MODES) {
      const b = el("button", ld.range === m.range ? "on" : "", m.label);
      unlisten.push(
        listen(b, "click", () => {
          if (m.range !== "fixed") return dispatch({ type: "setLineDiagramRange", range: m.range });
          const c = lineDiagramClimOf(s);
          const from = ld.fixed ?? (c.min !== null && c.max !== null && c.min < c.max ? { min: c.min, max: c.max } : null);
          if (from) dispatch({ type: "setLineDiagramRange", range: "fixed", fixed: from });
        }),
      );
      rangeModes.append(b);
    }
    box.append(rangeModes);

    if (ld.range === "fixed" && ld.fixed) {
      const fixedBox = el("div", "contour-fixed line-diagram-fixed");
      const lo = el("input", "contour-fixed-min line-diagram-fixed-min");
      const hi = el("input", "contour-fixed-max line-diagram-fixed-max");
      for (const [i, val, t] of [[lo, ld.fixed.min, "min"], [hi, ld.fixed.max, "max"]] as const) {
        i.type = "number";
        i.value = String(val);
        i.title = t;
        i.step = "any";
      }
      const apply = () => {
        const min = Number(lo.value),
          max = Number(hi.value);
        if (Number.isFinite(min) && Number.isFinite(max) && min < max) {
          dispatch({ type: "setLineDiagramRange", range: "fixed", fixed: { min, max } });
        } else {
          lo.setCustomValidity("min must be below max");
        }
      };
      unlisten.push(listen(lo, "change", apply), listen(hi, "change", apply));
      fixedBox.append(lo, hi);
      box.append(fixedBox);
    }

    const cm = el("select", "contour-cmap line-diagram-cmap");
    cm.title = "line diagram colormap";
    for (const name of LINE_DIAGRAM_CMAP_NAMES) {
      const o = el("option", `line-diagram-cmap-${name}` + (ld.cmap === name ? " on" : ""), name);
      o.value = name;
      o.selected = ld.cmap === name;
      unlisten.push(listen(o, "click", () => dispatch({ type: "setLineDiagramCmap", cmap: name })));
      cm.append(o);
    }
    unlisten.push(
      listen(cm, "change", () => dispatch({ type: "setLineDiagramCmap", cmap: cm.value as LineDiagramCmap })),
    );
    box.append(cm);
  }

  return box;
}

function lineDiagramSignature(s: State): string {
  const st = s.phase.at?.kind === "results" ? s.results?.stages[s.phase.at.stage] : undefined;
  return JSON.stringify([
    st?.lineStations ?? [],
    s.lineDiagram,
    s.lineDiagram.on ? lineDiagramClimOf(s) : null,
  ]);
}
