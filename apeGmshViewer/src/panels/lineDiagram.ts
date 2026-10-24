// Standalone 3-D Line Diagram panel (`#line-diagram`), positioned on the left
// above `#legend` so both `Contour` (in `#legend`) and `Line diagram` (in
// `#line-diagram`) can be controlled and viewed simultaneously without scrolling.
//
// Follows ADR 0112 D6 (`panel-allow-list`): imports only `store`, `selectors`,
// `types`, and `ui/dom`.

import {
  buildBeamChainPlotModel,
  defaultLineDiagramComponent,
  effectiveFlipSign,
  fmtValue,
  formatHoverReadout,
  LINE_DIAGRAM_CMAP_NAMES,
  LINE_DIAGRAM_COMPONENT_META,
  lineDiagramClimOf,
  lineDiagramFieldsOf,
  pickChainPlotAtS,
  rampCss,
  RANGE_MODES,
  scaleLabel,
  signRampCss,
} from "../state/selectors.ts";
import type { State, Store } from "../state/store.ts";
import type { BeamChain, Event, LineDiagramCmap, LineDiagramPlane, LineStationComponent } from "../state/types.ts";
import { byId, el, listen } from "../ui/dom.ts";

const LINE_PLANES: readonly { plane: LineDiagramPlane; label: string }[] = [
  { plane: "local", label: "local axes" },
  { plane: "screen", label: "screen plane" },
];

const SVG_NS = "http://www.w3.org/2000/svg";

function svgEl(tag: string, cls?: string, attrs?: Record<string, string>, text?: string): HTMLElement {
  const doc = document as Document & { createElementNS?: (ns: string, q: string) => Element };
  const node = (typeof doc.createElementNS === "function"
    ? doc.createElementNS(SVG_NS, tag)
    : document.createElement(tag)) as unknown as HTMLElement & {
    setAttribute?: (k: string, v: string) => void;
    getAttribute?: (k: string) => string | null;
    dataset?: Record<string, string | undefined>;
  };
  const attrMap: Record<string, string> = {};
  if (cls) {
    attrMap["class"] = cls;
    if (typeof node.setAttribute === "function") node.setAttribute("class", cls);
    try {
      (node as unknown as { className: string }).className = cls;
    } catch {
      // SVGElement.className in browsers is a read-only SVGAnimatedString; setAttribute above already set it.
    }
  }
  if (attrs) {
    for (const [k, v] of Object.entries(attrs)) {
      attrMap[k] = v;
      if (typeof node.setAttribute === "function") node.setAttribute(k, v);
      if (node.dataset) {
        if (k.startsWith("data-")) {
          const camel = k.slice(5).replace(/-([a-z])/g, (_, c: string) => c.toUpperCase());
          node.dataset[camel] = v;
        } else {
          node.dataset[k] = v;
        }
      }
    }
  }
  if (typeof node.getAttribute !== "function") {
    node.getAttribute = (k: string) => attrMap[k] ?? null;
  }
  if (text !== undefined) node.textContent = text;
  return node;
}

const currentTheme = (): "dark" | "light" =>
  typeof document !== "undefined" && document.documentElement?.dataset?.["theme"] === "light" ? "light" : "dark";

interface ChainHoverSlot {
  layer: HTMLElement;
  chain: BeamChain;
  chainIndex: number;
}

function syncHoverReadoutSlot(slot: HTMLElement, s: State): void {
  const ld = s.lineDiagram;
  if (!ld.on || ld.hover === null) {
    slot.hidden = true;
    slot.replaceChildren();
    return;
  }
  const readout = el(
    "div",
    "line-diagram-hover-readout",
    formatHoverReadout(ld.hover, ld.component),
  );
  readout.dataset["elementId"] = String(ld.hover.elementId);
  readout.dataset["xi"] = ld.hover.xi.toFixed(4);
  readout.dataset["s"] = ld.hover.s.toFixed(4);
  readout.dataset["value"] = String(ld.hover.value);
  readout.dataset["withinStationSpan"] = String(ld.hover.withinStationSpan);
  slot.hidden = false;
  slot.replaceChildren(readout);
}

function syncChainHoverLayer(
  layer: HTMLElement,
  chain: BeamChain,
  chainIndex: number,
  s: State,
): void {
  const ld = s.lineDiagram;
  const model = buildBeamChainPlotModel(chain, chainIndex, ld.chains.length, {
    flipSign: effectiveFlipSign(ld),
    fill: ld.fill,
    component: ld.component,
    hover: ld.hover,
  });
  if (model.hoverMarker === null) {
    layer.replaceChildren();
    return;
  }
  layer.replaceChildren(
    svgEl("line", "line-diagram-chain-hover-needle", {
      x1: String(model.hoverMarker.x),
      y1: String(model.hoverMarker.zeroY),
      x2: String(model.hoverMarker.x),
      y2: String(model.hoverMarker.y),
      "data-element-id": String(model.hoverMarker.elementId),
      "data-xi": String(model.hoverMarker.xi),
      "data-s": String(model.hoverMarker.s),
      "data-value": String(model.hoverMarker.value),
    }),
    svgEl("circle", "line-diagram-chain-hover-dot", {
      cx: String(model.hoverMarker.x),
      cy: String(model.hoverMarker.y),
      r: "3.5",
      "data-element-id": String(model.hoverMarker.elementId),
    }),
  );
}

export function mountLineDiagram(store: Store, subscribeTheme?: (fn: () => void) => () => void): () => void {
  const panel = byId("line-diagram");
  let unlisten: (() => void)[] = [];
  let lastSig = "";
  let lastTheme = "";
  let lastDrawnBeams: State["lineDiagram"]["drawnBeams"] | null = null;
  let lastHover: State["lineDiagram"]["hover"] = null;
  let hoverSlot: HTMLElement | null = null;
  let chainHoverSlots: ChainHoverSlot[] = [];

  const render = (s: State, prev: State | null) => {
    const theme = currentTheme();
    const sig = lineDiagramSignature(s);
    if (
      prev &&
      theme === lastTheme &&
      sig === lastSig &&
      s.lineDiagram.drawnBeams === lastDrawnBeams
    ) {
      if (s.lineDiagram.hover !== lastHover) {
        lastHover = s.lineDiagram.hover;
        if (hoverSlot) syncHoverReadoutSlot(hoverSlot, s);
        for (const item of chainHoverSlots) {
          syncChainHoverLayer(item.layer, item.chain, item.chainIndex, s);
        }
      }
      return;
    }
    lastTheme = theme;
    lastSig = sig;
    lastDrawnBeams = s.lineDiagram.drawnBeams;
    lastHover = s.lineDiagram.hover;
    for (const u of unlisten) u();
    unlisten = [];
    hoverSlot = null;
    chainHoverSlots = [];
    const content = renderLineDiagram(
      s,
      (e) => store.dispatch(e),
      unlisten,
      (slot) => {
        hoverSlot = slot;
      },
      chainHoverSlots,
    );
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

function renderChainCard(
  chain: BeamChain,
  chainIndex: number,
  s: State,
  flipped: boolean,
  dispatch: (e: Event) => void,
  unlisten: (() => void)[],
  chainHoverSlots?: ChainHoverSlot[],
): HTMLElement {
  const ld = s.lineDiagram;
  const model = buildBeamChainPlotModel(chain, chainIndex, ld.chains.length, {
    flipSign: flipped,
    fill: ld.fill,
    component: ld.component,
    hover: ld.hover,
  });

  const card = el("div", "line-diagram-chain-card");
  card.dataset["chainIndex"] = String(chainIndex);
  card.dataset["spanCount"] = String(chain.spans.length);
  card.dataset["totalLength"] = String(chain.totalLength);

  const header = el("div", "line-diagram-chain-header");
  header.append(el("span", "line-diagram-chain-title", model.headerLabel));
  if (model.unknownEndBadge !== null) {
    header.append(el("span", "line-diagram-chain-unknown-ends", model.unknownEndBadge));
  }
  card.append(header);

  const svg = svgEl("svg", "line-diagram-chain-svg", {
    viewBox: `0 0 ${model.width} ${model.height}`,
    width: String(model.width),
    height: String(model.height),
    "data-chain-index": String(chainIndex),
    "data-flip-sign": String(model.flipSign),
  });

  svg.append(
    svgEl("line", "line-diagram-chain-axis", {
      x1: String(model.padLeft),
      y1: String(model.zeroY),
      x2: String(model.width - model.padRight),
      y2: String(model.zeroY),
    }),
  );

  for (const j of model.joints) {
    svg.append(
      svgEl("line", "line-diagram-chain-joint" + (j.isInternal ? " internal" : " boundary"), {
        x1: String(j.x),
        y1: String(model.padTop),
        x2: String(j.x),
        y2: String(model.height - model.padBottom),
        "data-node-id": String(j.nodeId),
        "data-s": String(j.s),
      }),
      svgEl(
        "text",
        "line-diagram-chain-joint-label",
        {
          x: String(j.x),
          y: String(model.height - 4),
          "text-anchor": "middle",
          "data-node-id": String(j.nodeId),
        },
        j.label,
      ),
    );
  }

  for (const sp of model.spans) {
    svg.append(
      svgEl(
        "text",
        "line-diagram-chain-span-label",
        {
          x: String(sp.midX),
          y: String(model.padTop - 4),
          "text-anchor": "middle",
          "data-element-id": String(sp.elementId),
        },
        sp.label,
      ),
    );

    if (sp.polygonPoints !== null) {
      svg.append(
        svgEl("polygon", "line-diagram-chain-fill", {
          points: sp.polygonPoints,
          "data-element-id": String(sp.elementId),
        }),
      );
    }

    if (sp.polylinePoints !== "") {
      svg.append(
        svgEl("polyline", "line-diagram-chain-curve", {
          points: sp.polylinePoints,
          fill: "none",
          "data-element-id": String(sp.elementId),
          "data-start-end-known": String(sp.startEndKnown),
          "data-end-end-known": String(sp.endEndKnown),
          "data-xi-min": String(sp.xiMin),
          "data-xi-max": String(sp.xiMax),
        }),
      );
    }

    if (sp.unknownStartGap !== null) {
      svg.append(
        svgEl("line", "line-diagram-chain-unknown-end line-diagram-chain-unknown-start", {
          x1: String(sp.unknownStartGap.x0),
          y1: String(sp.unknownStartGap.y0),
          x2: String(sp.unknownStartGap.x1),
          y2: String(sp.unknownStartGap.y1),
          "stroke-dasharray": "3,3",
          "data-element-id": String(sp.elementId),
        }),
        svgEl("circle", "line-diagram-chain-unknown-end-dot", {
          cx: String(sp.unknownStartGap.stationX),
          cy: String(sp.unknownStartGap.stationY),
          r: "2.5",
          fill: "none",
          "data-element-id": String(sp.elementId),
        }),
      );
    }

    if (sp.unknownEndGap !== null) {
      svg.append(
        svgEl("line", "line-diagram-chain-unknown-end line-diagram-chain-unknown-finish", {
          x1: String(sp.unknownEndGap.x0),
          y1: String(sp.unknownEndGap.y0),
          x2: String(sp.unknownEndGap.x1),
          y2: String(sp.unknownEndGap.y1),
          "stroke-dasharray": "3,3",
          "data-element-id": String(sp.elementId),
        }),
        svgEl("circle", "line-diagram-chain-unknown-end-dot", {
          cx: String(sp.unknownEndGap.stationX),
          cy: String(sp.unknownEndGap.stationY),
          r: "2.5",
          fill: "none",
          "data-element-id": String(sp.elementId),
        }),
      );
    }

    if (sp.minMarker !== null) {
      svg.append(
        svgEl("circle", "line-diagram-chain-extrema line-diagram-chain-min", {
          cx: String(sp.minMarker.x),
          cy: String(sp.minMarker.y),
          r: "2.5",
          "data-element-id": String(sp.elementId),
          "data-s": String(sp.minMarker.s),
          "data-value": String(sp.minMarker.value),
        }),
        svgEl(
          "text",
          "line-diagram-chain-extrema-label line-diagram-chain-min-label",
          {
            x: String(sp.minMarker.x),
            y: String(sp.minMarker.y + (sp.minMarker.y >= model.zeroY ? 10 : -4)),
            "text-anchor": "middle",
            "data-element-id": String(sp.elementId),
          },
          `min ${sp.minMarker.label}`,
        ),
      );
    }

    if (sp.maxMarker !== null) {
      svg.append(
        svgEl("circle", "line-diagram-chain-extrema line-diagram-chain-max", {
          cx: String(sp.maxMarker.x),
          cy: String(sp.maxMarker.y),
          r: "2.5",
          "data-element-id": String(sp.elementId),
          "data-s": String(sp.maxMarker.s),
          "data-value": String(sp.maxMarker.value),
        }),
        svgEl(
          "text",
          "line-diagram-chain-extrema-label line-diagram-chain-max-label",
          {
            x: String(sp.maxMarker.x),
            y: String(sp.maxMarker.y + (sp.maxMarker.y <= model.zeroY ? -4 : 10)),
            "text-anchor": "middle",
            "data-element-id": String(sp.elementId),
          },
          `max ${sp.maxMarker.label}`,
        ),
      );
    }
  }

  const hoverLayer = svgEl("g", "line-diagram-chain-hover-layer");
  syncChainHoverLayer(hoverLayer, chain, chainIndex, s);
  svg.append(hoverLayer);
  chainHoverSlots?.push({ layer: hoverLayer, chain, chainIndex });

  unlisten.push(
    listen(svg, "pointermove", (ev: PointerEvent) => {
      const plotW = Math.max(1, model.width - model.padLeft - model.padRight);
      let localX = typeof ev?.offsetX === "number" ? ev.offsetX : model.padLeft + plotW / 2;
      if (typeof ev?.offsetX !== "number" && typeof ev?.clientX === "number" && typeof svg.getBoundingClientRect === "function") {
        const rect = svg.getBoundingClientRect();
        if (rect.width > 0) {
          localX = ((ev.clientX - rect.left) / rect.width) * model.width;
        }
      }
      const u = Math.max(0, Math.min(1, (localX - model.padLeft) / plotW));
      const sCoord = u * chain.totalLength;
      const pick = pickChainPlotAtS(chain, sCoord, ld.drawnBeams);
      if (pick) dispatch({ type: "hover", at: pick });
    }),
    listen(svg, "pointerleave", () => dispatch({ type: "hover", at: null })),
  );

  card.append(svg);
  return card;
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
  onHoverSlot?: (slot: HTMLElement) => void,
  chainHoverSlots?: ChainHoverSlot[],
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

    const hoverSlot = el("div", "line-diagram-hover-slot");
    syncHoverReadoutSlot(hoverSlot, s);
    onHoverSlot?.(hoverSlot);
    box.append(hoverSlot);

    const chainsBox = el("div", "line-diagram-chains");
    if (ld.chains.length === 0) {
      chainsBox.append(
        el(
          "div",
          "legend-note line-diagram-chain-empty",
          "Select a 1-D beam (or connected beams) to plot its 2-D chain diagram.",
        ),
      );
    } else {
      ld.chains.forEach((chain, idx) => {
        chainsBox.append(renderChainCard(chain, idx, s, flipped, dispatch, unlisten, chainHoverSlots));
      });
    }
    box.append(chainsBox);
  }

  return box;
}

function lineDiagramSignature(s: State): string {
  const st = s.phase.at?.kind === "results" ? s.results?.stages[s.phase.at.stage] : undefined;
  const ld = s.lineDiagram;
  const chainsKey = ld.chains
    .map((c) => `${c.spans.map((sp) => `${sp.decl}:${sp.elementId}:${sp.reversed ? 1 : 0}`).join(">")}:${c.totalLength}`)
    .join("|");
  return JSON.stringify([
    st?.lineStations ?? [],
    ld.on,
    ld.component,
    ld.scale,
    ld.plane,
    ld.cmap,
    ld.fill,
    ld.flipSign,
    ld.range,
    ld.fixed,
    ld.stepRange,
    ld.scan,
    ld.notice,
    ld.drawnBeams.length,
    chainsKey,
    ld.on ? lineDiagramClimOf(s) : null,
  ]);
}
