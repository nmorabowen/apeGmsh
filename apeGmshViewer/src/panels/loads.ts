// Standalone 3-D Structural Loads & Prescribed-SP panel (`#loads`), positioned
// on the left stack so loads, contours, and line diagrams can be controlled
// and inspected together. Auto-hidden when the loaded model has no loads.
//
// Follows ADR 0112 D6 (`panel-allow-list`): imports only `store`, `selectors`,
// `types`, and `ui/dom`.

import {
  fmtValue,
  loadsColourPalette,
  loadsSummaryOf,
  scaleLabel,
  UNIT_DISPLAY_OPTIONS,
  UNIT_SYSTEM_IDS,
  UNIT_SYSTEMS,
} from "../state/selectors.ts";
import type { State, Store } from "../state/store.ts";
import type { Event, UnitDisplay, UnitSystemId } from "../state/types.ts";
import { byId, el, listen } from "../ui/dom.ts";

const cssRgb = (rgb: readonly number[]): string =>
  `rgb(${rgb.map((c) => Math.round(c * 255)).join(",")})`;

const currentTheme = (): "dark" | "light" =>
  typeof document !== "undefined" && document.documentElement?.dataset?.["theme"] === "light" ? "light" : "dark";

export function mountLoads(store: Store, subscribeTheme?: (fn: () => void) => () => void): () => void {
  const panel = byId("loads");
  let unlisten: (() => void)[] = [];
  let lastSig = "";
  let lastTheme = "";

  const render = (s: State, prev: State | null) => {
    const theme = currentTheme();
    const sig = loadsSignature(s);
    if (prev && theme === lastTheme && sig === lastSig) return;
    lastTheme = theme;
    lastSig = sig;
    for (const u of unlisten) u();
    unlisten = [];
    const content = renderLoads(s, (e) => store.dispatch(e), unlisten);
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

export function renderLoads(
  s: State,
  dispatch: (e: Event) => void = () => {},
  unlisten: (() => void)[] = [],
): HTMLElement | null {
  if (!s.mesh || s.phase.at?.kind === "geometry") return null;
  const summary = loadsSummaryOf(s);
  if (!summary.hasLoads) return null;

  const ld = s.loads;
  const theme = currentTheme();
  const palette = loadsColourPalette(theme);
  const box = el("div", "loads");

  const head = el("h3", "loads-header", "Loads & SPs");
  if (ld.selectedPatterns !== null) {
    const allBtn = el("button", "legend-all loads-all", "active stage");
    allBtn.title = "reset pattern filter to active stage patterns";
    unlisten.push(listen(allBtn, "click", () => dispatch({ type: "setLoads", selectedPatterns: null })));
    head.append(allBtn);
  }
  box.append(head);

  const modes = el("div", "legend-mode loads-modes");
  const toggleBtn = el("button", "loads-toggle" + (ld.on ? " on" : ""), "3D loads");
  toggleBtn.title = "Show 3-D nodal forces, moments, element loads, and prescribed SPs";
  const forceBtn = el("button", "loads-forces" + (ld.showNodalForces ? " on" : ""), "F");
  forceBtn.title = "Show nodal forces (F)";
  const momentBtn = el("button", "loads-moments" + (ld.showNodalMoments ? " on" : ""), "M");
  momentBtn.title = "Show nodal moments (M, double-headed)";
  const elemBtn = el("button", "loads-element" + (ld.showElementLoads ? " on" : ""), "elem");
  elemBtn.title = "Show distributed beam and surface element loads";
  const spBtn = el("button", "loads-sp" + (ld.showSp ? " on" : ""), "SP");
  spBtn.title = "Show non-zero prescribed nodal displacements/rotations (u₀, θ₀)";
  unlisten.push(
    listen(toggleBtn, "click", () => dispatch({ type: "setLoads", on: !ld.on })),
    listen(forceBtn, "click", () => dispatch({ type: "setLoads", showNodalForces: !ld.showNodalForces })),
    listen(momentBtn, "click", () => dispatch({ type: "setLoads", showNodalMoments: !ld.showNodalMoments })),
    listen(elemBtn, "click", () => dispatch({ type: "setLoads", showElementLoads: !ld.showElementLoads })),
    listen(spBtn, "click", () => dispatch({ type: "setLoads", showSp: !ld.showSp })),
  );
  modes.append(toggleBtn, forceBtn, momentBtn, elemBtn, spBtn);
  box.append(modes);

  const optsRow = el("div", "legend-mode loads-options");
  const seriesBtn = el("button", "loads-series" + (ld.scaleBySeries ? " on" : ""), "scale by λ(t)");
  seriesBtn.title =
    summary.time !== null
      ? `Scale pattern loads by timeSeries factor λ(t) at t = ${fmtValue(summary.time)}`
      : "Scale pattern loads by timeSeries factor λ(t) when scrubbing results steps (nominal 1× on mesh/sequence)";
  const holdBtn = el("button", "loads-support-reactions" + (ld.includeSupportReactions ? " on" : ""), "support holds");
  holdBtn.title = "Include support_reaction hold patterns (role='support_reaction' / sp_holds)";
  unlisten.push(
    listen(seriesBtn, "click", () => dispatch({ type: "setLoads", scaleBySeries: !ld.scaleBySeries })),
    listen(holdBtn, "click", () =>
      dispatch({ type: "setLoads", includeSupportReactions: !ld.includeSupportReactions }),
    ),
  );
  optsRow.append(seriesBtn, holdBtn);
  box.append(optsRow);

  const scaleRow = el("div", "deform-scale-row loads-scale-row");
  const scaleInput = el("input", "deform-scale loads-scale");
  scaleInput.type = "number";
  scaleInput.step = "any";
  scaleInput.min = "0";
  scaleInput.value = String(ld.scale);
  scaleInput.title = "3-D load arrow & comb glyph scale multiplier";
  unlisten.push(
    listen(scaleInput, "change", () => {
      const v = Number(scaleInput.value);
      if (Number.isFinite(v) && v > 0) dispatch({ type: "setLoads", scale: v });
      else scaleInput.setCustomValidity("the scale must be a number above 0");
    }),
  );
  scaleRow.append(
    el("span", undefined, "scale"),
    scaleInput,
    el("span", "deform-scale-label loads-scale-label" + (ld.scale !== 1 ? " scaled" : ""), scaleLabel(ld.scale)),
  );
  box.append(scaleRow);

  // Shared unit conversion controls (`s.contour.{declaredUnits, sourceUnits, targetUnits}`)
  const u = summary.units;
  const unitBadges: string[] = [];
  if (u.force.unit) unitBadges.push(`F [${u.force.unit}]`);
  if (u.moment.unit) unitBadges.push(`M [${u.moment.unit}]`);
  if (u.beamLoad.unit) unitBadges.push(`q [${u.beamLoad.unit}]`);
  if (u.surfacePressure.unit) unitBadges.push(`p [${u.surfacePressure.unit}]`);
  if (u.displacement.unit) unitBadges.push(`u₀ [${u.displacement.unit}]`);
  if (u.rotation.unit) unitBadges.push(`θ₀ [${u.rotation.unit}]`);
  if (u.acceleration.unit) unitBadges.push(`a_g [${u.acceleration.unit}]`);
  const unitNote = unitBadges.length > 0 ? `${unitBadges.join(" · ")} — ${u.force.note}` : u.force.note;
  box.append(el("div", "legend-note loads-unit-note", unitNote));

  const unitRow = el("div", "contour-units loads-units");
  const dst = el("select", "contour-units-target");
  dst.title = "display unit system (baseUnits)";
  for (const opt of UNIT_DISPLAY_OPTIONS) {
    const label = opt.value !== "as_stored" && opt.value === s.contour.declaredUnits ? `${opt.label} (native)` : opt.label;
    const o = el("option", undefined, label);
    o.value = opt.value;
    o.selected = opt.value === s.contour.targetUnits;
    dst.append(o);
  }
  unlisten.push(
    listen(dst, "change", () => dispatch({ type: "setContourUnits", target: dst.value as UnitDisplay })),
  );
  unitRow.append(dst);
  if (s.contour.targetUnits !== "as_stored" && s.contour.declaredUnits === null) {
    const src = el("select", "contour-units-source");
    src.title = "file native unit system (from)";
    if (s.contour.sourceUnits === null) {
      const p = el("option", undefined, "from … (pick source units)");
      p.value = "";
      p.disabled = true;
      p.selected = true;
      src.append(p);
    }
    for (const id of UNIT_SYSTEM_IDS) {
      const o = el("option", undefined, `from ${UNIT_SYSTEMS[id].label}`);
      o.value = id;
      o.selected = id === s.contour.sourceUnits;
      src.append(o);
    }
    unlisten.push(
      listen(src, "change", () => {
        if (src.value) dispatch({ type: "setContourUnits", source: src.value as UnitSystemId });
      }),
    );
    unitRow.append(src);
  }
  box.append(unitRow);

  // Load kind legend rows with converted peak magnitudes and unit symbols
  const fmtPeak = (peak: number, unit: string | null): string =>
    peak > 0 ? `max ${fmtValue(peak)}${unit ? ` ${unit}` : ""}` : "";

  const pk = summary.peaks;
  const c = summary.counts;
  const kindRows: readonly { label: string; rgb: readonly number[]; count: number; peakText: string }[] = [
    {
      label: u.force.unit ? `Nodal force [${u.force.unit}]` : "Nodal force (F)",
      rgb: palette.nodalForce,
      count: c.nodalForces,
      peakText: fmtPeak(pk.nodalForce, u.force.unit),
    },
    {
      label: u.moment.unit ? `Nodal moment [${u.moment.unit}]` : "Nodal moment (M)",
      rgb: palette.nodalMoment,
      count: c.nodalMoments,
      peakText: fmtPeak(pk.nodalMoment, u.moment.unit),
    },
    {
      label: u.beamLoad.unit ? `Beam distributed [${u.beamLoad.unit}]` : "Beam distributed (q)",
      rgb: palette.elementLoad,
      count: c.beamUniform,
      peakText: fmtPeak(pk.beamUniform, u.beamLoad.unit),
    },
    {
      label: u.force.unit ? `Beam point [${u.force.unit}]` : "Beam point (P)",
      rgb: palette.elementLoad,
      count: c.beamPoint,
      peakText: fmtPeak(pk.beamPoint, u.force.unit),
    },
    {
      label: u.surfacePressure.unit ? `Surface pressure [${u.surfacePressure.unit}]` : "Surface pressure (p)",
      rgb: palette.elementLoad,
      count: c.surfacePressure,
      peakText: fmtPeak(pk.surfacePressure, u.surfacePressure.unit),
    },
    {
      label: u.displacement.unit ? `Prescribed SP [${u.displacement.unit}]` : "Prescribed SP (u₀)",
      rgb: palette.sp,
      count: c.spTranslations,
      peakText: fmtPeak(pk.spTranslation, u.displacement.unit),
    },
    {
      label: u.rotation.unit ? `Prescribed rot SP [${u.rotation.unit}]` : "Prescribed rot SP (θ₀)",
      rgb: palette.sp,
      count: c.spRotations,
      peakText: fmtPeak(pk.spRotation, u.rotation.unit),
    },
    {
      label: u.acceleration.unit ? `Base excitation [${u.acceleration.unit}]` : "Base excitation (a_g)",
      rgb: palette.sp,
      count: c.baseExcitations,
      peakText: fmtPeak(pk.baseExcitation, u.acceleration.unit),
    },
  ];
  for (const kr of kindRows) {
    if (kr.count === 0) continue;
    const row = el("div", "legend-row loads-kind-row");
    const chip = el("span", "chip");
    chip.style.backgroundColor = cssRgb(kr.rgb);
    const meta = kr.peakText ? `${kr.count.toLocaleString()} · ${kr.peakText}` : kr.count.toLocaleString();
    row.append(chip, el("span", undefined, kr.label), el("span", "legend-count loads-kind-meta", meta));
    box.append(row);
  }

  if (ld.on && summary.notice) {
    box.append(el("div", "legend-note loads-notice", summary.notice));
  }

  // Pattern / Case list
  const patBox = el("div", "loads-patterns");
  const stageCaption =
    summary.time !== null
      ? `Active: ${summary.activeStageLabel} · t = ${fmtValue(summary.time)}`
      : `Active: ${summary.activeStageLabel}`;
  patBox.append(el("div", "legend-note loads-stage-caption", stageCaption));

  for (const pat of summary.patterns) {
    if (pat.role === "support_reaction" && !ld.includeSupportReactions) continue;
    const row = el("div", "legend-row clickable loads-pattern-row" + (pat.active ? "" : " hidden-group"));
    row.dataset["patternId"] = pat.id;
    const badge = pat.stageIndex >= 0 ? `S${pat.stageIndex}` : pat.zone === "neutral" ? "neutral" : "global";
    const factorSuffix =
      ld.scaleBySeries && summary.time !== null ? ` · λ=${fmtValue(pat.seriesFactor)}` : "";
    const label = `${pat.title} (${badge}${factorSuffix})`;
    row.append(
      el("span", "loads-pattern-title", label),
      el("span", "legend-count", pat.counts.total.toLocaleString()),
    );
    row.title = "Click to toggle this pattern in the 3-D viewport";
    unlisten.push(
      listen(row, "click", () => {
        const currentIds =
          ld.selectedPatterns !== null
            ? [...ld.selectedPatterns]
            : summary.activePatterns.map((p) => p.id);
        const set = new Set(currentIds);
        if (set.has(pat.id)) set.delete(pat.id);
        else set.add(pat.id);
        dispatch({ type: "setLoads", on: true, selectedPatterns: [...set] });
      }),
    );
    patBox.append(row);
  }
  box.append(patBox);

  return box;
}

function loadsSignature(s: State): string {
  const summary = s.mesh && s.phase.at?.kind !== "geometry" ? loadsSummaryOf(s) : null;
  return JSON.stringify([
    s.mesh !== null,
    s.phase.at,
    s.loads,
    s.contour.declaredUnits,
    s.contour.sourceUnits,
    s.contour.targetUnits,
    summary
      ? {
          hasLoads: summary.hasLoads,
          counts: summary.counts,
          peaks: summary.peaks,
          notice: summary.notice,
          time: summary.time,
          patterns: summary.patterns.map((p) => [p.id, p.active, p.seriesFactor, p.counts.total]),
        }
      : null,
  ]);
}
