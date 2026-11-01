// Header panel: the model file's name and counts, the sibling
// artifacts whose paths are known, the global Display Units control,
// and the Spotlight (Ctrl+K) trigger pill. Renders from selectors alone.

import {
  modelSummary,
  resolveFieldUnits,
  UNIT_DISPLAY_OPTIONS,
  UNIT_SYSTEM_IDS,
  UNIT_SYSTEMS,
} from "../state/selectors.ts";
import type { State, Store } from "../state/store.ts";
import type { UnitDisplay, UnitSystemId } from "../state/types.ts";
import { byId, el, listen } from "../ui/dom.ts";

export function mountHeader(store: Store): () => void {
  const header = byId("header");
  const name = byId("file-name");
  const stats = byId("file-stats");
  const unitsWrap = el("div", "header-units");
  const pill = el("button", "spotlight-pill", "Search · Ctrl+K");
  pill.title = "Open Spotlight command & search palette (Ctrl+K)";
  const offPill = listen(pill, "click", () => store.dispatch({ type: "openSpotlight" }));
  header.append(unitsWrap, pill);

  const unitUnsubs: Array<() => void> = [];
  const clearUnitUnsubs = () => {
    while (unitUnsubs.length) unitUnsubs.pop()!();
  };

  let lastUnitsKey: string | null = null;
  let lastArtifacts: State["artifacts"] | null = null;
  let lastResults: State["results"] | null = null;

  const render = (s: State, prev: State | null) => {
    const unitsKey = `${s.contour.declaredUnits ?? ""}|${s.contour.sourceUnits ?? ""}|${s.contour.targetUnits}`;
    const unitsChanged = !prev || unitsKey !== lastUnitsKey;
    const artifactsChanged = !prev || s.artifacts !== lastArtifacts || s.results !== lastResults;
    if (!unitsChanged && !artifactsChanged) return;

    const summary = modelSummary(s);
    const resArt = s.artifacts.results;
    const hasResultsOnly =
      summary === null && (s.results !== null || (resArt !== null && resArt.status === "ready"));
    header.hidden = summary === null && !hasResultsOnly;
    if (header.hidden) {
      lastUnitsKey = null;
      lastArtifacts = null;
      lastResults = null;
      return;
    }

    if (artifactsChanged) {
      lastArtifacts = s.artifacts;
      lastResults = s.results;
      if (summary) {
        name.textContent = summary.name;
        stats.replaceChildren(summary.stats);
        for (const k of ["geometry", "results"] as const) {
          const a = s.artifacts[k];
          if (!a) continue;
          const file = a.path.split(/[\\/]/).pop() ?? a.path;
          stats.append(
            " · ",
            el("span", "sibling", `${k}: ${file}${a.status === "ready" ? "" : " (not read)"}`),
          );
        }
        document.title = `${summary.name} - apeGmshViewer`;
      } else {
        const rPath = s.results?.path ?? resArt?.path ?? "results.h5";
        const rFile = rPath.split(/[\\/]/).pop() ?? rPath;
        name.textContent = rFile;
        const stageCount = s.results?.stages.length ?? 0;
        const stageLabel = s.results
          ? `${stageCount} ${stageCount === 1 ? "stage" : "stages"} · results ${s.results.version}`
          : "results";
        stats.replaceChildren(stageLabel);
        document.title = `${rFile} - apeGmshViewer`;
      }
    }

    if (unitsChanged) {
      lastUnitsKey = unitsKey;
      clearUnitUnsubs();
      unitsWrap.replaceChildren();
      const u = resolveFieldUnits(
        "displacement",
        "nodal",
        s.contour.sourceUnits,
        s.contour.targetUnits,
        undefined,
        s.contour.declaredUnits,
      );
      unitsWrap.title = u.note;

      if (s.contour.declaredUnits === null) {
        const srcLabel = el("label", "header-units-label", "Native");
        srcLabel.title = u.note;
        const srcSelect = el("select", "header-units-source");
        srcSelect.title = u.note;
        const noneOpt = el("option", undefined, "unknown (no conversion)");
        noneOpt.value = "";
        noneOpt.selected = s.contour.sourceUnits === null;
        srcSelect.append(noneOpt);
        for (const id of UNIT_SYSTEM_IDS) {
          const o = el("option", undefined, UNIT_SYSTEMS[id].label);
          o.value = id;
          o.selected = id === s.contour.sourceUnits;
          srcSelect.append(o);
        }
        unitUnsubs.push(
          listen(srcSelect, "change", () => {
            const val = (srcSelect.value || null) as UnitSystemId | null;
            store.dispatch({ type: "setContourUnits", source: val, sourceUnits: val });
          }),
        );
        srcLabel.append(srcSelect);
        unitsWrap.append(srcLabel);
      }

      const dstLabel = el("label", "header-units-label", "Units");
      dstLabel.title = u.note;
      const dstSelect = el("select", "header-units-target");
      dstSelect.title = u.note;
      for (const opt of UNIT_DISPLAY_OPTIONS) {
        const label =
          opt.value !== "as_stored" && opt.value === s.contour.declaredUnits
            ? `${opt.label} (native)`
            : opt.label;
        const o = el("option", undefined, label);
        o.value = opt.value;
        o.selected = opt.value === s.contour.targetUnits;
        dstSelect.append(o);
      }
      unitUnsubs.push(
        listen(dstSelect, "change", () => {
          const val = dstSelect.value as UnitDisplay;
          store.dispatch({ type: "setContourUnits", target: val, targetUnits: val });
        }),
      );
      dstLabel.append(dstSelect);
      unitsWrap.append(dstLabel);
    }
  };
  render(store.get(), null);
  const unsubscribe = store.subscribe(render);
  return () => {
    unsubscribe();
    clearUnitUnsubs();
    offPill();
    unitsWrap.remove();
    pill.remove();
  };
}
