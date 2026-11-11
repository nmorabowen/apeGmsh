// The line-diagram slice of the reducer (state/reduce.ts delegates here). It
// writes `lineDiagram` only. What a case reads of the rest of the state comes
// in as `LineDiagramInputs`; the derived chains and hover (`syncLineDiagramDerived`)
// and the stage pick (`withLineDiagramPick`) read the selection and the phase,
// so reduce.ts applies them after this returns.

import type { LineDiagramEvent } from "../types/events/lineDiagram.ts";
import { defaultLineDiagramComponent, LINE_DIAGRAM_CMAP_NAMES } from "../lineDiagram.ts";
import { LINE_STATION_COMPONENTS, type LineDiagramState, type ResultsStage } from "../types.ts";

/** What a line-diagram event reads of the rest of the state; each is asked only by the case that needs it. */
export interface LineDiagramInputs {
  /** The results stage shown (`setLineDiagramComponent`). */
  stage: () => ResultsStage | undefined;
  /** The keys of the step and scan on screen (`lineDiagramStepRange`, `lineDiagramScan`). */
  keys: () => { step: string; scan: string } | null;
}

/**
 * Auto-pick a line-station component when landing on a results stage with
 * `lineStations` and `lineDiagram.component` is null or absent from the stage.
 * The line diagram on `stage` (the stage shown), the same object when nothing
 * changed; `sync` asks the caller to re-derive the chains and the hover
 * (`syncLineDiagramDerived`, which reads the selection). reduce.ts
 * `withLineDiagramPick` applies both.
 */
export function pickLineDiagram(
  ld: LineDiagramState,
  stage: ResultsStage | undefined,
  keysOf: () => { step: string; scan: string } | null,
  preserveNotice: boolean,
): { ld: LineDiagramState; sync: boolean } {
  const avail = stage?.lineStations;
  if (!avail || avail.length === 0) {
    if (stage !== undefined && ld.component !== null) {
      const prevComp = ld.component;
      const notice = `Stage ${stage.name || stage.id} has no ${prevComp} line stations; the line-diagram pick was cleared.`;
      return {
        ld: {
          ...ld,
          component: null,
          flipSign: null,
          stepRange: null,
          scan: null,
          notice,
          drawnBeams: [],
          chains: [],
          hover: null,
        },
        sync: false,
      };
    }
    if (
      ld.stepRange === null &&
      ld.scan === null &&
      ld.drawnBeams.length === 0 &&
      ld.chains.length === 0 &&
      ld.hover === null
    ) {
      return { ld, sync: false };
    }
    return {
      ld: {
        ...ld,
        stepRange: null,
        scan: null,
        drawnBeams: [],
        chains: [],
        hover: null,
      },
      sync: false,
    };
  }
  if (ld.component !== null && avail.includes(ld.component)) {
    const keys = keysOf();
    const keepStep = keys !== null && ld.stepRange?.key === keys.step;
    const keepScan = keys !== null && ld.scan?.key === keys.scan;
    const nextNotice = keepStep || preserveNotice ? ld.notice : null;
    if (keepStep && keepScan && ld.notice === nextNotice) return { ld, sync: true };
    return {
      ld: {
        ...ld,
        stepRange: keepStep ? ld.stepRange : null,
        scan: keepScan ? ld.scan : null,
        notice: nextNotice,
        drawnBeams: keepStep ? ld.drawnBeams : [],
        chains: keepStep ? ld.chains : [],
        hover: keepStep ? ld.hover : null,
      },
      sync: true,
    };
  }
  const prevComp = ld.component;
  const picked = defaultLineDiagramComponent(avail);
  const notice =
    prevComp !== null && prevComp !== picked
      ? picked !== null
        ? `Stage ${stage.name || stage.id} has no ${prevComp} line stations; switched to ${picked}.`
        : `Stage ${stage.name || stage.id} has no ${prevComp} line stations; the line-diagram pick was cleared.`
      : null;
  if (picked === ld.component && ld.notice === notice) return { ld, sync: true };
  return {
    ld: {
      ...ld,
      component: picked,
      flipSign: null,
      stepRange: null,
      scan: null,
      notice,
      drawnBeams: [],
      chains: [],
      hover: null,
    },
    sync: false,
  };
}

/** The line diagram after `e`: the same object when nothing changed. */
export function reduceLineDiagram(ld: LineDiagramState, e: LineDiagramEvent, inputs: LineDiagramInputs): LineDiagramState {
  switch (e.type) {
    case "setLineDiagram": {
      const on: unknown = e.on;
      if (typeof on !== "boolean") throw new RangeError(`setLineDiagram: ${JSON.stringify(on)}`);
      return ld.on === on ? ld : { ...ld, on };
    }
    case "setLineDiagramComponent": {
      if (!LINE_STATION_COMPONENTS.includes(e.component)) {
        throw new RangeError(`setLineDiagramComponent: ${JSON.stringify(e.component)}`);
      }
      const stage = inputs.stage();
      if (stage?.lineStations && stage.lineStations.length > 0 && !stage.lineStations.includes(e.component)) {
        throw new RangeError(`setLineDiagramComponent: ${e.component} is not a line-station component of stage ${stage.id}`);
      }
      if (ld.component === e.component && ld.flipSign === null && ld.notice === null) return ld;
      return {
        ...ld,
        component: e.component,
        flipSign: null,
        stepRange: null,
        scan: null,
        notice: null,
        drawnBeams: [],
        chains: [],
        hover: null,
      };
    }
    case "setLineDiagramScale":
      if (typeof e.scale !== "number" || !Number.isFinite(e.scale) || e.scale <= 0) {
        throw new RangeError(`setLineDiagramScale: the scale must be a finite number above 0; got ${JSON.stringify(e.scale)}`);
      }
      return ld.scale === e.scale ? ld : { ...ld, scale: e.scale };
    case "setLineDiagramPlane":
      if (e.plane !== "local" && e.plane !== "screen") {
        throw new RangeError(`setLineDiagramPlane: ${JSON.stringify(e.plane)}`);
      }
      return ld.plane === e.plane && ld.notice === null ? ld : { ...ld, plane: e.plane, notice: null };
    case "setLineDiagramCmap":
      if (!LINE_DIAGRAM_CMAP_NAMES.includes(e.cmap)) {
        throw new RangeError(`setLineDiagramCmap: ${JSON.stringify(e.cmap)}`);
      }
      return ld.cmap === e.cmap ? ld : { ...ld, cmap: e.cmap };
    case "setLineDiagramFill":
      if (typeof e.fill !== "boolean") throw new RangeError(`setLineDiagramFill: ${JSON.stringify(e.fill)}`);
      return ld.fill === e.fill ? ld : { ...ld, fill: e.fill };
    case "setLineDiagramFlipSign":
      if (e.flipSign !== null && typeof e.flipSign !== "boolean") {
        throw new RangeError(`setLineDiagramFlipSign: ${JSON.stringify(e.flipSign)}`);
      }
      return ld.flipSign === e.flipSign ? ld : { ...ld, flipSign: e.flipSign };
    case "setLineDiagramRange": {
      if (e.range !== "history" && e.range !== "step" && e.range !== "fixed") {
        throw new RangeError(`setLineDiagramRange: ${JSON.stringify(e.range)}`);
      }
      let fixed = ld.fixed;
      if (e.range === "fixed") {
        fixed = e.fixed ?? ld.fixed;
        if (!fixed || !Number.isFinite(fixed.min) || !Number.isFinite(fixed.max) || !(fixed.min < fixed.max)) {
          throw new RangeError(`setLineDiagramRange: a fixed range needs a finite min below a finite max; got ${JSON.stringify(fixed)}`);
        }
      }
      return { ...ld, range: e.range, fixed };
    }
    case "lineDiagramStepRange": {
      if (inputs.keys()?.step !== e.key) return ld;
      const prevNotice = ld.notice;
      const keepSwitchNotice =
        ld.stepRange === null &&
        prevNotice !== null &&
        prevNotice.includes(" line stations; switched to ");
      const nextNotice =
        e.notice === undefined
          ? prevNotice
          : keepSwitchNotice
            ? e.notice === null || prevNotice.includes(e.notice)
              ? prevNotice
              : `${prevNotice} ${e.notice}`
            : e.notice;
      const drawnBeams =
        e.drawnBeams !== undefined
          ? e.drawnBeams.map((b) => ({
              decl: b.decl,
              elementId: b.elementId,
              nodeStart: b.nodeStart,
              nodeEnd: b.nodeEnd,
              pStart: [b.pStart[0], b.pStart[1], b.pStart[2]] as [number, number, number],
              pEnd: [b.pEnd[0], b.pEnd[1], b.pEnd[2]] as [number, number, number],
              length: b.length,
              xLocal: [b.xLocal[0], b.xLocal[1], b.xLocal[2]] as [number, number, number],
              extrudeDir: [b.extrudeDir[0], b.extrudeDir[1], b.extrudeDir[2]] as [number, number, number],
              xi: [...b.xi],
              values: [...b.values],
              autoScale: b.autoScale,
              ...(b.spanMode !== undefined ? { spanMode: b.spanMode } : {}),
              ...(b.spanReason !== undefined ? { spanReason: b.spanReason } : {}),
            }))
          : e.min === null
            ? []
            : ld.drawnBeams;
      return {
        ...ld,
        stepRange: { key: e.key, min: e.min, max: e.max },
        notice: nextNotice,
        drawnBeams,
      };
    }
    case "lineDiagramScan":
      if (inputs.keys()?.scan !== e.scan.key) return ld;
      return { ...ld, scan: e.scan };
    default: {
      const unknown: never = e;
      throw new Error(`reduceLineDiagram: unknown event ${JSON.stringify(unknown)}`);
    }
  }
}
