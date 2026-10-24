// The pure reducer: `reduce(state, event)` returns the next state and never
// mutates its arguments (test/store.test.ts freezes the input and checks).
// The event union is exhaustive: the `never` check at the end of the switch
// fails to compile when an event is added without a case, and an event the
// code does not know raises at run time.

import { initialClip, reduceClip } from "./clip.ts";
import { CMAP_NAMES, componentName, contourKeysOf, defaultPick, hasPick } from "./contour.ts";
import {
  defaultLineDiagramComponent,
  initialLineDiagramState,
  LINE_DIAGRAM_CMAP_NAMES,
  lineDiagramKeysOf,
} from "./lineDiagram.ts";
import { elementsOfDecl } from "./search.ts";
import { deformOf, elementJoinKey, geometryPairing, resultsPairing, stageCount, stageIdentity, syncLineDiagramDerived } from "./selectors.ts";
import { initialSearch, initialSpotlight } from "./spotlight.ts";
import {
  LINE_STATION_COMPONENTS,
  WINDOW_KINDS,
  type ArtifactInfo,
  type ArtifactKind,
  type ContourState,
  type DeclPath,
  type DeformState,
  type Event,
  type LineDiagramState,
  type LoadsState,
  type PhaseKey,
  type SpatialNdm,
  type State,
  type UnitSystemId,
} from "./types.ts";
import { UNIT_SYSTEM_IDS } from "./units.ts";

/** The deformed shape before the user turns it on: true scale, no ghost (V4d; the Qt defaults). */
export const initialDeform: DeformState = { on: false, scale: 1, ghost: false, measured: null };

export const initialLineDiagram: LineDiagramState = initialLineDiagramState;

export const initialLoads: LoadsState = {
  on: false,
  scale: 1,
  showNodalForces: true,
  showNodalMoments: true,
  showElementLoads: true,
  showSp: true,
  includeSupportReactions: false,
  scaleBySeries: false,
  selectedPatterns: null,
};

/** No model, no dimension. */
const NO_NDM: SpatialNdm = { ndm: null, reason: "no model is loaded" };

/** A record with no prototype: a key named `constructor` or `__proto__` is just a key. */
export const emptyRecord = <T>(): Record<string, T> => Object.create(null) as Record<string, T>;

/** The contour before anything is picked: whole-history range, jet (V4c1). */
export const initialContour: ContourState = {
  source: "nodal",
  gauss: "cell",
  join: null,
  field: null,
  component: null,
  range: "history",
  fixed: null,
  cmap: "jet",
  declaredUnits: null,
  sourceUnits: null,
  targetUnits: "as_stored",
  stepRange: null,
  scan: null,
  notice: null,
  reads: 0,
};

function syncContourUnits(c: ContourState, declaredUnits: UnitSystemId | null, keepManual: boolean): ContourState {
  if (declaredUnits !== null) {
    const targetUnits = keepManual && c.declaredUnits === declaredUnits && c.targetUnits !== "as_stored" ? c.targetUnits : declaredUnits;
    return { ...c, declaredUnits, sourceUnits: declaredUnits, targetUnits };
  }
  if (!keepManual || c.declaredUnits !== null) {
    return { ...c, declaredUnits: null, sourceUnits: null, targetUnits: "as_stored" };
  }
  return { ...c, declaredUnits: null };
}

export const initialState: State = {
  artifacts: { geometry: null, model: null, results: null },
  decls: emptyRecord(),
  names: emptyRecord(),
  blocks: [],
  mesh: null,
  phase: { axis: [], at: null },
  selection: { decls: [], picks: [] },
  hover: null,
  visibility: { hidden: [], edges: true, opacity: 1, colourBy: "group" },
  inspector: { pinned: [] },
  windows: { open: [] },
  overrides: {},
  geometry: null,
  provenance: [],
  provenanceOrigin: false,
  source: { seq: 0, request: null, last: null },
  view: { frameSeq: 0 },
  definitions: null,
  neutralLoads: null,
  sequence: { status: "absent" },
  stage: null,
  lineage: null,
  stageElements: { status: "none" },
  results: null,
  resultStep: null,
  contour: initialContour,
  ndm: NO_NDM,
  units: null,
  deform: initialDeform,
  clip: initialClip,
  lineDiagram: initialLineDiagram,
  loads: initialLoads,
  spotlight: initialSpotlight,
  search: initialSearch,
};

const NO_SELECTION: State["selection"] = { decls: [], picks: [] };

function opened(path: string): ArtifactInfo {
  return {
    path,
    status: "opened",
    error: null,
    stale: false,
    sizeBytes: 0,
    readMs: 0,
    zones: {},
    warnings: [],
    counts: { nodes: 0, cells: 0, opsOnly: 0 },
    sessionId: null,
  };
}

function samePhase(a: PhaseKey, b: PhaseKey): boolean {
  if (a.kind !== b.kind) return false;
  if (a.kind === "stage" && b.kind === "stage") return a.index === b.index;
  // The same results stage is the same phase; the step is the position within it.
  if (a.kind === "results" && b.kind === "results") return a.stage === b.stage;
  return true;
}

/** The steps of results stage `stage`, or 0 when there is no such stage. */
const stepsOf = (s: State, stage: number): number => s.results?.stages[stage]?.time.length ?? 0;

function checkStep(s: State, stage: number, step: number, who: string): void {
  const n = stepsOf(s, stage);
  if (!Number.isInteger(step) || step < 0 || step >= n) {
    throw new RangeError(`${who}: step ${JSON.stringify(step)} is outside results stage ${stage} (0..${n - 1})`);
  }
}

function artifactAt(s: State, path: string): ArtifactKind | null {
  for (const k of ["geometry", "model", "results"] as const) if (s.artifacts[k]?.path === path) return k;
  return null;
}

/** The decl paths of every colour group (physical groups), the unit `isolate` and `showAll` work on. */
function groupPaths(s: State): DeclPath[] {
  return Object.values(s.decls).filter((d) => d.kind === "physical_group").map((d) => d.path);
}

const uniq = (a: readonly DeclPath[]): DeclPath[] => [...new Set(a)];

/**
 * The phase axis from what is loaded: geometry when the sibling may be drawn
 * (it pairs with the model, or there is no model), then the mesh. The phase
 * shown stays where it was while it is still on the axis; otherwise it is the
 * mesh, or the geometry when there is no mesh.
 */
function withPhase(s: State, preserveLineNotice = false): State {
  const axis: PhaseKey[] = [];
  if (geometryPairing(s).draw) axis.push({ kind: "geometry" });
  if (s.mesh) axis.push({ kind: "mesh" });
  // V4b: one entry per results stage, only while the results pair with the
  // model by content (a mismatch draws nothing of them).
  if (s.mesh && s.results && resultsPairing(s).state === "paired") {
    s.results.stages.forEach((_, i) => axis.push({ kind: "results", stage: i, step: 0 }));
  }
  const prev = s.phase.at;
  let at = prev && axis.some((k) => samePhase(k, prev)) ? prev : (axis.find((k) => k.kind === "mesh") ?? axis[0] ?? null);
  // A results position keeps its step while the stage still has it; a stage
  // that shrank (a re-read) clamps it to its last step.
  if (at?.kind === "results") {
    const n = stepsOf(s, at.stage);
    if (n === 0) at = axis.find((k) => k.kind === "mesh") ?? null;
    else if (at.step >= n) at = { ...at, step: n - 1 };
  }
  const resultStep = at?.kind === "results" && s.resultStep && s.resultStep.stage === at.stage && s.resultStep.step === at.step ? s.resultStep : null;
  return withLineDiagramPick(withContourPick({ ...s, phase: { axis, at }, resultStep }), preserveLineNotice);
}

/** The stage a contour pick is checked against: the results stage shown. */
const shownStage = (s: State) => (s.phase.at?.kind === "results" ? s.results?.stages[s.phase.at.stage] : undefined);

/**
 * Colouring by contour with nothing picked, on a results stage: its first
 * field and component (V4c1). Not after a re-read cleared the pick: the
 * notice stands, and the user picks again (`setContourField` clears it).
 */
function withContourPick(s: State): State {
  if (s.visibility.colourBy !== "contour" || s.contour.field !== null || s.contour.notice !== null) return s;
  const pick = defaultPick(shownStage(s));
  return pick ? { ...s, contour: { ...s.contour, ...pick } } : s;
}

/**
 * Auto-pick a line-station component when landing on a results stage with
 * `lineStations` and `lineDiagram.component` is null or absent from the stage.
 */
function withLineDiagramPick(s: State, preserveNotice = false): State {
  const stage = shownStage(s);
  const avail = stage?.lineStations;
  if (!avail || avail.length === 0) {
    if (stage !== undefined && s.lineDiagram.component !== null) {
      const prevComp = s.lineDiagram.component;
      const notice = `Stage ${stage.name || stage.id} has no ${prevComp} line stations; the line-diagram pick was cleared.`;
      return {
        ...s,
        lineDiagram: {
          ...s.lineDiagram,
          component: null,
          flipSign: null,
          stepRange: null,
          scan: null,
          notice,
          drawnBeams: [],
          chains: [],
          hover: null,
        },
      };
    }
    if (
      s.lineDiagram.stepRange === null &&
      s.lineDiagram.scan === null &&
      s.lineDiagram.drawnBeams.length === 0 &&
      s.lineDiagram.chains.length === 0 &&
      s.lineDiagram.hover === null
    ) {
      return s;
    }
    return {
      ...s,
      lineDiagram: {
        ...s.lineDiagram,
        stepRange: null,
        scan: null,
        drawnBeams: [],
        chains: [],
        hover: null,
      },
    };
  }
  if (s.lineDiagram.component !== null && avail.includes(s.lineDiagram.component)) {
    const keys = lineDiagramKeysOf(s);
    const keepStep = keys !== null && s.lineDiagram.stepRange?.key === keys.step;
    const keepScan = keys !== null && s.lineDiagram.scan?.key === keys.scan;
    const nextNotice = keepStep || preserveNotice ? s.lineDiagram.notice : null;
    if (keepStep && keepScan && s.lineDiagram.notice === nextNotice) return syncLineDiagramDerived(s);
    return syncLineDiagramDerived({
      ...s,
      lineDiagram: {
        ...s.lineDiagram,
        stepRange: keepStep ? s.lineDiagram.stepRange : null,
        scan: keepScan ? s.lineDiagram.scan : null,
        notice: nextNotice,
        drawnBeams: keepStep ? s.lineDiagram.drawnBeams : [],
        chains: keepStep ? s.lineDiagram.chains : [],
        hover: keepStep ? s.lineDiagram.hover : null,
      },
    });
  }
  const prevComp = s.lineDiagram.component;
  const picked = defaultLineDiagramComponent(avail);
  const notice =
    prevComp !== null && prevComp !== picked
      ? picked !== null
        ? `Stage ${stage.name || stage.id} has no ${prevComp} line stations; switched to ${picked}.`
        : `Stage ${stage.name || stage.id} has no ${prevComp} line stations; the line-diagram pick was cleared.`
      : null;
  if (picked === s.lineDiagram.component && s.lineDiagram.notice === notice) return syncLineDiagramDerived(s);
  return {
    ...s,
    lineDiagram: {
      ...s.lineDiagram,
      component: picked,
      flipSign: null,
      stepRange: null,
      scan: null,
      notice,
      drawnBeams: [],
      chains: [],
      hover: null,
    },
  };
}

export function reduce(s: State, e: Event): State {
  switch (e.type) {
    case "fileOpened": {
      const prev = s.artifacts[e.artifact];
      // The same path again (a reload after fileChanged) keeps what was read until the new read lands.
      const info: ArtifactInfo = prev && prev.path === e.path ? { ...prev, stale: false, error: null } : opened(e.path);
      // Another results file: the metadata read from the old one goes with it.
      const clearedResults = e.artifact === "results" && !(prev && prev.path === e.path);
      const results = clearedResults ? null : s.results;
      const contour = clearedResults ? syncContourUnits(s.contour, s.units, s.artifacts.model?.status === "ready") : s.contour;
      const lineDiagram = clearedResults
        ? { ...s.lineDiagram, stepRange: null, scan: null, drawnBeams: [], chains: [], hover: null }
        : s.lineDiagram;
      return withPhase({ ...s, artifacts: { ...s.artifacts, [e.artifact]: info }, results, contour, lineDiagram });
    }
    case "fileLoaded": {
      // A fileChanged that arrived while this read ran keeps the artifact
      // stale, so the effects read it once more.
      const prev = s.artifacts[e.artifact];
      const info: ArtifactInfo = { ...e.load.info, stale: prev !== null && prev.path === e.load.info.path && prev.stale };
      if (e.artifact === "geometry") {
        return withPhase({ ...s, artifacts: { ...s.artifacts, geometry: info }, geometry: e.load.geometry });
      }
      const { decls, names, blocks, mesh, provenance, provenanceOrigin, definitions, sequence, lineage, stageElements } = e.load;
      // A model read for the first time opens on the mesh. A re-read of the
      // same file (D1 rewrites it every run) keeps the phase shown only while
      // that phase is still on the axis: a re-read model with a new
      // session_id no longer pairs with the old geometry, so the geometry
      // phase leaves the axis. That is why a D1 re-run re-reads the model and
      // its geometry together and lands them in one `setLoaded`.
      const reread = prev !== null && prev.status === "ready" && prev.path === info.path;
      // The stage selection survives a re-read only while its row still
      // names the same stage; a new file, or a changed row, clears it.
      const keep = reread && s.stage !== null && stageIdentity(s.sequence, s.stage) === stageIdentity(sequence, s.stage);
      const modelUnits = e.load.units ?? null;
      const declaredUnits = modelUnits ?? s.results?.units ?? null;
      const hasLines = mesh.counts.lineCells > 0 || mesh.counts.opsOnly > 0;
      const opacity = reread ? s.visibility.opacity : mesh.counts.solidCells > 0 && hasLines ? 0.35 : 1;
      return withPhase({
        ...s,
        phase: reread ? s.phase : { ...s.phase, at: null },
        artifacts: { ...s.artifacts, model: info },
        decls,
        names,
        blocks,
        mesh,
        provenance,
        provenanceOrigin,
        definitions,
        neutralLoads: e.load.neutralLoads ?? null,
        sequence,
        lineage,
        stageElements,
        ndm: e.load.ndm ?? { ndm: null, reason: "the model's /meta/ndm was not read" },
        units: modelUnits,
        contour: syncContourUnits(s.contour, declaredUnits, reread || s.results !== null),
        loads: reread ? s.loads : { ...s.loads, selectedPatterns: null },
        stage: keep ? s.stage : null,
        selection: NO_SELECTION,
        hover: null,
        visibility: { ...s.visibility, hidden: [], opacity },
        inspector: { pinned: [] },
        spotlight: reread ? s.spotlight : { ...initialSpotlight, seq: s.spotlight.seq },
        search: reread ? s.search : { ...initialSearch, seq: s.search.seq },
        // The request count survives the re-read (see State.source).
        source: { seq: s.source.seq, request: null, last: null },
      });
    }
    case "setLoaded": {
      // Geometry first, then the model: the pairing is judged once, on the pair.
      let t = s;
      if (e.geometry) t = reduce(t, { type: "fileLoaded", artifact: "geometry", load: e.geometry });
      if (e.model) t = reduce(t, { type: "fileLoaded", artifact: "model", load: e.model });
      return t;
    }
    case "fileFailed": {
      const prev = s.artifacts[e.artifact];
      // A fileChanged that arrived during the failed read keeps the artifact
      // stale (the read may have hit a half-written file); the effects retry.
      const stale = prev !== null && prev.path === e.path && prev.stale;
      const info: ArtifactInfo = { ...(prev && prev.path === e.path ? prev : opened(e.path)), status: "failed", error: e.error, stale };
      if (e.artifact === "geometry") return withPhase({ ...s, artifacts: { ...s.artifacts, geometry: info }, geometry: null });
      if (e.artifact === "results") {
        return withPhase({
          ...s,
          artifacts: { ...s.artifacts, results: info },
          results: null,
          contour: syncContourUnits(s.contour, s.units, s.artifacts.model?.status === "ready"),
          lineDiagram: { ...s.lineDiagram, stepRange: null, scan: null, drawnBeams: [], chains: [], hover: null },
        });
      }
      // The model is gone with its derivations; the failure is shown instead.
      // The results stay read (they re-pair when a model loads).
      return withPhase({
        ...initialState,
        artifacts: { ...s.artifacts, model: info },
        visibility: s.visibility,
        windows: s.windows,
        geometry: s.geometry,
        results: s.results,
        contour: syncContourUnits(initialState.contour, s.results?.units ?? null, false),
        spotlight: { ...initialSpotlight, seq: s.spotlight.seq },
        search: { ...initialSearch, seq: s.search.seq },
        source: { ...initialState.source, seq: s.source.seq },
        // V4d: the user's deformed-shape settings stand; the count was for the old model.
        deform: { ...s.deform, measured: null },
        // V4e1: the section planes stand too (the slider re-derives its range from the next mesh).
        clip: s.clip,
        lineDiagram: { ...s.lineDiagram, stepRange: null, scan: null, drawnBeams: [], chains: [], hover: null },
        loads: { ...s.loads, selectedPatterns: null },
      });
    }
    case "fileClosed": {
      if (!s.artifacts[e.artifact]) return s;
      if (e.artifact === "geometry") return withPhase({ ...s, artifacts: { ...s.artifacts, geometry: null }, geometry: null });
      if (e.artifact === "results") {
        return withPhase({
          ...s,
          artifacts: { ...s.artifacts, results: null },
          results: null,
          contour: syncContourUnits(s.contour, s.units, s.artifacts.model?.status === "ready"),
          lineDiagram: { ...s.lineDiagram, stepRange: null, scan: null, drawnBeams: [], chains: [], hover: null },
        });
      }
      return withPhase({
        ...initialState,
        artifacts: { ...s.artifacts, model: null },
        visibility: s.visibility,
        windows: s.windows,
        geometry: s.geometry,
        results: s.results,
        contour: syncContourUnits(initialState.contour, s.results?.units ?? null, false),
        spotlight: { ...initialSpotlight, seq: s.spotlight.seq },
        search: { ...initialSearch, seq: s.search.seq },
        source: { ...initialState.source, seq: s.source.seq },
        // V4d: the user's deformed-shape settings stand; the count was for the old model.
        deform: { ...s.deform, measured: null },
        // V4e1: the section planes stand too (the slider re-derives its range from the next mesh).
        clip: s.clip,
        lineDiagram: { ...s.lineDiagram, stepRange: null, scan: null, drawnBeams: [], chains: [], hover: null },
        loads: { ...s.loads, selectedPatterns: null },
      });
    }
    case "fileChanged": {
      const k = artifactAt(s, e.path);
      if (!k) return s;
      // Only marks the file stale: the sequence shown is still the one read,
      // so the stage selection stands until the re-read lands (fileLoaded
      // keeps it only if that row still names the same stage).
      return { ...s, artifacts: { ...s.artifacts, [k]: { ...s.artifacts[k]!, stale: true } } };
    }
    case "zoneRefused": {
      const prev = s.artifacts[e.artifact];
      if (!prev) throw new Error(`zoneRefused for ${e.artifact} before fileOpened`);
      const zones = {
        ...prev.zones,
        [e.zone]: { status: "refused" as const, version: e.version, accepted: e.accepted, newer: e.newer, reason: e.reason },
      };
      return { ...s, artifacts: { ...s.artifacts, [e.artifact]: { ...prev, zones } } };
    }
    case "select":
      if (!(e.pick.decl in s.decls)) throw new Error(`select: ${e.pick.decl} is not a declaration of the loaded model`);
      return syncLineDiagramDerived({ ...s, selection: { decls: [e.pick.decl], picks: [e.pick] } });
    case "selectAdd": {
      if (!(e.pick.decl in s.decls)) throw new Error(`selectAdd: ${e.pick.decl} is not a declaration of the loaded model`);
      if (s.selection.decls.includes(e.pick.decl)) {
        const decls = s.selection.decls.filter((d) => d !== e.pick.decl);
        const picks = s.selection.picks.filter((p) => p.decl !== e.pick.decl);
        return syncLineDiagramDerived(
          decls.length === 0 ? { ...s, selection: NO_SELECTION } : { ...s, selection: { decls, picks } },
        );
      }
      return syncLineDiagramDerived({
        ...s,
        selection: { decls: [...s.selection.decls, e.pick.decl], picks: [...s.selection.picks, e.pick] },
      });
    }
    case "selectDecls": {
      for (const d of e.decls) {
        if (!(d in s.decls)) throw new Error(`selectDecls: ${d} is not a declaration of the loaded model`);
      }
      const next = uniq(e.add ? [...s.selection.decls, ...e.decls] : e.decls);
      const picks = next.map((decl) => s.selection.picks.find((p) => p.decl === decl) ?? { decl, at: null, nodes: null });
      return syncLineDiagramDerived({ ...s, selection: { decls: next, picks } });
    }
    case "clearSelection":
      return s.selection.decls.length === 0 ? s : syncLineDiagramDerived({ ...s, selection: NO_SELECTION });
    case "hover":
      return syncLineDiagramDerived({ ...s, hover: e.at });
    case "setHidden": {
      const hidden = e.hidden
        ? uniq([...s.visibility.hidden, ...e.decls])
        : s.visibility.hidden.filter((d) => !e.decls.includes(d));
      return { ...s, visibility: { ...s.visibility, hidden } };
    }
    case "isolate": {
      const allGroups = e.decls.length > 0 && e.decls.every((d) => s.decls[d]?.kind === "physical_group");
      if (allGroups) {
        const keep = new Set(e.decls);
        return { ...s, visibility: { ...s.visibility, hidden: groupPaths(s).filter((p) => !keep.has(p)) } };
      }
      const keepElems = new Set<DeclPath>();
      for (const d of e.decls) {
        for (const el of elementsOfDecl(s, d)) keepElems.add(el);
      }
      const allElems = s.mesh ? s.mesh.elements : Object.values(s.decls).filter((d) => d.kind === "element").map((d) => d.path);
      return { ...s, visibility: { ...s.visibility, hidden: allElems.filter((p) => !keepElems.has(p)) } };
    }
    case "showAll":
      return { ...s, visibility: { ...s.visibility, hidden: [] } };
    case "setEdges":
      return { ...s, visibility: { ...s.visibility, edges: e.on } };
    case "setOpacity":
      if (!Number.isFinite(e.value)) throw new RangeError(`setOpacity: ${e.value} is not a number`);
      return { ...s, visibility: { ...s.visibility, opacity: Math.min(1, Math.max(0, e.value)) } };
    case "setColourBy":
      if (e.by !== "group" && e.by !== "role" && e.by !== "contour") throw new RangeError(`setColourBy: ${JSON.stringify(e.by)}`);
      return s.visibility.colourBy === e.by ? s : withContourPick({ ...s, visibility: { ...s.visibility, colourBy: e.by } });
    case "setPhase":
      if (!s.phase.axis.some((k) => samePhase(k, e.at))) throw new Error(`setPhase: ${JSON.stringify(e.at)} is not on the phase axis`);
      if (e.at.kind === "results") checkStep(s, e.at.stage, e.at.step, "setPhase");
      return withPhase({
        ...s,
        phase: { ...s.phase, at: e.at },
        ...(e.at.kind !== "results" ? { lineDiagram: { ...s.lineDiagram, notice: null } } : {}),
      });
    case "setResultStep": {
      if (!s.phase.axis.some((k) => k.kind === "results")) throw new Error("setResultStep: the phase axis has no results");
      const at = s.phase.at;
      if (at?.kind !== "results") throw new Error("setResultStep: the phase shown is not a results stage (setPhase to one first)");
      checkStep(s, at.stage, e.step, "setResultStep");
      if (at.step === e.step) return s;
      return withPhase({ ...s, phase: { ...s.phase, at: { kind: "results", stage: at.stage, step: e.step } } });
    }
    case "resultsLoaded": {
      const prev = s.artifacts.results;
      if (!prev || prev.path !== e.info.path) throw new Error(`resultsLoaded: ${e.info.path} is not the open results artifact`);
      const info: ArtifactInfo = {
        ...prev,
        status: "ready",
        error: null,
        sizeBytes: e.sizeBytes,
        readMs: e.readMs,
        zones: { results: { status: "ready", version: e.info.version } },
        warnings: [...e.warnings, ...(e.info.partial ? [`${e.info.path.split(/[\\/]/).pop()}: ${e.info.partial}; it is read as it stands`] : [])],
      };
      // A re-read keeps the results position only while that index still
      // names the same stage (its id and name); otherwise the phase leaves it.
      const at = s.phase.at;
      const old = at?.kind === "results" ? s.results?.stages[at.stage] : undefined;
      const now = at?.kind === "results" ? e.info.stages[at.stage] : undefined;
      const keep = !old || (now !== undefined && now.id === old.id && now.name === old.name);
      const declaredUnits = s.units ?? e.info.units ?? null;
      const rereadResults = prev.status === "ready";
      const firstLineStage = e.info.stages.find((st) => (st.lineStations?.length ?? 0) > 0);
      const curComp = s.lineDiagram.component;
      const validOnAny =
        curComp !== null && e.info.stages.some((st) => st.lineStations?.includes(curComp));
      // Mixed continuum + beam stages (`gauss.length > 0`, e.g. 3-D floor slabs
      // on horizontal girders with `vecxz = +Z`, #2142) carry vertical gravity
      // bending in `My` (bending about local y in the XZ plane) rather than
      // weak-axis `Mz`, so initial load prefers `My` when both are recorded.
      const preferSlabMy =
        (firstLineStage?.gauss.length ?? 0) > 0 &&
        Boolean(firstLineStage?.lineStations?.includes("My")) &&
        Boolean(firstLineStage?.lineStations?.includes("Mz"));
      const nextLineComp = validOnAny
        ? curComp
        : preferSlabMy
          ? "My"
          : defaultLineDiagramComponent(firstLineStage?.lineStations);
      const fileName = e.info.path.split(/[\\/]/).pop() ?? e.info.path;
      const lineSwitchNotice =
        curComp !== null && nextLineComp !== curComp
          ? nextLineComp !== null
            ? `${fileName} has no ${curComp} line stations; switched to ${nextLineComp}.`
            : `${fileName} has no ${curComp} line stations; the line-diagram pick was cleared.`
          : null;
      // V4c1: a re-read invalidates the measured ranges (the same path, other
      // data); a pick the re-read stage no longer has is cleared, with a notice.
      const next = withPhase(
        {
          ...s,
          artifacts: { ...s.artifacts, results: info },
          results: e.info,
          phase: keep ? s.phase : { ...s.phase, at: null },
          // V4c2a: the element join is checked again on the re-read's rows.
          contour: syncContourUnits(
            { ...s.contour, stepRange: null, scan: null, join: null, reads: s.contour.reads + 1 },
            declaredUnits,
            rereadResults || s.artifacts.model?.status === "ready",
          ),
          lineDiagram: {
            ...s.lineDiagram,
            component: nextLineComp,
            flipSign: nextLineComp === curComp ? s.lineDiagram.flipSign : null,
            stepRange: null,
            scan: null,
            notice: lineSwitchNotice,
            drawnBeams: [],
            chains: [],
            hover: null,
          },
        },
        lineSwitchNotice !== null,
      );
      const c = next.contour;
      if (c.field === null || c.component === null) return next;
      const field = c.field, component = c.component;
      const stage = shownStage(next);
      const has = stage ? hasPick(stage, c.source, field, component) : e.info.stages.some((st) => hasPick(st, c.source, field, component));
      if (has) return next;
      const notice =
        `The re-read ${e.info.path.split(/[\\/]/).pop()} has no ${field} ${componentName(component)}` +
        `${stage ? ` in stage ${stage.name || stage.id}` : ""}: the contour pick was cleared.`;
      return { ...next, contour: { ...c, field: null, component: null, notice } };
    }
    case "setContourField": {
      const source = e.source ?? "nodal";
      if (source !== "nodal" && source !== "element") throw new RangeError(`setContourField: source ${JSON.stringify(source)}`);
      const stage = shownStage(s);
      const ok = stage ? hasPick(stage, source, e.field, e.component) : (s.results?.stages ?? []).some((st) => hasPick(st, source, e.field, e.component));
      if (!ok) throw new RangeError(`setContourField: ${e.field} / ${e.component} is not a ${source} field component of ${stage ? `stage ${stage.id}` : "the results"}`);
      const c = s.contour;
      if (c.source === source && c.field === e.field && c.component === e.component && c.notice === null) return s;
      // Another field is joined to the mesh again (its groups can differ).
      const join = c.source === source && c.field === e.field ? c.join : null;
      return { ...s, contour: { ...c, source, field: e.field, component: e.component, join, notice: null } };
    }
    case "setContourGauss":
      if (e.mode !== "cell" && e.mode !== "cell_averaged" && e.mode !== "node" && e.mode !== "node_discrete") throw new RangeError(`setContourGauss: ${JSON.stringify(e.mode)}`);
      return s.contour.gauss === e.mode ? s : { ...s, contour: { ...s.contour, gauss: e.mode } };
    case "contourElementJoin":
      // Only for the field, file read and mesh shown: a late verdict for another is dropped.
      if (elementJoinKey(s) !== e.verdict.key) return s;
      return { ...s, contour: { ...s.contour, join: e.verdict } };
    case "setContourRange": {
      if (e.range !== "history" && e.range !== "step" && e.range !== "fixed") throw new RangeError(`setContourRange: ${JSON.stringify(e.range)}`);
      let fixed = s.contour.fixed;
      if (e.range === "fixed") {
        fixed = e.fixed ?? s.contour.fixed;
        if (!fixed || !Number.isFinite(fixed.min) || !Number.isFinite(fixed.max) || !(fixed.min < fixed.max)) {
          throw new RangeError(`setContourRange: a fixed range needs a finite min below a finite max; got ${JSON.stringify(fixed)}`);
        }
      }
      return { ...s, contour: { ...s.contour, range: e.range, fixed } };
    }
    case "setContourCmap":
      if (!CMAP_NAMES.includes(e.cmap)) throw new RangeError(`setContourCmap: ${JSON.stringify(e.cmap)}`);
      return s.contour.cmap === e.cmap ? s : { ...s, contour: { ...s.contour, cmap: e.cmap } };
    case "setContourUnits": {
      if (e.source !== undefined && !UNIT_SYSTEM_IDS.includes(e.source)) {
        throw new RangeError(`setContourUnits: source ${JSON.stringify(e.source)}`);
      }
      const targetUnits = e.target ?? s.contour.targetUnits;
      if (targetUnits !== "as_stored" && !UNIT_SYSTEM_IDS.includes(targetUnits)) {
        throw new RangeError(`setContourUnits: target ${JSON.stringify(targetUnits)}`);
      }
      const sourceUnits = s.contour.declaredUnits ?? e.source ?? s.contour.sourceUnits;
      if (s.contour.sourceUnits === sourceUnits && s.contour.targetUnits === targetUnits) return s;
      return { ...s, contour: { ...s.contour, sourceUnits, targetUnits } };
    }
    case "contourStepRange":
      // Only the step and component on screen: a late answer for another is dropped.
      if (contourKeysOf(s)?.step !== e.key) return s;
      return { ...s, contour: { ...s.contour, stepRange: { key: e.key, min: e.min, max: e.max } } };
    case "contourScan":
      if (contourKeysOf(s)?.scan !== e.scan.key) return s;
      return { ...s, contour: { ...s.contour, scan: e.scan } };
    case "setDeform": {
      const on: unknown = e.on;
      if (typeof on !== "boolean") throw new RangeError(`setDeform: ${JSON.stringify(on)}`);
      return s.deform.on === on ? s : { ...s, deform: { ...s.deform, on } };
    }
    case "setDeformScale":
      if (typeof e.scale !== "number" || !Number.isFinite(e.scale) || e.scale <= 0) {
        throw new RangeError(`setDeformScale: the scale must be a finite number above 0; got ${JSON.stringify(e.scale)}`);
      }
      return s.deform.scale === e.scale ? s : { ...s, deform: { ...s.deform, scale: e.scale } };
    case "setReferenceGhost": {
      const on: unknown = e.on;
      if (typeof on !== "boolean") throw new RangeError(`setReferenceGhost: ${JSON.stringify(on)}`);
      return s.deform.ghost === on ? s : { ...s, deform: { ...s.deform, ghost: on } };
    }
    case "deformMeasured": {
      // A count for anything but what is shown now (another step, a re-read, another hidden set) is stale.
      const d = deformOf(s);
      if (!d.ok || d.draw.keys.count !== e.key) return s;
      if (!Number.isInteger(e.drawn) || !Number.isInteger(e.missing) || e.missing < 0 || e.missing > e.drawn) {
        throw new RangeError(`deformMeasured: ${e.missing} missing of ${e.drawn} drawn nodes`);
      }
      return { ...s, deform: { ...s.deform, measured: { key: e.key, drawn: e.drawn, missing: e.missing } } };
    }
    case "resultStep": {
      // A status for a step that is no longer shown is dropped (the effects
      // also drop replies that are not their latest request).
      const at = s.phase.at;
      if (at?.kind !== "results" || at.stage !== e.status.stage || at.step !== e.status.step) return s;
      return { ...s, resultStep: e.status };
    }
    case "inspectorPin":
      if (!(e.decl in s.decls)) throw new Error(`inspectorPin: ${e.decl} is not a declaration of the loaded model`);
      if (s.inspector.pinned.includes(e.decl)) return s;
      return { ...s, inspector: { pinned: [...s.inspector.pinned, e.decl] } };
    case "inspectorUnpin":
      return { ...s, inspector: { pinned: s.inspector.pinned.filter((d) => d !== e.decl) } };
    case "openWindow":
      if (!WINDOW_KINDS.includes(e.window)) throw new RangeError(`openWindow: ${JSON.stringify(e.window)} is not a window kind`);
      if (s.windows.open.includes(e.window)) return s;
      return { ...s, windows: { open: [...s.windows.open, e.window] } };
    case "closeWindow":
      return { ...s, windows: { open: s.windows.open.filter((w) => w !== e.window) } };
    case "requestSource": {
      // A declaration, or a /provenance record the app has no declaration
      // for (a synthesised series or pattern, from the sources listing).
      if (!(e.decl in s.decls) && !s.provenance.some((p) => p.key === e.decl)) {
        throw new Error(`requestSource: ${e.decl} is not a declaration or a /provenance record of the loaded model`);
      }
      const seq = s.source.seq + 1;
      return { ...s, source: { seq, request: { decl: e.decl, seq }, last: null } };
    }
    case "sourceResult":
      return { ...s, source: { ...s.source, last: { decl: e.decl, ok: e.ok, reason: e.reason } } };
    case "frameSelection":
      if (!s.mesh) return s;
      return { ...s, view: { frameSeq: s.view.frameSeq + 1 } };
    case "selectStage": {
      if (e.index === null) return s.stage === null ? s : { ...s, stage: null };
      const n = stageCount(s.sequence);
      if (!Number.isInteger(e.index) || e.index < 0 || e.index >= n) {
        throw new RangeError(`selectStage: ${JSON.stringify(e.index)} is not a /sequence/stages row (the loaded model has ${n})`);
      }
      return s.stage === e.index ? s : { ...s, stage: e.index };
    }
    case "addClipPlane":
    case "removeClipPlane":
    case "setClipActive":
    case "setClipFlipped":
    case "setClipOffset":
    case "setClipNormal":
    case "setApplyCuts":
    case "setClipPose":
    case "setShowGizmos":
    case "setClipGizmo":
    case "setCutFace": {
      // V4e1: planes are world-space; a new one passes through the reference (undeformed) mesh centre.
      const clip = reduceClip(s.clip, e, s.mesh?.center ?? [0, 0, 0]);
      return clip === s.clip ? s : { ...s, clip };
    }
    case "setLineDiagram": {
      const on: unknown = e.on;
      if (typeof on !== "boolean") throw new RangeError(`setLineDiagram: ${JSON.stringify(on)}`);
      return s.lineDiagram.on === on
        ? s
        : syncLineDiagramDerived(withLineDiagramPick({ ...s, lineDiagram: { ...s.lineDiagram, on } }));
    }
    case "setLineDiagramComponent": {
      if (!LINE_STATION_COMPONENTS.includes(e.component)) {
        throw new RangeError(`setLineDiagramComponent: ${JSON.stringify(e.component)}`);
      }
      const stage = shownStage(s);
      if (stage?.lineStations && stage.lineStations.length > 0 && !stage.lineStations.includes(e.component)) {
        throw new RangeError(`setLineDiagramComponent: ${e.component} is not a line-station component of stage ${stage.id}`);
      }
      if (s.lineDiagram.component === e.component && s.lineDiagram.flipSign === null && s.lineDiagram.notice === null) return s;
      return {
        ...s,
        lineDiagram: {
          ...s.lineDiagram,
          component: e.component,
          flipSign: null,
          stepRange: null,
          scan: null,
          notice: null,
          drawnBeams: [],
          chains: [],
          hover: null,
        },
      };
    }
    case "setLineDiagramScale":
      if (typeof e.scale !== "number" || !Number.isFinite(e.scale) || e.scale <= 0) {
        throw new RangeError(`setLineDiagramScale: the scale must be a finite number above 0; got ${JSON.stringify(e.scale)}`);
      }
      return s.lineDiagram.scale === e.scale
        ? s
        : syncLineDiagramDerived({ ...s, lineDiagram: { ...s.lineDiagram, scale: e.scale } });
    case "setLineDiagramPlane":
      if (e.plane !== "local" && e.plane !== "screen") {
        throw new RangeError(`setLineDiagramPlane: ${JSON.stringify(e.plane)}`);
      }
      return s.lineDiagram.plane === e.plane && s.lineDiagram.notice === null
        ? s
        : { ...s, lineDiagram: { ...s.lineDiagram, plane: e.plane, notice: null } };
    case "setLineDiagramCmap":
      if (!LINE_DIAGRAM_CMAP_NAMES.includes(e.cmap)) {
        throw new RangeError(`setLineDiagramCmap: ${JSON.stringify(e.cmap)}`);
      }
      return s.lineDiagram.cmap === e.cmap ? s : { ...s, lineDiagram: { ...s.lineDiagram, cmap: e.cmap } };
    case "setLineDiagramFill":
      if (typeof e.fill !== "boolean") throw new RangeError(`setLineDiagramFill: ${JSON.stringify(e.fill)}`);
      return s.lineDiagram.fill === e.fill ? s : { ...s, lineDiagram: { ...s.lineDiagram, fill: e.fill } };
    case "setLineDiagramFlipSign":
      if (e.flipSign !== null && typeof e.flipSign !== "boolean") {
        throw new RangeError(`setLineDiagramFlipSign: ${JSON.stringify(e.flipSign)}`);
      }
      return s.lineDiagram.flipSign === e.flipSign
        ? s
        : syncLineDiagramDerived({ ...s, lineDiagram: { ...s.lineDiagram, flipSign: e.flipSign } });
    case "setLineDiagramRange": {
      if (e.range !== "history" && e.range !== "step" && e.range !== "fixed") {
        throw new RangeError(`setLineDiagramRange: ${JSON.stringify(e.range)}`);
      }
      let fixed = s.lineDiagram.fixed;
      if (e.range === "fixed") {
        fixed = e.fixed ?? s.lineDiagram.fixed;
        if (!fixed || !Number.isFinite(fixed.min) || !Number.isFinite(fixed.max) || !(fixed.min < fixed.max)) {
          throw new RangeError(`setLineDiagramRange: a fixed range needs a finite min below a finite max; got ${JSON.stringify(fixed)}`);
        }
      }
      return { ...s, lineDiagram: { ...s.lineDiagram, range: e.range, fixed } };
    }
    case "lineDiagramStepRange": {
      if (lineDiagramKeysOf(s)?.step !== e.key) return s;
      const prevNotice = s.lineDiagram.notice;
      const keepSwitchNotice =
        s.lineDiagram.stepRange === null &&
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
            }))
          : e.min === null
            ? []
            : s.lineDiagram.drawnBeams;
      return syncLineDiagramDerived({
        ...s,
        lineDiagram: {
          ...s.lineDiagram,
          stepRange: { key: e.key, min: e.min, max: e.max },
          notice: nextNotice,
          drawnBeams,
        },
      });
    }
    case "lineDiagramScan":
      if (lineDiagramKeysOf(s)?.scan !== e.scan.key) return s;
      return { ...s, lineDiagram: { ...s.lineDiagram, scan: e.scan } };
    case "setLoads": {
      if (e.on !== undefined && typeof e.on !== "boolean") {
        throw new RangeError(`setLoads: on ${JSON.stringify(e.on)}`);
      }
      if (e.scale !== undefined && (typeof e.scale !== "number" || !Number.isFinite(e.scale) || e.scale <= 0)) {
        throw new RangeError(`setLoads: the scale must be a finite number above 0; got ${JSON.stringify(e.scale)}`);
      }
      const cur = s.loads;
      let on = e.on ?? cur.on;
      const scale = e.scale ?? cur.scale;
      const showNodalForces = e.showNodalForces ?? cur.showNodalForces;
      const showNodalMoments = e.showNodalMoments ?? cur.showNodalMoments;
      const showElementLoads = e.showElementLoads ?? cur.showElementLoads;
      const showSp = e.showSp ?? cur.showSp;
      const includeSupportReactions = e.includeSupportReactions ?? cur.includeSupportReactions;
      const scaleBySeries = e.scaleBySeries ?? cur.scaleBySeries;
      let selectedPatterns = cur.selectedPatterns;
      if (e.selectedPatterns !== undefined) {
        selectedPatterns = e.selectedPatterns === null ? null : [...new Set(e.selectedPatterns)];
      } else if (e.togglePattern !== undefined) {
        const key = e.togglePattern;
        if (!cur.on || cur.selectedPatterns === null) {
          on = e.on ?? true;
          selectedPatterns = [key];
        } else if (cur.selectedPatterns.includes(key)) {
          const rest = cur.selectedPatterns.filter((p) => p !== key);
          if (rest.length === 0) {
            on = e.on ?? false;
            selectedPatterns = null;
          } else {
            selectedPatterns = rest;
          }
        } else {
          selectedPatterns = [...cur.selectedPatterns, key];
        }
      }
      const samePatterns =
        selectedPatterns === cur.selectedPatterns ||
        (selectedPatterns !== null &&
          cur.selectedPatterns !== null &&
          selectedPatterns.length === cur.selectedPatterns.length &&
          selectedPatterns.every((p, i) => p === cur.selectedPatterns![i]));
      if (
        on === cur.on &&
        scale === cur.scale &&
        showNodalForces === cur.showNodalForces &&
        showNodalMoments === cur.showNodalMoments &&
        showElementLoads === cur.showElementLoads &&
        showSp === cur.showSp &&
        includeSupportReactions === cur.includeSupportReactions &&
        scaleBySeries === cur.scaleBySeries &&
        samePatterns
      ) {
        return s;
      }
      return {
        ...s,
        loads: {
          on,
          scale,
          showNodalForces,
          showNodalMoments,
          showElementLoads,
          showSp,
          includeSupportReactions,
          scaleBySeries,
          selectedPatterns,
        },
      };
    }
    case "openSpotlight": {
      const mode = e.mode ?? "root";
      const query = e.query ?? s.spotlight.query;
      if (s.spotlight.open && s.spotlight.mode === mode && s.spotlight.query === query) return s;
      return { ...s, spotlight: { ...s.spotlight, open: true, mode, query } };
    }
    case "closeSpotlight":
      return !s.spotlight.open && s.spotlight.mode === "root" && s.spotlight.query === "" && s.spotlight.clauses.length === 0
        ? s
        : { ...s, spotlight: { ...s.spotlight, open: false, mode: "root", query: "", clauses: [] } };
    case "setSpotlightMode": {
      const query = e.query ?? "";
      if (s.spotlight.mode === e.mode && s.spotlight.query === query) return s;
      return { ...s, spotlight: { ...s.spotlight, mode: e.mode, query } };
    }
    case "setSpotlightQuery":
      return s.spotlight.query === e.query ? s : { ...s, spotlight: { ...s.spotlight, query: e.query } };
    case "addSpotlightClause": {
      const seq = s.spotlight.seq + 1;
      const clause = { ...e.clause, id: `sp-${seq}` };
      return { ...s, spotlight: { ...s.spotlight, clauses: [...s.spotlight.clauses, clause], seq } };
    }
    case "removeSpotlightClause": {
      const clauses = s.spotlight.clauses.filter((c) => c.id !== e.id);
      return clauses.length === s.spotlight.clauses.length ? s : { ...s, spotlight: { ...s.spotlight, clauses } };
    }
    case "clearSpotlightClauses":
      return s.spotlight.clauses.length === 0 ? s : { ...s, spotlight: { ...s.spotlight, clauses: [] } };
    case "setSearchText":
      return s.search.text === e.text ? s : { ...s, search: { ...s.search, text: e.text } };
    case "addSearchClause": {
      const seq = s.search.seq + 1;
      const clause = { ...e.clause, id: `q-${seq}` };
      return { ...s, search: { ...s.search, clauses: [...s.search.clauses, clause], seq } };
    }
    case "removeSearchClause": {
      const clauses = s.search.clauses.filter((c) => c.id !== e.id);
      return clauses.length === s.search.clauses.length ? s : { ...s, search: { ...s.search, clauses } };
    }
    case "clearSearch":
      return s.search.text === "" && s.search.clauses.length === 0 ? s : { ...s, search: { ...s.search, text: "", clauses: [] } };
    default: {
      const unknown: never = e;
      throw new Error(`reduce: unknown event ${JSON.stringify(unknown)}`);
    }
  }
}
