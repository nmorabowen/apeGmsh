// The pure reducer: `reduce(state, event)` returns the next state and never
// mutates its arguments (test/store.test.ts freezes the input and checks).
// The event union is exhaustive: the `never` check at the end of the switch
// fails to compile when an event is added without a case, and an event the
// code does not know raises at run time.
//
// A slice whose events write only its own part of the state has its own
// sub-reducer under reduce/, the way the clip events call `reduceClip`: each
// takes its sub-state, and what it reads of the rest as explicit inputs, and
// returns the same object when nothing changed. The model's events and the
// cases that read or write several slices (the files, the selection, the
// phase, the results, the contour and the deformed shape) stay here, as do
// the follow-ups a delegated event needs from another slice (`withPhase`,
// `withLineDiagramPick`, `syncLineDiagramDerived`).

import { reduceClip } from "./clip.ts";
import { CMAP_NAMES, componentName, contourKeysOf, defaultPick, hasPick } from "./contour.ts";
import { defaultLineDiagramComponent, lineDiagramKeysOf } from "./lineDiagram.ts";
import { NODE_SIZE_MAX, NODE_SIZE_MIN } from "./nodes.ts";
import { initialCamera, reduceCamera } from "./reduce/camera.ts";
import { initialState } from "./reduce/initial.ts";
import { pickLineDiagram, reduceLineDiagram } from "./reduce/lineDiagram.ts";
import { reduceOffers } from "./reduce/offers.ts";
import { reduceOverlays } from "./reduce/overlays.ts";
import { reduceUser } from "./reduce/user.ts";
import { reduceVariants } from "./reduce/variants.ts";
import { reduceWindows } from "./reduce/windows.ts";
import { reduceWorkspace, workspaceForAnotherModel } from "./reduce/workspace.ts";
import { elementsOfDecl, selectedElementsOf } from "./search.ts";
import {
  combinationFieldsOf,
  combinationRequestOf,
  deformOf,
  differencePlanOf,
  elementJoinKey,
  geometryPairing,
  resultsPairing,
  stageCount,
  stageIdentity,
  syncLineDiagramDerived,
  userResultPickOf,
  userResultRequestOf,
} from "./selectors.ts";
import { initialSearch, initialSpotlight } from "./spotlight.ts";
import { THEME_MODES } from "../theme/tokens.ts";
import type { ArtifactInfo, ArtifactKind, ContourState, DeclPath, Event, MeshInfo, PhaseKey, State, UnitSystemId } from "./types.ts";
import { UNIT_SYSTEM_IDS } from "./units.ts";

export { canonicalPose, initialCamera, ROLL_TOLERANCE } from "./reduce/camera.ts";
export { emptyRecord, initialContour, initialDeform, initialLineDiagram, initialState } from "./reduce/initial.ts";
export { initialOffers, MAX_OFFERS } from "./reduce/offers.ts";
export { initialLoads, initialPrincipal, initialReactions, initialSprings, initialSupports, initialVectors } from "./reduce/overlays.ts";
export { NO_USER_FILE, userEventOf, userInfoOf, userPathOf, userReadable } from "./reduce/user.ts";
export { initialVariants } from "./reduce/variants.ts";
export { initialWorkspace } from "./reduce/workspace.ts";

type Markers = { on: boolean; size: number | null; auto: boolean };

/** On while the size is automatic asks for it to be chosen again (null); the user's own size stays. */
function markersOn<M extends Markers>(m: M, on: boolean): M {
  if (m.on === on) return m;
  return { ...m, on, size: on && m.auto ? null : m.size };
}

/** A user's size (clamped), the effects' choice (`auto`), or null to choose again. */
function markerSize<M extends Markers>(m: M, size: number | null, auto: boolean | undefined): M {
  if (size === null) return { ...m, size: null, auto: true };
  if (!Number.isFinite(size)) return m;
  return { ...m, size: Math.max(NODE_SIZE_MIN, Math.min(NODE_SIZE_MAX, size)), auto: auto === true };
}

/** The opacity a model opens on: solids translucent when they would hide beams, else opaque. */
export function defaultOpacity(mesh: MeshInfo): number {
  const hasLines = mesh.counts.lineCells > 0 || mesh.counts.opsOnly > 0;
  return mesh.counts.solidCells > 0 && hasLines ? 0.35 : 1;
}

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
  if (a.kind === "combination" && b.kind === "combination") return a.name === b.name;
  return true;
}

/**
 * Why User Combination `name` cannot be shown now, or null when it can: the
 * mesh is loaded and the set's `.user.h5` lists it (a refused row is never
 * shown). While the file is read again the combination stays (`pending`), so a
 * re-read does not drop the state the user picked; its old value is not shown.
 */
function combinationProblem(s: State, name: string): string | null {
  if (!s.mesh) return "no mesh is loaded";
  const u = s.user;
  if (u === null) return "the set has no .user.h5";
  if (u.status === "reading") return null;
  if (u.status === "failed") return `the user file is refused: ${u.error}`;
  const c = u.info.combinations;
  if (c === null) return `${u.path} holds no User Combination`;
  if (!c.ok) return `/combinations is refused: ${c.reason}`;
  if (c.items.some((x) => x.name === name)) return null;
  const r = c.refused.find((x) => x.name === name);
  return r ? `User Combination ${JSON.stringify(name)} is refused: ${r.reason} (${r.detail})` : `the user file lists no User Combination ${JSON.stringify(name)}`;
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
  // VU7b-2: a User Combination is off the axis; it stays shown while the user file still lists it.
  const kept = prev !== null && (prev.kind === "combination" ? combinationProblem(s, prev.name) === null : axis.some((k) => samePhase(k, prev)));
  let at = kept ? prev : (axis.find((k) => k.kind === "mesh") ?? axis[0] ?? null);
  // A results position keeps its step while the stage still has it; a stage
  // that shrank (a re-read) clamps it to its last step.
  if (at?.kind === "results") {
    const n = stepsOf(s, at.stage);
    if (n === 0) at = axis.find((k) => k.kind === "mesh") ?? null;
    else if (at.step >= n) at = { ...at, step: n - 1 };
  }
  const resultStep = at?.kind === "results" && s.resultStep && s.resultStep.stage === at.stage && s.resultStep.step === at.step ? s.resultStep : null;
  // A combination's read belongs to the combination shown, and to nothing after it.
  const combination = at?.kind === "combination" ? s.combination : null;
  return withLineDiagramPick(withContourPick({ ...s, phase: { axis, at }, resultStep, combination }), preserveLineNotice);
}

/** The stage a contour pick is checked against: the results stage shown. */
const shownStage = (s: State) => (s.phase.at?.kind === "results" ? s.results?.stages[s.phase.at.stage] : undefined);

/**
 * Colouring by contour with nothing picked, on a results stage: its first
 * field and component (V4c1). Not after a re-read cleared the pick: the
 * notice stands, and the user picks again (`setContourField` clears it).
 */
function withContourPick(s: State): State {
  if (s.phase.at?.kind === "combination") return withCombinationPick(s, s.phase.at.name);
  if (s.visibility.colourBy !== "contour" || s.contour.field !== null || s.contour.notice !== null) return s;
  const pick = defaultPick(shownStage(s));
  return pick ? { ...s, contour: { ...s.contour, ...pick } } : s;
}

/**
 * VU7b-2: on a User Combination, a pick it offers stays; any other is replaced
 * by its first component (a stage's `DISPLACEMENT/Ux` carries over, because a
 * case names its columns as a `.ladruno` stage does). A combination colours by
 * one nodal component; it never offers a magnitude.
 */
function withCombinationPick(s: State, name: string): State {
  if (s.visibility.colourBy !== "contour") return s;
  const f = combinationFieldsOf(s, name);
  if (!f.ok) return s;
  const c = s.contour;
  if (c.source === "nodal" && f.fields.some((x) => x.name === c.field && x.components.some((y) => y.key === c.component))) return s;
  const first = f.fields[0]!;
  return { ...s, contour: { ...c, source: "nodal", field: first.name, component: first.components[0]!.key, join: null, notice: null } };
}

/**
 * Auto-pick a line-station component when landing on a results stage with
 * `lineStations` and `lineDiagram.component` is null or absent from the stage
 * (reduce/lineDiagram.ts `pickLineDiagram`), then re-derive the chains and the
 * hover from the selection when it asks.
 */
function withLineDiagramPick(s: State, preserveNotice = false): State {
  const { ld, sync } = pickLineDiagram(s.lineDiagram, shownStage(s), () => lineDiagramKeysOf(s), preserveNotice);
  const t = ld === s.lineDiagram ? s : { ...s, lineDiagram: ld };
  return sync ? syncLineDiagramDerived(t) : t;
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
      // VA1: opening another model drops the camera of the one shown; the same path again (a reload, a retry after a failed read) keeps it.
      const anotherModel = e.artifact === "model" && !(prev && prev.path === e.path);
      const camera = anotherModel ? { ...initialCamera, seq: s.camera.seq } : s.camera;
      // VU9a: another model's set has another workspace; the last one's read is never edited against it.
      const workspace = anotherModel ? workspaceForAnotherModel(s.workspace) : s.workspace;
      return withPhase({ ...s, artifacts: { ...s.artifacts, [e.artifact]: info }, results, contour, lineDiagram, camera, workspace });
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
      const opacity = reread ? s.visibility.opacity : defaultOpacity(mesh);
      const extrudeFrames = reread ? s.visibility.extrudeFrames : false;
      const showOffsets = reread ? s.visibility.showOffsets : Boolean(mesh.frameExtrusion?.hasJointOffsets);
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
        visibility: { ...s.visibility, hidden: [], opacity, extrudeFrames, showOffsets },
        // A chosen size belongs to the model it was chosen for; the user's own stays.
        nodes: { ...s.nodes, on: reread ? s.nodes.on : false, size: s.nodes.auto ? null : s.nodes.size },
        gaussPoints: { ...s.gaussPoints, on: reread ? s.gaussPoints.on : false, size: s.gaussPoints.auto ? null : s.gaussPoints.size, measured: null },
        inspector: { pinned: [] },
        spotlight: reread ? s.spotlight : { ...initialSpotlight, seq: s.spotlight.seq },
        search: reread ? s.search : { ...initialSearch, seq: s.search.seq },
        // The request count survives the re-read (see State.source).
        source: { seq: s.source.seq, request: null, last: null },
        // VA1: a read of the path already read keeps the view (a re-read, and a
        // retry after a failed re-read). A path never read opens framed: a pose
        // moved while it loaded, or after its first read failed, is not its own.
        camera: s.camera.read === info.path ? s.camera : { ...initialCamera, seq: s.camera.seq, read: info.path },
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
        // ADR 0130: the .user.h5 is the set's; renderer/app.ts opens and closes it with the set.
        user: s.user,
        contour: syncContourUnits(initialState.contour, s.results?.units ?? null, false),
        spotlight: { ...initialSpotlight, seq: s.spotlight.seq },
        search: { ...initialSearch, seq: s.search.seq },
        source: { ...initialState.source, seq: s.source.seq },
        // V4d: the user's deformed-shape settings stand; the count was for the old model.
        deform: { ...s.deform, measured: null },
        // V4e1: the section planes stand too (the slider re-derives its range from the next mesh).
        clip: s.clip,
        // VA1: the theme is the window's. The camera stands through a failed
        // re-read of the model shown; a failed first read of a path drops a
        // pose moved while it loaded (the model shown then was another one),
        // and the path stays unread (camera.read), so its retry opens framed.
        camera: s.camera.read === e.path ? s.camera : { ...initialCamera, seq: s.camera.seq },
        theme: s.theme,
        // VA5: offers last for the window session (ADR 0128 D5).
        offers: s.offers,
        // VU10: the study and the compare set are the window's, not the model's (a switch loads another model).
        variants: s.variants,
        // VU9a: a failed read of the model leaves no workspace edited against it; a read that lands reads it again.
        workspace: workspaceForAnotherModel(s.workspace),
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
        user: s.user,
        contour: syncContourUnits(initialState.contour, s.results?.units ?? null, false),
        spotlight: { ...initialSpotlight, seq: s.spotlight.seq },
        search: { ...initialSearch, seq: s.search.seq },
        source: { ...initialState.source, seq: s.source.seq },
        // V4d: the user's deformed-shape settings stand; the count was for the old model.
        deform: { ...s.deform, measured: null },
        // V4e1: the section planes stand too (the slider re-derives its range from the next mesh).
        clip: s.clip,
        // VA1: the camera and the theme are the window's, not the model's.
        camera: s.camera,
        theme: s.theme,
        // VA5: offers last for the window session (ADR 0128 D5).
        offers: s.offers,
        // VU10: the study and the compare set are the window's, not the model's (a switch loads another model).
        variants: s.variants,
        // VU9a: the closed set's workspace leaves; the pick kind and the request count are the window's.
        workspace: workspaceForAnotherModel(s.workspace),
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
    case "toggleExtrudeFrames": {
      const next = e.on !== undefined ? Boolean(e.on) : !s.visibility.extrudeFrames;
      return s.visibility.extrudeFrames === next ? s : { ...s, visibility: { ...s.visibility, extrudeFrames: next } };
    }
    case "toggleShowOffsets": {
      const next = e.on !== undefined ? Boolean(e.on) : !s.visibility.showOffsets;
      return s.visibility.showOffsets === next ? s : { ...s, visibility: { ...s.visibility, showOffsets: next } };
    }
    case "setNodes": {
      const nodes = markersOn(s.nodes, e.on);
      return nodes === s.nodes ? s : { ...s, nodes };
    }
    case "setNodeSize": {
      const nodes = markerSize(s.nodes, e.size, e.auto);
      return nodes === s.nodes ? s : { ...s, nodes };
    }
    case "setGaussPoints": {
      const gaussPoints = markersOn(s.gaussPoints, e.on);
      return gaussPoints === s.gaussPoints ? s : { ...s, gaussPoints };
    }
    case "setGaussSize": {
      const gaussPoints = markerSize(s.gaussPoints, e.size, e.auto);
      return gaussPoints === s.gaussPoints ? s : { ...s, gaussPoints };
    }
    case "gaussMeasured": {
      if (!Number.isInteger(e.count) || e.count < 0) throw new RangeError(`gaussMeasured: ${e.count} points`);
      return { ...s, gaussPoints: { ...s.gaussPoints, measured: { stage: e.stage, count: e.count, refused: e.refused, notes: e.notes } } };
    }
    case "setPhase":
      if (e.at.kind === "combination") {
        if (typeof e.at.name !== "string") throw new RangeError(`setPhase: a combination is named by a string; got ${JSON.stringify(e.at)}`);
        const why = combinationProblem(s, e.at.name);
        if (why !== null) throw new Error(`setPhase: User Combination ${JSON.stringify(e.at.name)} cannot be shown: ${why}`);
        // The user file must be read: a pending re-read keeps a combination shown, it does not open one.
        if (s.user?.status !== "ready") throw new Error(`setPhase: User Combination ${JSON.stringify(e.at.name)} cannot be shown: the user file is not read yet`);
      } else if (!s.phase.axis.some((k) => samePhase(k, e.at))) throw new Error(`setPhase: ${JSON.stringify(e.at)} is not on the phase axis`);
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
      // names the same stage (its id and program name; W4d #2843: a display
      // rename re-stamped onto the file is a label, not another stage);
      // otherwise the phase leaves it.
      const at = s.phase.at;
      const old = at?.kind === "results" ? s.results?.stages[at.stage] : undefined;
      const now = at?.kind === "results" ? e.info.stages[at.stage] : undefined;
      const keep = !old || (now !== undefined && now.id === old.id && now.programName === old.programName);
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
      // VU16: a user result is the user file's, not the results'; a re-read of the results keeps it.
      if (c.field === null || c.component === null || c.source === "user") return next;
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
      if (source !== "nodal" && source !== "element" && source !== "user") throw new RangeError(`setContourField: source ${JSON.stringify(source)}`);
      const at = s.phase.at;
      if (source === "user") {
        // VU16: a state-free ON_NODES user result the set's .user.h5 lists, and one of its columns. A stale one is
        // picked, and the contour refuses it with A6's reason (selectors `userResultContour`).
        if (at?.kind === "combination") throw new RangeError(`setContourField: User Combination ${JSON.stringify(at.name)} is shown; a user result is not drawn over it`);
        const p = userResultPickOf(s, e.field);
        if (!p.ok) throw new RangeError(`setContourField: ${p.reason}`);
        if (!p.entry.components.includes(e.component)) {
          throw new RangeError(`setContourField: user result ${JSON.stringify(e.field)} has no component ${JSON.stringify(e.component)}; its components are ${p.entry.components.join(", ")}`);
        }
        const c = s.contour;
        if (c.source === source && c.field === e.field && c.component === e.component && c.notice === null) return s;
        return { ...s, contour: { ...c, source, field: e.field, component: e.component, join: null, notice: null } };
      }
      if (at?.kind === "combination") {
        // VU7b-2: what the combination offers: one nodal component of the cases it reaches.
        const f = combinationFieldsOf(s, at.name);
        if (!f.ok) throw new RangeError(`setContourField: User Combination ${JSON.stringify(at.name)} offers no field: ${f.reason}`);
        if (source !== "nodal" || !f.fields.some((x) => x.name === e.field && x.components.some((y) => y.key === e.component))) {
          throw new RangeError(`setContourField: ${e.field} / ${e.component} is not a ${source} component User Combination ${JSON.stringify(at.name)} offers`);
        }
        const c = s.contour;
        if (c.source === source && c.field === e.field && c.component === e.component && c.notice === null) return s;
        return { ...s, contour: { ...c, source, field: e.field, component: e.component, join: null, notice: null } };
      }
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
      const hasSource = e.source !== undefined || e.sourceUnits !== undefined;
      const rawSource = e.source !== undefined ? e.source : e.sourceUnits;
      if (rawSource !== undefined && rawSource !== null && !UNIT_SYSTEM_IDS.includes(rawSource)) {
        throw new RangeError(`setContourUnits: source ${JSON.stringify(rawSource)}`);
      }
      const targetUnits = e.target ?? e.targetUnits ?? s.contour.targetUnits;
      if (targetUnits !== "as_stored" && !UNIT_SYSTEM_IDS.includes(targetUnits)) {
        throw new RangeError(`setContourUnits: target ${JSON.stringify(targetUnits)}`);
      }
      const sourceUnits = s.contour.declaredUnits ?? (hasSource ? (rawSource ?? null) : s.contour.sourceUnits);
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
    case "setCamera":
    case "cameraMoved": {
      const camera = reduceCamera(s.camera, e, { drawn: s.mesh !== null || s.geometry !== null, nothingSelected: () => selectedElementsOf(s).length === 0 });
      return camera === s.camera ? s : { ...s, camera };
    }
    case "setTheme":
      if (!(THEME_MODES as readonly string[]).includes(e.theme)) throw new RangeError(`setTheme: ${JSON.stringify(e.theme)} is not a theme (${THEME_MODES.join(", ")})`);
      return s.theme === e.theme ? s : { ...s, theme: e.theme };
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
    case "setLineDiagram":
    case "setLineDiagramComponent":
    case "setLineDiagramScale":
    case "setLineDiagramPlane":
    case "setLineDiagramCmap":
    case "setLineDiagramFill":
    case "setLineDiagramFlipSign":
    case "setLineDiagramRange":
    case "lineDiagramStepRange":
    case "lineDiagramScan": {
      const lineDiagram = reduceLineDiagram(s.lineDiagram, e, { stage: () => shownStage(s), keys: () => lineDiagramKeysOf(s) });
      if (lineDiagram === s.lineDiagram) return s;
      const t = { ...s, lineDiagram };
      // Turned on, it picks a component for the stage shown; a new scale, sign or drawn set re-derives the chains and the hover.
      if (e.type === "setLineDiagram") return syncLineDiagramDerived(withLineDiagramPick(t));
      return e.type === "setLineDiagramScale" || e.type === "setLineDiagramFlipSign" || e.type === "lineDiagramStepRange" ? syncLineDiagramDerived(t) : t;
    }
    case "setSupports":
    case "setPrincipal":
    case "setReactions":
    case "setSprings":
    case "setVectors":
    case "setLoads": {
      const cur = { supports: s.supports, principal: s.principal, reactions: s.reactions, springs: s.springs, vectors: s.vectors, loads: s.loads };
      const next = reduceOverlays(cur, e);
      return next === cur ? s : { ...s, ...next };
    }
    case "openWindow":
    case "closeWindow":
    case "openSpotlight":
    case "closeSpotlight":
    case "setSpotlightMode":
    case "setSpotlightQuery":
    case "addSpotlightClause":
    case "removeSpotlightClause":
    case "clearSpotlightClauses":
    case "setSearchText":
    case "addSearchClause":
    case "removeSearchClause":
    case "clearSearch": {
      const cur = { windows: s.windows, spotlight: s.spotlight, search: s.search };
      const next = reduceWindows(cur, e);
      return next === cur ? s : { ...s, ...next };
    }
    case "offerReceived":
    case "offerAssessed":
    case "applyOffer":
    case "backOffer":
    case "offerApplied":
    case "offerRefused":
    case "offerBackDone": {
      // INV-5: an offer event touches the offers and nothing else of the state.
      const offers = reduceOffers(s.offers, e);
      return offers === s.offers ? s : { ...s, offers };
    }
    case "userOpened":
    case "userRead":
    case "userClosed":
    case "combinationRead":
    case "userResultRead": {
      const cur = { user: s.user, combination: s.combination, userResult: s.userResult };
      const next = reduceUser(cur, e, { combinationKey: () => combinationRequestOf(s)?.key, userResultKey: () => userResultRequestOf(s)?.key });
      if (next === cur) return s;
      const t = { ...s, ...next };
      // VU7b-2: another user file state leaves a User Combination shown that the file no longer lists (withPhase); off a combination the phase is not touched.
      const fileChanged = e.type === "userOpened" || e.type === "userRead" || e.type === "userClosed";
      return fileChanged && s.phase.at?.kind === "combination" ? withPhase(t) : t;
    }
    case "variantsRead":
    case "variantSwitch":
    case "variantSwitched":
    case "variantCompare":
    case "compareOpened":
    case "setDifference":
    case "differenceRead": {
      const differenceKey = () => {
        const p = differencePlanOf(s);
        return p !== null && p.ok ? p.key : null;
      };
      const variants = reduceVariants(s.variants, e, { differenceKey });
      return variants === s.variants ? s : { ...s, variants };
    }
    case "workspaceReading":
    case "workspaceRead":
    case "setWorkspaceKind":
    case "workspaceSave":
    case "workspaceSaved":
    case "dismissWorkspaceAside": {
      // VU9a: a workspace event touches the workspace slice and nothing else.
      const workspace = reduceWorkspace(s.workspace, e);
      return workspace === s.workspace ? s : { ...s, workspace };
    }
    default: {
      const unknown: never = e;
      throw new Error(`reduce: unknown event ${JSON.stringify(unknown)}`);
    }
  }
}
