// Every side effect of the renderer (D6, decision 15): opening a file
// through the main-process bridge, the drop target, the open and
// file-changed notifications V2f adds to `window.viewer`, go-to-source, and
// releasing blobs the state no longer holds. Nothing here renders, and
// nothing outside this module calls the bridge for a file.
//
// V2f (src/main, preload.ts) adds `onOpen`, `onFileChanged` and
// `goToSource` to the bridge; each is feature-detected here, so this module
// works before and after that PR.

import type { ModelFile } from "./model/types.ts";
import type { GeometryZone } from "./reader/geometry.ts";
import type { LadrunoLineStationStep } from "./reader/ladruno.ts";
import type { ProvenanceZone } from "./reader/provenance.ts";
import { GEOMETRY_TARGET, NEUTRAL_TARGET, OPENSEES_TARGET, PROVENANCE_TARGET, ZONE_FLOOR } from "./reader/read.ts";
import type { BlobStore } from "./state/blobs.ts";
import { loadGeometry } from "./state/geometry.ts";
import { loadModel } from "./state/load.ts";
import { columnOf, finiteRange, finiteRangeOf, gaussCellValues, gaussFields, isNodeTopology, joinElements } from "./state/contour.ts";
import { applyPlan, cellLookup, extrapolationVerdict, planFor, planRange, type ExtrapPlan } from "./state/extrapolation.ts";
import { missingDrawnNodes, spatialNdm, vertexDisplacements, type VertexDisplacement } from "./state/deform.ts";
import { buildLineDiagram3D, joinLadrunoElements } from "./state/lineDiagram.ts";
import { contourView, deformView, elementJoinKey, lineDiagramKeysOf, resultsPairing, sourceFor, summarizeDrawnBeams } from "./state/selectors.ts";
import type { Store } from "./state/store.ts";
import { focusOf, type Focus } from "./renderer/selection.ts";
import { elementMaskOf, visibleMaps } from "./renderer/visible.ts";
import type { ArtifactKind, BlobRef, ContourScan, ElementJoinVerdict, Lineage, LineDiagramBeamSummary, LineStationComponent, ResultsInfo, StageElements, State, ZoneStatus } from "./state/types.ts";

const TARGET = { neutral: NEUTRAL_TARGET, opensees: OPENSEES_TARGET, geometry: GEOMETRY_TARGET, provenance: PROVENANCE_TARGET } as const;

/** What the frame effect moves: the viewport (or a test double). `null` frames the whole model. */
export interface FrameTarget {
  frameTo(focus: Focus | null): void;
}

export interface OpenSet {
  model: string | null;
  geometry: string | null;
  results: string | null;
}

/** `window.viewer`, as preload.ts exposes it; the optional members arrive with V2f. */
export interface Bridge {
  config(): Promise<{
    mode: "view" | "measure" | "capture";
    file: string | null;
    t0: number;
    appReadyMs: number;
    configMs: number;
    pick: string | null;
    /** V4b, capture mode: the results sibling of the captured model, or null */
    results?: string | null;
  }>;
  /** V2f: a successful answer also carries the model file's /provenance (absent from older builds); V4b its /meta/lineage and /sequence element tags. */
  openModel(path: string): Promise<
    | {
        ok: true;
        model: ModelFile;
        provenance?: { ok: true; zone: ProvenanceZone | null } | { ok: false; error: string };
        lineage?: { ok: true; lineage: Lineage | null } | { ok: false; error: string };
        stageElements?: { ok: true; sets: Record<string, number[]> } | { ok: false; error: string };
      }
    | { ok: false; error: string }
  >;
  /** V4b: read a results.h5's index in main (src/main/results.ts); V4c1 adds each stage's nodal `_ids`. */
  openResults?(path: string): Promise<
    | {
        ok: true;
        meta: ResultsInfo;
        nodeIds?: Record<string, Float64Array>;
        /** V4c2a: each gauss group's `_element_index` by stage and group id, a verified file only */
        elementIds?: Record<string, Record<string, Float64Array>>;
        warnings: string[];
        sizeBytes: number;
        readMs: number;
      }
    | { ok: false; error: string }
  >;
  /** V4c1: the history range of a contour field (main/results.ts `ResultsHost.scan`); the reply echoes `scanId`. V4c2a: `source: "element"`. */
  scanResults?(req: { path: string; stage: string; components: string[]; magnitude: boolean; scanId: number; source?: "element" | "line"; plan?: ExtrapPlan }): Promise<
    | { ok: true; scanId: number; min: number | null; max: number | null; steps: number; ms: number }
    | { ok: false; scanId: number; error: string; cancelled?: true; changed?: true }
  >;
  /** V4c1: stop scan `scanId` (it answers `cancelled`). */
  cancelResultsScan?(scanId: number): Promise<unknown>;
  /** V4c1, measure mode: the machine's CPU load (%) since the last call, and the main and renderer working sets (MB). */
  measureCpu?(): Promise<number>;
  measureMemory?(): Promise<{ main: number; renderer: number }>;
  /** V4c1: the progress of the running scans; returns the unsubscribe. */
  onScanProgress?(cb: (p: { scanId: number; done: number; total: number }) => void): (() => void) | void;
  /** V4b: one step of the open results file; the reply echoes `seq`. V4c2a: `element` reads one Gauss component too (`gauss`, by group). */
  readResultsStep?(req: { path: string; stage: string; step: number; seq: number; element?: string; lineDiagram?: LineStationComponent }): Promise<
    | { ok: true; seq: number; stage: string; step: number; nodal: Record<string, Float64Array>; gauss?: Record<string, Float64Array>; lineDiagram?: LadrunoLineStationStep | null; lineDiagramError?: string }
    | { ok: false; seq: number; error: string; changed?: true }
  >;
  /** V2f: read a /geometry sibling (null: the file has no such zone). */
  openGeometry?(path: string): Promise<{ ok: true; geometry: GeometryZone | null; sizeBytes: number; readMs: number } | { ok: false; error: string }>;
  pathForFile(f: File): string;
  metrics(): Promise<{ mainMB: number; rendererMB: number; gpuProcessMB: number; gpuDevices: unknown[] }>;
  measureDone(result: unknown): Promise<void>;
  captureStill(suffix: string): Promise<string | null>;
  captureDone(): Promise<void>;
  /** V4b, capture mode: resize the page for a still at another window width. */
  captureResize?(width: number, height: number): Promise<void>;
  fail(message: string): Promise<void>;
  /** V2f: replays the set already open at subscription, then every later open; the page must not also load `config().file`. Returns the unsubscribe. */
  onOpen?(cb: (set: OpenSet) => void): (() => void) | void;
  onFileChanged?(cb: (path: string) => void): (() => void) | void;
  /** V2f: ask main to open a dropped file; main answers through `onOpen` */
  requestOpen?(path: string): Promise<unknown>;
  goToSource?(file: string, line: number): Promise<{ ok: boolean; reason?: string }>;
}

/**
 * A reader refusal, from the message the reader throws (`<zone>_schema_version
 * <version>: <reason>`): the zone, its version, the range this app accepts
 * (the zone's floor through the current major, from the reader's one floor
 * table; a later minor is read with a banner, so it is not a refusal) and
 * whether the file is ahead of this app (a newer major) or below the floor.
 * Any other message is a plain load failure. (#1303 owns the reader's policy;
 * a structured refusal across the bridge would retire this parse.)
 */
export function parseRefusal(error: string): { zone: string; version: string; accepted: string; newer: boolean; reason: string } | null {
  const m = /^(neutral|opensees|geometry|provenance)_schema_version (\d+)\.(\d+)\.(\d+): (.+)$/.exec(error);
  if (!m) return null;
  // `/sequence` is not here: its refusal is a zone status (`SequenceZone`), never a thrown message.
  const zone = m[1] as keyof typeof TARGET;
  const major = Number(m[2]);
  const t = TARGET[zone];
  const accepted = `${t.major}.${ZONE_FLOOR[zone]} and any later ${t.major}.x`;
  return { zone, version: `${major}.${m[3]}.${m[4]}`, accepted, newer: major > t.major, reason: m[5]! };
}

/** Every BlobRef the state holds (what the BlobStore keeps after a load). */
export function blobRefsOf(s: State): BlobRef[] {
  const out: BlobRef[] = [];
  if (s.mesh) {
    const m = s.mesh;
    out.push(m.linePositions, m.triPositions, m.edgePositions, m.lineElement, m.triElement, m.lineGroup, m.triGroup);
  }
  for (const d of Object.values(s.decls)) if (d.group) out.push(d.group.ids);
  for (const b of s.blocks) out.push(b.connectivity);
  if (s.geometry) out.push(s.geometry.curvePositions, s.geometry.surfacePositions, s.geometry.pointPositions);
  // V4c1: the node tag of every drawn vertex, and each results stage's nodal `_ids`.
  if (s.mesh) out.push(s.mesh.vertexNodes.line, s.mesh.vertexNodes.tri, s.mesh.vertexNodes.edge);
  for (const st of s.results?.stages ?? []) if (st.nodeIds) out.push(st.nodeIds);
  // V4c2a: each gauss group's element ids (a verified file only).
  for (const st of s.results?.stages ?? []) for (const g of st.gauss) if (g.elementIds) out.push(g.elementIds);
  // V4e3: every node, interior ones too (the cut face's points).
  if (s.mesh?.nodes) out.push(s.mesh.nodes.ids, s.mesh.nodes.xyz);
  return out;
}

/**
 * V4e3: a model load with its node table in the mesh (`MeshInfo.nodes`): the
 * cut face of a solid meets interior nodes no drawn face has, so the caps
 * need every node's position, not the drawn vertices'.
 */
export function withNodes<L extends ReturnType<typeof loadModel>>(load: L, model: ModelFile, blobs: BlobStore): L {
  const n = model.nodeIds.length;
  if (model.nodeCoords.length !== 3 * n) throw new RangeError(`model: ${model.nodeCoords.length} coordinates for ${n} nodes`);
  const nodes = { ids: blobs.put("model/nodes/ids", model.nodeIds), xyz: blobs.put("model/nodes/xyz", model.nodeCoords, [n, 3]) };
  return { ...load, mesh: { ...load.mesh, nodes } };
}


/**
 * A /provenance read error as the model artifact's zone status; the model
 * loads either way. A version refusal is `refused` ("written by an older /
 * newer apeGmsh"); any other error (a bad sha256, a backslash path, a broken
 * table) is `malformed`, with the reader's message, which names the path.
 */
export function provenanceStatus(a: { ok: false; error: string }): ZoneStatus {
  const r = parseRefusal(a.error);
  if (r && r.zone === "provenance") return { status: "refused", version: r.version, accepted: r.accepted, newer: r.newer, reason: r.reason };
  return { status: "malformed", reason: a.error };
}

/**
 * The V4b parts of an `openModel` answer as `loadModel` takes them: a lineage
 * or element-tag read that failed is a loud warning, and the model loads
 * without it (results then do not pair, or activation is not applied, and
 * the banner says why).
 */
export function modelExtras(res: {
  lineage?: { ok: true; lineage: Lineage | null } | { ok: false; error: string };
  stageElements?: { ok: true; sets: Record<string, number[]> } | { ok: false; error: string };
}): { lineage: Lineage | null; stageElements: StageElements; warnings: string[] } {
  const warnings: string[] = [];
  const l = res.lineage;
  if (l && !l.ok) warnings.push(`/meta/lineage could not be read (results cannot pair with this model): ${l.error}`);
  const e = res.stageElements;
  const stageElements: StageElements = !e ? { status: "none" } : e.ok ? { status: "ready", sets: e.sets } : { status: "error", reason: e.error };
  return { lineage: l?.ok ? l.lineage : null, stageElements, warnings };
}

/**
 * V4d: the displacement of every drawn-or-not vertex at the step shown,
 * through the vertex node ids (state/deform.ts), and the key it is for
 * (selectors `deformView(...).keys.disp`). Kept here, outside the store.
 */
export interface DisplacementData {
  key: string;
  line: VertexDisplacement;
  tri: VertexDisplacement;
  /** the outline edges (2 vertices each), through their own node ids */
  edge: VertexDisplacement;
}

/** The current results step's data: kept here, outside the store, and only the latest one (D5). */
export interface StepData {
  stage: number;
  step: number;
  nodal: Record<string, Float64Array>;
  /** V4c2a: the element component fetched with the step, or null; and its values by gauss group, (E * G,) element-major */
  element: string | null;
  gauss: Record<string, Float64Array>;
  /** Line diagram component fetched with the step, or null */
  lineDiagram?: LadrunoLineStationStep | null;
  /** Refusal message when the step's line-station read failed while nodal/gauss succeeded */
  lineDiagramError?: string | null;
}

export class Effects {
  private readonly store: Store;
  private readonly blobs: BlobStore;
  private readonly bridge: Bridge;
  /** The latest open wins: a read that finishes after a newer open started is dropped. */
  private openToken = 0;
  private reading = false;
  /** Failed re-reads of the current path after a mid-read change; reset by a successful read or a new path. */
  private retries = 0;
  private retryPath: string | null = null;
  private frameTarget: FrameTarget | null = null;
  private readonly off: (() => void)[] = [];

  /**
   * How long a rewrite of the model or of its geometry waits for the other
   * sibling's rewrite before both are re-read (a D1 re-run rewrites both;
   * each run mints a new session_id, so reading one alone would pair a new
   * file with an old one and flash the stale-geometry notice).
   */
  private readonly setWindowMs: number;
  private setTimer: ReturnType<typeof setTimeout> | null = null;

  constructor(store: Store, blobs: BlobStore, bridge: Bridge, opts: { setWindowMs?: number } = {}) {
    this.store = store;
    this.blobs = blobs;
    this.bridge = bridge;
    this.setWindowMs = opts.setWindowMs ?? 1000;
    // A file rewritten on disk (D1 rewrites on every run) is read again; one
    // that changes mid-read is re-read when that read ends (see `open`).
    this.off.push(
      this.store.subscribe((s) => {
        const m = s.artifacts.model;
        if (!m || !m.stale || this.reading) return;
        if (Effects.isSet(s)) return this.scheduleSetReload();
        // A change after the retries ran out starts a fresh count.
        if (m.status === "failed" && m.path === this.retryPath && this.retries >= Effects.MAX_RETRIES) this.retries = 0;
        void this.open(m.path);
      }),
      // The geometry sibling is re-read the same way (no retries: a failed
      // read shows, and the next change reads it again).
      this.store.subscribe((s) => {
        const g = s.artifacts.geometry;
        if (!g || !g.stale || this.readingGeometry) return;
        if (Effects.isSet(s)) return this.scheduleSetReload();
        void this.openGeometry(g.path);
      }),
      // Go-to-source: a panel dispatches `requestSource`; this answers it.
      this.store.subscribe((s) => {
        const r = s.source.request;
        if (!r || r.seq === this.sourceSeq) return;
        this.sourceSeq = r.seq;
        void this.jumpTo(r.decl);
      }),
    );
    // V4b: a results file rewritten on disk is read again (the pairing is a
    // selector, so the re-read re-pairs); the current results step is
    // fetched whenever the phase names a step that is not the one fetched.
    this.off.push(
      this.store.subscribe((s) => {
        const r = s.artifacts.results;
        if (!r || !r.stale || this.readingResults) return;
        void this.openResults(r.path);
      }),
      this.store.subscribe((s) => this.syncStep(s)),
      // V4c1: the contour's step range and history scan follow what is shown.
      this.store.subscribe((s) => this.syncContour(s)),
      // V4c2a: the picked element field's rows joined to the mesh.
      this.store.subscribe((s) => this.syncJoin(s)),
      // V4d: the displacement of the step shown, and the count of drawn nodes without one.
      this.store.subscribe((s) => this.syncDeform(s)),
      // 3-D line diagram step range and history scan.
      this.store.subscribe((s) => this.syncLineDiagram(s)),
    );
    if (typeof this.bridge.onScanProgress === "function") {
      const off = this.bridge.onScanProgress((p) => {
        const run = this.scanRun;
        if (run && p.scanId === run.id) {
          this.store.dispatch({ type: "contourScan", scan: { key: run.key, status: "running", done: p.done, total: p.total, min: null, max: null, error: null } });
        }
        const lineRun = this.lineScanRun;
        if (lineRun && p.scanId === lineRun.id) {
          this.store.dispatch({ type: "lineDiagramScan", scan: { key: lineRun.key, status: "running", done: p.done, total: p.total, min: null, max: null, error: null } });
        }
      });
      if (typeof off === "function") this.off.push(off);
    }
    // frameSelection: the camera frames the selection's bounds (from the
    // mesh blobs), or the whole model when nothing is selected.
    this.off.push(
      this.store.subscribe((s, prev) => {
        if (s.view.frameSeq === prev.view.frameSeq) return;
        if (!this.frameTarget) throw new Error("frameSelection: no frame target is attached");
        this.frameTarget.frameTo(focusOf(s, this.blobs));
      }),
    );
  }

  /** Attach what `frameSelection` moves (the viewport). */
  setFrameTarget(target: FrameTarget | null): void {
    this.frameTarget = target;
  }

  private readingGeometry = false;
  private geometryToken = 0;
  private sourceSeq = 0;

  // -- V4b: results ----------------------------------------------------------

  private readingResults = false;
  private resultsToken = 0;
  /** The request number of the latest step fetch; a reply with another number is dropped (latest wins). */
  private stepSeq = 0;
  /** What the latest fetch asked for: `<path>|<stage id>|<step>`. */
  private stepKey: string | null = null;
  private step: StepData | null = null;

  /** The current step's data, or null; never a step other than the one the store shows. */
  currentStep(): StepData | null {
    return this.step;
  }

  // -- V4c1: the contour's step range and history scan ------------------------

  /** The step data the last step range was measured on, and its key (a re-fetch or another pick measures again). */
  private rangeFor: { key: string; data: StepData } | null = null;
  /** The running history scan: its id (main echoes it) and key. */
  private scanRun: { id: number; key: string } | null = null;
  private scanSeq = 0;
  /** Finished scans of the file read now, by key (cleared when it is read again). */
  private readonly scanCache = new Map<string, { min: number | null; max: number | null; total: number }>();

  /** Stop the running scan, in main too; its late answer is dropped. */
  private stopScan(): void {
    const run = this.scanRun;
    if (!run) return;
    this.scanRun = null;
    if (typeof this.bridge.cancelResultsScan === "function") void this.bridge.cancelResultsScan(run.id).catch(() => undefined);
  }

  // -- 3-D line diagram step range and history scan ---------------------------

  private lineRangeFor: { key: string; data: StepData } | null = null;
  private lineScanRun: { id: number; key: string } | null = null;
  private readonly lineScanCache = new Map<string, { min: number | null; max: number | null; total: number }>();

  /** The current line diagram step data and its key, or null. */
  currentLineDiagram(): { key: string; step: LadrunoLineStationStep } | null {
    const s = this.store.get();
    const ldKeys = lineDiagramKeysOf(s);
    const at = s.phase.at;
    const data = this.step;
    if (
      !ldKeys ||
      at?.kind !== "results" ||
      !data ||
      data.stage !== at.stage ||
      data.step !== at.step ||
      !data.lineDiagram ||
      data.lineDiagram.component !== s.lineDiagram.component
    ) {
      return null;
    }
    return { key: ldKeys.step, step: data.lineDiagram };
  }

  private stopLineScan(): void {
    const run = this.lineScanRun;
    if (!run) return;
    this.lineScanRun = null;
    if (typeof this.bridge.cancelResultsScan === "function") void this.bridge.cancelResultsScan(run.id).catch(() => undefined);
  }

  private syncLineDiagram(s: State): void {
    const ldKeys = lineDiagramKeysOf(s);
    if (!s.lineDiagram.on || !ldKeys || s.lineDiagram.component === null) {
      if (this.lineScanRun) this.stopLineScan();
      return;
    }
    const at = s.phase.at;
    const data = this.step;
    if (
      at?.kind === "results" &&
      data &&
      data.stage === at.stage &&
      data.step === at.step &&
      s.resultStep?.status === "ready"
    ) {
      if (data.lineDiagramError) {
        const rangeContextKey = `${ldKeys.step}|error|${data.lineDiagramError}`;
        const stepMeasured = s.lineDiagram.stepRange?.key === ldKeys.step && this.lineRangeFor?.key === rangeContextKey;
        if (!stepMeasured && !(this.lineRangeFor && this.lineRangeFor.key === rangeContextKey && this.lineRangeFor.data === data)) {
          this.lineRangeFor = { key: rangeContextKey, data };
          const key = ldKeys.step;
          const notice = data.lineDiagramError;
          queueMicrotask(() =>
            this.store.dispatch({ type: "lineDiagramStepRange", key, min: null, max: null, notice, drawnBeams: [] }),
          );
        }
      } else if (data.lineDiagram && data.lineDiagram.component === s.lineDiagram.component) {
        const dv = deformView(s);
        const deformActive = s.deform.on && (dv.state === "ready" || dv.state === "partial" || dv.state === "loading");
        const deformCtx = s.deform.on ? `def:${dv.state}:${s.deform.scale}:${this.disp?.key ?? "none"}` : "undef";
        const rangeContextKey = `${ldKeys.step}|${s.lineDiagram.plane}|${deformCtx}|${s.visibility.hidden.join(",")}|${s.mesh?.linePositions.id ?? "nomesh"}`;
        const stepMeasured = s.lineDiagram.stepRange?.key === ldKeys.step && this.lineRangeFor?.key === rangeContextKey;
        if (!stepMeasured && !(this.lineRangeFor && this.lineRangeFor.key === rangeContextKey && this.lineRangeFor.data === data)) {
          this.lineRangeFor = { key: rangeContextKey, data };
          let r: { min: number; max: number } | null;
          let notice: string | null = null;
          let drawnBeams: LineDiagramBeamSummary[] = [];
          if (s.mesh && s.results && resultsPairing(s).state === "paired") {
            const join = joinLadrunoElements(s.decls, resultsPairing(s), s.results.elementIndexSpace);
            if (!join.ok) {
              r = null;
              notice = join.reason;
            } else {
              const basePos = this.blobs.f32(s.mesh.linePositions);
              let drawnLinePositions = basePos;
              let deformApplied = false;
              if (deformActive && this.disp && this.disp.key === dv.keys.disp && this.disp.line.d.length === basePos.length) {
                drawnLinePositions = new Float32Array(basePos.length);
                const scale = s.deform.scale;
                const rawLine = this.disp.line.d;
                for (let i = 0; i < basePos.length; i++) {
                  drawnLinePositions[i] = basePos[i]! + scale * rawLine[i]!;
                }
                deformApplied = true;
              }
              if (s.deform.on && !deformApplied) {
                r = null;
                notice =
                  dv.state === "unavailable" || dv.state === "refused"
                    ? `Deformed line diagram hidden: displacement data is unavailable or does not match the drawn 1-D mesh (${dv.reason}).`
                    : "Deformed line diagram hidden: displacement data is unavailable or does not match the drawn 1-D mesh.";
              } else {
                const visibleIndices = visibleMaps(
                  s.mesh.legend,
                  s.visibility.hidden,
                  this.blobs.i32(s.mesh.lineGroup),
                  this.blobs.i32(s.mesh.triGroup),
                  elementMaskOf(s, s.mesh, this.blobs),
                ).lineMap;
                const lineNodes = this.blobs.get(s.mesh.vertexNodes.line) as Float64Array;
                const geom = buildLineDiagram3D({
                  decls: s.decls,
                  meshElements: s.mesh.elements,
                  lineElement: this.blobs.i32(s.mesh.lineElement),
                  drawnLinePositions,
                  lineNodes,
                  visibleLineMap: visibleIndices,
                  stepData: data.lineDiagram,
                  join,
                  ndm: s.ndm.ndm,
                  charLength: 0.8 * s.mesh.radius,
                  deformed: deformApplied,
                  scale: s.lineDiagram.scale,
                  flipSign: s.lineDiagram.flipSign,
                  cmap: s.lineDiagram.cmap,
                  plane: s.lineDiagram.plane,
                  cameraViewDir: [0, 0, -1],
                });
                r = geom.beams.length > 0 ? finiteRangeOf(geom.beams.map((b) => b.values)) : null;
                notice = geom.notice;
                drawnBeams = summarizeDrawnBeams(geom, lineNodes);
              }
            }
          } else {
            r = finiteRange([data.lineDiagram.values], false);
            if (s.mesh && s.results) {
              const p = resultsPairing(s);
              if (p.state === "mismatch" || p.state === "unpairable") notice = p.reason;
            } else if (!s.mesh) {
              notice = "No model mesh is loaded to draw 1-D beam diagrams.";
            }
          }
          const key = ldKeys.step;
          queueMicrotask(() =>
            this.store.dispatch({
              type: "lineDiagramStepRange",
              key,
              min: r?.min ?? null,
              max: r?.max ?? null,
              notice,
              drawnBeams,
            }),
          );
        }
      }
    }
    const want = s.lineDiagram.range === "history" ? ldKeys.scan : null;
    if (want === (this.lineScanRun?.key ?? null)) return;
    if (this.lineScanRun) this.stopLineScan();
    if (want === null) return;
    if (s.lineDiagram.scan?.key === want && s.lineDiagram.scan.status !== "running") return;
    const cached = this.lineScanCache.get(want);
    if (cached) {
      queueMicrotask(() =>
        this.store.dispatch({
          type: "lineDiagramScan",
          scan: { key: want, status: "done", done: cached.total, total: cached.total, min: cached.min, max: cached.max, error: null },
        }),
      );
      return;
    }
    this.startLineScan(s, want, s.lineDiagram.component);
  }

  private startLineScan(s: State, key: string, component: LineStationComponent): void {
    const info = s.results!;
    const at = s.phase.at as { kind: "results"; stage: number };
    const st = info.stages[at.stage]!;
    const id = ++this.scanSeq;
    this.lineScanRun = { id, key };
    const total = st.time.length;
    const report = (scan: ContourScan) =>
      this.store.dispatch({ type: "lineDiagramScan", scan });
    queueMicrotask(() => {
      if (this.lineScanRun?.id !== id) return;
      report({ key, status: "running", done: 0, total, min: null, max: null, error: null });
      if (typeof this.bridge.scanResults !== "function") {
        this.lineScanRun = null;
        report({ key, status: "failed", done: 0, total, min: null, max: null, error: "this build has no history scan" });
        return;
      }
      void this.bridge
        .scanResults({ path: info.path, stage: st.id, components: [component], magnitude: false, scanId: id, source: "line" })
        .catch((err: unknown) => ({ ok: false as const, scanId: id, error: err instanceof Error ? err.message : String(err) }))
        .then((res) => {
          if (this.lineScanRun?.id !== id || res.scanId !== id) return;
          this.lineScanRun = null;
          if (res.ok) {
            this.lineScanCache.set(key, { min: res.min, max: res.max, total: res.steps });
            report({ key, status: "done", done: res.steps, total: res.steps, min: res.min, max: res.max, error: null });
            return;
          }
          if ("cancelled" in res && res.cancelled) return;
          report({ key, status: "failed", done: 0, total, min: null, max: null, error: res.error });
          if ("changed" in res && res.changed) this.store.dispatch({ type: "fileChanged", path: info.path });
        });
    });
  }

  /**
   * Measure the step range of the pick on screen once its step data is here,
   * and run the history scan the range mode needs: one scan at a time, for
   * the key on screen; a new field, component, stage or file cancels it.
   */
  private syncContour(s: State): void {
    const v = contourView(s);
    if (v.state !== "ready" && v.state !== "element") {
      if (this.scanRun) this.stopScan();
      return;
    }
    const at = s.phase.at;
    const data = this.step;
    if (at?.kind === "results" && data && data.stage === at.stage && data.step === at.step && s.resultStep?.status === "ready") {
      // V4c2a: an element field is measured on the step fetched with its component (a fetch without it is replaced).
      const fetched = v.state === "ready" || data.element === v.component.key;
      if (fetched && !v.stepMeasured && !(this.rangeFor && this.rangeFor.key === v.keys.step && this.rangeFor.data === data)) {
        this.rangeFor = { key: v.keys.step, data };
        let r: { min: number; max: number } | null;
        if (v.state === "element" && isNodeTopology(v.topology)) {
          // V4c2b: a node-located topology is measured on the values it draws (extrapolation overshoots the Gauss points).
          const plan = planFor({ ...v, topology: v.topology }, s, this.blobs);
          r = planRange(
            applyPlan(
              plan,
              v.groups.map((g) => {
                const vals = data.gauss[g.id];
                if (!(vals instanceof Float64Array)) throw new Error(`element contour: the step data has no ${v.component.key} for group ${g.id}`);
                return vals;
              }),
            ),
          );
        } else if (v.state === "element") {
          r = finiteRangeOf(
            v.groups.map((g) => {
              const vals = data.gauss[g.id];
              if (!(vals instanceof Float64Array)) throw new Error(`element contour: the step data has no ${v.component.key} for group ${g.id}`);
              return gaussCellValues(vals, g.nElements, g.nGauss);
            }),
          );
        } else {
          const rows = v.component.datasets.map((d) => data.nodal[d]);
          r = rows.every((x): x is Float64Array => x instanceof Float64Array) ? finiteRange(rows, v.component.magnitude) : null;
        }
        const key = v.keys.step;
        queueMicrotask(() => this.store.dispatch({ type: "contourStepRange", key, min: r?.min ?? null, max: r?.max ?? null }));
      }
    }
    const want = s.contour.range === "history" ? v.keys.scan : null;
    if (want === (this.scanRun?.key ?? null)) return;
    if (this.scanRun) this.stopScan();
    if (want === null) return;
    // A scan already in the store for this key (done or failed) is not run again.
    if (s.contour.scan?.key === want && s.contour.scan.status !== "running") return;
    const cached = this.scanCache.get(want);
    if (cached) {
      queueMicrotask(() =>
        this.store.dispatch({ type: "contourScan", scan: { key: want, status: "done", done: cached.total, total: cached.total, min: cached.min, max: cached.max, error: null } }),
      );
      return;
    }
    // V4c2b: a node-located topology's history scan carries the plan, so main measures the values drawn.
    const plan = v.state === "element" && isNodeTopology(v.topology) ? planFor({ ...v, topology: v.topology }, s, this.blobs) : undefined;
    this.startScan(s, want, v.component.datasets, v.component.magnitude, v.state === "element", plan);
  }

  /** The join last checked (its key); a re-read, another field or mesh checks again. */
  private joinedFor: string | null = null;

  /**
   * V4c2a: join the picked element field's rows to the model's cells, by FEM
   * id (state/contour.ts `joinElements`), and report the verdict. The rows
   * exist only for a verified file (main sends them for no other), so an
   * unverified file is never joined.
   */
  private syncJoin(s: State): void {
    const key = elementJoinKey(s);
    if (key === null || s.contour.join?.key === key || this.joinedFor === key) return;
    const at = s.phase.at as { kind: "results"; stage: number };
    const st = s.results!.stages[at.stage]!;
    const f = gaussFields(st.gauss).find((x) => x.name === s.contour.field);
    if (!f) return;
    const groups: { id: string; eids: Float64Array; nElements: number; nGauss: number; naturalCoords: number[][] | null }[] = [];
    for (const gid of f.groups) {
      const g = st.gauss.find((x) => x.id === gid);
      const ref = g?.elementIds;
      if (!g || !ref) return;
      const ids = this.blobs.get(ref);
      if (!(ids instanceof Float64Array)) throw new TypeError(`element contour: ${ref.id} is not f64`);
      groups.push({ id: gid, eids: ids, nElements: g.nElements, nGauss: g.nGauss, naturalCoords: g.naturalCoords });
    }
    this.joinedFor = key;
    const j = joinElements(s.decls, groups);
    // V4c2b: whether the joined rows can be carried to the nodes (their element types and natural coordinates).
    const verdict: ElementJoinVerdict = j.ok
      ? { key, ok: true, rows: j.rows, extrapolation: extrapolationVerdict(groups, cellLookup(s.decls, s.blocks, this.blobs)) }
      : { key, ok: false, code: j.code, reason: j.reason };
    queueMicrotask(() => this.store.dispatch({ type: "contourElementJoin", verdict }));
  }

  private startScan(s: State, key: string, components: string[], magnitude: boolean, element = false, plan?: ExtrapPlan): void {
    const info = s.results!;
    const at = s.phase.at as { kind: "results"; stage: number };
    const st = info.stages[at.stage]!;
    const id = ++this.scanSeq;
    this.scanRun = { id, key };
    const total = st.time.length;
    const report = (scan: ContourScan) =>
      this.store.dispatch({ type: "contourScan", scan });
    queueMicrotask(() => {
      if (this.scanRun?.id !== id) return;
      report({ key, status: "running", done: 0, total, min: null, max: null, error: null });
      if (typeof this.bridge.scanResults !== "function") {
        this.scanRun = null;
        report({ key, status: "failed", done: 0, total, min: null, max: null, error: "this build has no history scan" });
        return;
      }
      void this.bridge
        .scanResults({ path: info.path, stage: st.id, components, magnitude, scanId: id, ...(element ? { source: "element" as const } : {}), ...(plan ? { plan } : {}) })
        .catch((err: unknown) => ({ ok: false as const, scanId: id, error: err instanceof Error ? err.message : String(err) }))
        .then((res) => {
          // A cancelled or replaced scan never lands.
          if (this.scanRun?.id !== id || res.scanId !== id) return;
          this.scanRun = null;
          if (res.ok) {
            this.scanCache.set(key, { min: res.min, max: res.max, total: res.steps });
            report({ key, status: "done", done: res.steps, total: res.steps, min: res.min, max: res.max, error: null });
            return;
          }
          if ("cancelled" in res && res.cancelled) return;
          report({ key, status: "failed", done: 0, total, min: null, max: null, error: res.error });
          if ("changed" in res && res.changed) this.store.dispatch({ type: "fileChanged", path: info.path });
        });
    });
  }

  // -- V4d: the deformed shape --------------------------------------------------

  /** The displacement last computed, and the count key last counted (each recomputed only when its key changes). */
  private disp: DisplacementData | null = null;
  private countedFor: string | null = null;

  /** The displacement of the step shown (null before it is computed); the viewport checks its key against the state's. */
  currentDisplacement(): DisplacementData | null {
    return this.disp;
  }

  /**
   * Once the step shown is here: its displacement at every vertex (when the
   * displacement key changed: another step, file read, mesh or axes), and the
   * count of drawn nodes without one (when the count key changed: also the
   * hidden set). The keys carry the results-read count, so a re-read of the
   * same path is computed again, never reused.
   */
  private syncDeform(s: State): void {
    const v = deformView(s);
    if (v.state !== "loading" && v.state !== "ready" && v.state !== "partial") return;
    const at = s.phase.at;
    const data = this.step;
    if (at?.kind !== "results" || !data || data.stage !== at.stage || data.step !== at.step || s.resultStep?.status !== "ready") return;
    const mesh = s.mesh!;
    if (this.disp?.key !== v.keys.disp) {
      const [dx, dy, dz] = v.datasets;
      const row = (name: string): Float64Array => {
        const r = data.nodal[name];
        if (!(r instanceof Float64Array)) throw new Error(`deformed shape: the step data has no ${name}`);
        return r;
      };
      const rows = [row(dx), row(dy), dz === null ? null : row(dz)] as const;
      const ids = this.blobs.get(v.nodeIds);
      if (!(ids instanceof Float64Array)) throw new TypeError(`deformed shape: ${v.nodeIds.id} is not f64`);
      const column = columnOf(ids);
      const lineNodes = this.blobs.get(mesh.vertexNodes.line) as Float64Array;
      const triNodes = this.blobs.get(mesh.vertexNodes.tri) as Float64Array;
      const edgeNodes = this.blobs.get(mesh.vertexNodes.edge) as Float64Array;
      this.disp = {
        key: v.keys.disp,
        line: vertexDisplacements(lineNodes, column, rows),
        tri: vertexDisplacements(triNodes, column, rows),
        edge: vertexDisplacements(edgeNodes, column, rows),
      };
    }
    if (s.deform.measured?.key === v.keys.count || this.countedFor === v.keys.count) return;
    this.countedFor = v.keys.count;
    const { lineMap, triMap } = visibleMaps(mesh.legend, s.visibility.hidden, this.blobs.i32(mesh.lineGroup), this.blobs.i32(mesh.triGroup), elementMaskOf(s, mesh, this.blobs));
    const n = missingDrawnNodes(
      this.blobs.get(mesh.vertexNodes.line) as Float64Array,
      this.blobs.get(mesh.vertexNodes.tri) as Float64Array,
      this.disp.line.missing,
      this.disp.tri.missing,
      lineMap,
      triMap,
    );
    const key = v.keys.count;
    queueMicrotask(() => this.store.dispatch({ type: "deformMeasured", key, drawn: n.drawn, missing: n.missing }));
  }

  /** Read a results file's index into the store; the latest open wins. */
  async openResults(path: string): Promise<boolean> {
    const token = ++this.resultsToken;
    this.store.dispatch({ type: "fileOpened", artifact: "results", path });
    if (typeof this.bridge.openResults !== "function") {
      this.store.dispatch({ type: "fileFailed", artifact: "results", path, error: "this build has no results reader" });
      return false;
    }
    this.readingResults = true;
    // A re-read invalidates the step on screen: its data may have changed.
    this.step = null;
    this.stepKey = null;
    // V4c1 seam "reload": the scan of the old file stops, and its ranges are forgotten.
    this.stopScan();
    this.scanCache.clear();
    this.rangeFor = null;
    this.stopLineScan();
    this.lineScanCache.clear();
    this.lineRangeFor = null;
    try {
      const res = await this.bridge.openResults(path).catch((err: unknown) => ({ ok: false as const, error: err instanceof Error ? err.message : String(err) }));
      if (token !== this.resultsToken) return false;
      if (!res.ok) {
        this.store.dispatch({ type: "fileFailed", artifact: "results", path, error: res.error });
        return false;
      }
      for (const w of res.warnings) console.warn(w);
      // V4c1: each stage's nodal `_ids` into the BlobStore; the store holds the ref.
      // V4c2a: each gauss group's element ids too (main sends them for a verified file only).
      const stages = res.meta.stages.map((st) => {
        const ids = res.nodeIds?.[st.id];
        const eids = res.elementIds?.[st.id];
        const gauss = st.gauss.map((g) => {
          const e = eids?.[g.id];
          return e ? { ...g, elementIds: this.blobs.put(`results/${st.id}/gauss/${g.id}/elementIds`, e) } : g;
        });
        return ids ? { ...st, gauss, nodeIds: this.blobs.put(`results/${st.id}/nodeIds`, ids) } : { ...st, gauss };
      });
      this.store.dispatch({ type: "resultsLoaded", info: { ...res.meta, stages }, sizeBytes: res.sizeBytes, readMs: res.readMs, warnings: res.warnings });
      this.blobs.retain(blobRefsOf(this.store.get()));
      return true;
    } finally {
      if (token === this.resultsToken) {
        this.readingResults = false;
        // A change during the read: read once more (the writer may still be going).
        const r = this.store.get().artifacts.results;
        if (r && r.path === path && r.stale && r.status === "ready") void this.openResults(path);
      }
    }
  }

  /** Fetch the step the phase names when it is not the one fetched (latest wins). */
  private syncStep(s: State): void {
    const at = s.phase.at;
    const info = s.results;
    if (at?.kind !== "results" || !info || s.artifacts.results?.status !== "ready") {
      if (this.stepKey !== null) {
        this.stepKey = null;
        this.step = null;
        this.stepSeq++;
      }
      return;
    }
    const st = info.stages[at.stage];
    if (!st) return;
    // V4c2a: an element contour on screen fetches its one Gauss component with the step (only once its
    // rows joined the mesh of a verified file: `contourView` is `element` for no other).
    const v = contourView(s);
    const element = v.state === "element" ? v.component.key : null;
    const lineDiagram = s.lineDiagram.on && s.lineDiagram.component !== null && st.lineStations?.includes(s.lineDiagram.component) ? s.lineDiagram.component : null;
    const key = `${info.path}|${st.id}|${at.step}|${element ?? ""}|${lineDiagram ?? ""}`;
    if (key === this.stepKey) return;
    this.stepKey = key;
    const seq = ++this.stepSeq;
    this.step = null;
    // Not from inside this listener: the other listeners have not seen `s` yet.
    queueMicrotask(() => void this.fetchStep(info.path, at.stage, st.id, at.step, seq, element, lineDiagram));
  }

  private async fetchStep(
    path: string,
    stage: number,
    stageId: string,
    step: number,
    seq: number,
    element: string | null = null,
    lineDiagram: LineStationComponent | null = null,
  ): Promise<void> {
    if (seq !== this.stepSeq) return;
    this.store.dispatch({ type: "resultStep", status: { stage, step, status: "loading", error: null } });
    if (typeof this.bridge.readResultsStep !== "function") {
      this.store.dispatch({ type: "resultStep", status: { stage, step, status: "failed", error: "this build has no results reader" } });
      return;
    }
    const res = await this.bridge
      .readResultsStep({
        path,
        stage: stageId,
        step,
        seq,
        ...(element !== null ? { element } : {}),
        ...(lineDiagram !== null ? { lineDiagram } : {}),
      })
      .catch((err: unknown) => ({ ok: false as const, seq, error: err instanceof Error ? err.message : String(err) }));
    // Latest wins: a slow reply for an older request never paints over a newer step.
    if (seq !== this.stepSeq || res.seq !== seq) return;
    if (!res.ok) {
      this.store.dispatch({ type: "resultStep", status: { stage, step, status: "failed", error: res.error } });
      // The file is not the one indexed (main checked): mark it changed, so it is re-indexed and the step fetched again.
      if ("changed" in res && res.changed) this.store.dispatch({ type: "fileChanged", path });
      return;
    }
    this.step = {
      stage,
      step,
      nodal: res.nodal,
      element,
      gauss: res.gauss ?? {},
      lineDiagram: res.lineDiagram ?? null,
      lineDiagramError: res.lineDiagramError ?? null,
    };
    this.store.dispatch({ type: "resultStep", status: { stage, step, status: "ready", error: null } });
  }

  /** A loaded model with a loaded geometry sibling: their rewrites reload as one set. */
  private static isSet(s: State): boolean {
    return s.artifacts.model?.status === "ready" && s.artifacts.geometry?.status === "ready";
  }

  private scheduleSetReload(): void {
    if (this.setTimer !== null) return;
    this.setTimer = setTimeout(() => {
      this.setTimer = null;
      void this.reloadSet();
    }, this.setWindowMs);
  }

  /**
   * Re-read whichever of the model and its geometry went stale in the window,
   * then land them in one `setLoaded`, so the pairing is judged on the new
   * pair only. A failed read is reported as usual and the other still lands.
   */
  private async reloadSet(): Promise<void> {
    const s = this.store.get();
    const mPath = s.artifacts.model?.stale ? s.artifacts.model.path : null;
    const gPath = s.artifacts.geometry?.stale && typeof this.bridge.openGeometry === "function" ? s.artifacts.geometry.path : null;
    if (!mPath && !gPath) return;
    const mt = mPath ? ++this.openToken : -1;
    const gt = gPath ? ++this.geometryToken : -1;
    if (mPath) {
      this.reading = true;
      this.store.dispatch({ type: "fileOpened", artifact: "model", path: mPath });
    }
    if (gPath) {
      this.readingGeometry = true;
      this.store.dispatch({ type: "fileOpened", artifact: "geometry", path: gPath });
    }
    const fail = (artifact: "model" | "geometry", path: string, error: string) => {
      const refusal = parseRefusal(error);
      if (refusal) this.store.dispatch({ type: "zoneRefused", artifact, ...refusal });
      this.store.dispatch({ type: "fileFailed", artifact, path, error });
    };
    const message = (err: unknown) => (err instanceof Error ? err.message : String(err));
    try {
      const [mr, gr] = await Promise.all([
        mPath ? this.bridge.openModel(mPath).catch((err: unknown) => ({ ok: false as const, error: message(err) })) : null,
        gPath ? this.bridge.openGeometry!(gPath).catch((err: unknown) => ({ ok: false as const, error: message(err) })) : null,
      ]);
      let model = null, geometry = null;
      if (mPath && mr && mt === this.openToken) {
        if (!mr.ok) fail("model", mPath, mr.error);
        else {
          const prov = mr.provenance;
          model = {
            ...withNodes(loadModel(mr.model, this.blobs, prov?.ok ? prov.zone : null, prov && !prov.ok ? provenanceStatus(prov) : null, modelExtras(mr)), mr.model, this.blobs),
            ndm: spatialNdm(mr.model.meta, mr.model.neutralVersion),
          };
        }
      }
      if (gPath && gr && gt === this.geometryToken) {
        if (!gr.ok) fail("geometry", gPath, gr.error);
        else if (!gr.geometry) fail("geometry", gPath, "the file has no /geometry zone");
        else geometry = loadGeometry(gr.geometry, this.blobs, gr.sizeBytes, gr.readMs);
      }
      if (model || geometry) {
        this.store.dispatch({ type: "setLoaded", model, geometry });
        this.blobs.retain(blobRefsOf(this.store.get()));
      }
    } finally {
      if (mt === this.openToken) this.reading = false;
      if (gt === this.geometryToken) this.readingGeometry = false;
      // A change that arrived during the reads starts another window.
      const after = this.store.get();
      if (after.artifacts.model?.stale || after.artifacts.geometry?.stale) this.scheduleSetReload();
    }
  }

  /** Answer a `requestSource`: open the editor at the declaration's line, or say why not. */
  private async jumpTo(decl: string): Promise<void> {
    const where = sourceFor(this.store.get(), decl);
    if (!where.ok) {
      this.store.dispatch({ type: "sourceResult", decl, ok: false, reason: where.reason });
      return;
    }
    const res = await this.goToSource(where.source.file, where.source.line);
    this.store.dispatch({ type: "sourceResult", decl, ok: res.ok, reason: res.ok ? null : (res.reason ?? "go-to-source failed") });
  }

  /** Read a geometry sibling into the store; the latest open wins. */
  async openGeometry(path: string): Promise<boolean> {
    const token = ++this.geometryToken;
    this.store.dispatch({ type: "fileOpened", artifact: "geometry", path });
    if (typeof this.bridge.openGeometry !== "function") return false; // a build without the reader: recorded, not read
    this.readingGeometry = true;
    try {
      const res = await this.bridge.openGeometry(path);
      if (token !== this.geometryToken) return false;
      if (!res.ok) {
        const refusal = parseRefusal(res.error);
        if (refusal) this.store.dispatch({ type: "zoneRefused", artifact: "geometry", ...refusal });
        this.store.dispatch({ type: "fileFailed", artifact: "geometry", path, error: res.error });
        return false;
      }
      if (!res.geometry) {
        this.store.dispatch({ type: "fileFailed", artifact: "geometry", path, error: "the file has no /geometry zone" });
        return false;
      }
      const load = loadGeometry(res.geometry, this.blobs, res.sizeBytes, res.readMs);
      for (const w of load.info.warnings) console.warn(w);
      this.store.dispatch({ type: "fileLoaded", artifact: "geometry", load });
      this.blobs.retain(blobRefsOf(this.store.get()));
      return true;
    } catch (err) {
      if (token !== this.geometryToken) return false;
      this.store.dispatch({ type: "fileFailed", artifact: "geometry", path, error: err instanceof Error ? err.message : String(err) });
      return false;
    } finally {
      if (token === this.geometryToken) this.readingGeometry = false;
    }
  }

  /** Open a model artifact; resolves true when it loaded and is still the latest open. */
  async open(path: string): Promise<boolean> {
    const token = ++this.openToken;
    this.reading = true;
    if (this.retryPath !== path) {
      this.retryPath = path;
      this.retries = 0;
    }
    this.store.dispatch({ type: "fileOpened", artifact: "model", path });
    let loaded = false;
    try {
      const res = await this.bridge.openModel(path);
      if (token !== this.openToken) return false;
      if (!res.ok) {
        const refusal = parseRefusal(res.error);
        if (refusal) this.store.dispatch({ type: "zoneRefused", artifact: "model", ...refusal });
        this.store.dispatch({ type: "fileFailed", artifact: "model", path, error: res.error });
        return false;
      }
      const prov = res.provenance;
      // V4d: the model's spatial dimension rides with the load (state/deform.ts `spatialNdm`).
      const load = {
        ...withNodes(loadModel(res.model, this.blobs, prov?.ok ? prov.zone : null, prov && !prov.ok ? provenanceStatus(prov) : null, modelExtras(res)), res.model, this.blobs),
        ndm: spatialNdm(res.model.meta, res.model.neutralVersion),
      };
      for (const w of load.info.warnings) console.warn(w);
      // One line per read, so a double load shows in the Electron log.
      console.info(`apeGmshViewer: loaded ${path} (read ${Math.round(load.info.readMs)} ms)`);
      this.store.dispatch({ type: "fileLoaded", artifact: "model", load });
      this.blobs.retain(blobRefsOf(this.store.get()));
      loaded = true;
      return true;
    } catch (err) {
      if (token !== this.openToken) return false;
      const error = err instanceof Error ? err.message : String(err);
      this.store.dispatch({ type: "fileFailed", artifact: "model", path, error });
      return false;
    } finally {
      if (token === this.openToken) {
        this.reading = false;
        if (loaded) this.retries = 0;
        // A fileChanged that arrived during this read: read once more. After
        // a failed read too (it may have hit a half-written file), but at
        // most MAX_RETRIES times in a row, so a broken file does not loop.
        const m = this.store.get().artifacts.model;
        if (m && m.path === path && m.stale) {
          if (loaded) void this.open(path);
          else if (this.retries < Effects.MAX_RETRIES) {
            this.retries++;
            void this.open(path);
          } else console.warn(`apeGmshViewer: ${path} changed during ${Effects.MAX_RETRIES} failed reads; not read again until it changes`);
        }
      }
    }
  }

  /** Failed re-reads allowed in a row after a mid-read change. */
  static readonly MAX_RETRIES = 3;

  /**
   * The files a V2f `onOpen` names: the model, the geometry and (V4b) the
   * results are read; a kind the set does not name is closed.
   */
  openSet(set: OpenSet): void {
    if (set.geometry) void this.openGeometry(set.geometry);
    else {
      this.geometryToken++;
      this.store.dispatch({ type: "fileClosed", artifact: "geometry" });
    }
    if (set.results) void this.openResults(set.results);
    else {
      this.resultsToken++;
      this.store.dispatch({ type: "fileClosed", artifact: "results" });
    }
    if (set.model) void this.open(set.model);
    else this.store.dispatch({ type: "fileClosed", artifact: "model" });
  }

  /** Detach everything `attach` and the bridge subscriptions registered. */
  dispose(): void {
    if (this.setTimer !== null) clearTimeout(this.setTimer);
    this.setTimer = null;
    for (const f of this.off.splice(0)) f();
  }

  /** Jump to the user code that declared something; without V2f it says so. */
  goToSource(file: string, line: number): Promise<{ ok: boolean; reason?: string }> {
    if (typeof this.bridge.goToSource !== "function") {
      return Promise.resolve({ ok: false, reason: "go-to-source is not available in this build (it arrives with V2f)" });
    }
    return this.bridge.goToSource(file, line);
  }

  /** Whether main feeds the open set through `onOpen` (V2f); then the page never loads `config().file` itself. */
  hasOpenFeed(): boolean {
    return typeof this.bridge.onOpen === "function";
  }

  /** A file the user dropped: main decides what it is when it can (V2f `requestOpen`), else it is read as the model. */
  openDropped(path: string): void {
    if (typeof this.bridge.requestOpen === "function") void this.bridge.requestOpen(path);
    else void this.open(path);
  }

  /** Attach the window and bridge listeners; the returned function detaches them. */
  attach(): () => void {
    const onDragOver = (e: DragEvent) => e.preventDefault();
    const onDrop = (e: DragEvent) => {
      e.preventDefault();
      const f = e.dataTransfer?.files[0];
      if (f) this.openDropped(this.bridge.pathForFile(f));
    };
    window.addEventListener("dragover", onDragOver);
    window.addEventListener("drop", onDrop);
    this.off.push(() => {
      window.removeEventListener("dragover", onDragOver);
      window.removeEventListener("drop", onDrop);
    });
    const keep = (u: (() => void) | void) => {
      if (typeof u === "function") this.off.push(u);
    };
    if (typeof this.bridge.onOpen === "function") keep(this.bridge.onOpen((set) => this.openSet(set)));
    if (typeof this.bridge.onFileChanged === "function") {
      keep(this.bridge.onFileChanged((path) => this.store.dispatch({ type: "fileChanged", path })));
    }
    return () => this.dispose();
  }

  /** The artifact kinds whose path is known (for the header). */
  static siblings(s: State): { kind: ArtifactKind; name: string; status: string }[] {
    const out: { kind: ArtifactKind; name: string; status: string }[] = [];
    for (const k of ["geometry", "results"] as const) {
      const a = s.artifacts[k];
      if (a) out.push({ kind: k, name: a.path.split(/[\\/]/).pop() ?? a.path, status: a.status });
    }
    return out;
  }
}
