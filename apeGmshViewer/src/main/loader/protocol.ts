// The messages between main and the loader process (ADR 0137 D2, #3058).
//
// Main asks for one read and the loader answers it. Each request carries an id, and its answer
// carries the same id. A history scan also sends progress events while it runs. Both directions
// cross a structured clone (Electron's `utilityProcess` port, or Node's
// `serialization: "advanced"` in the door tests), so the answers are plain data and typed arrays,
// as the IPC to the page already requires.

import type { ModelFile } from "../../model/types.ts";
import type {
  HingeHistoryAnswer,
  HingeHistoryRequest,
  NodeHistoryAnswer,
  NodeHistoryRequest,
  StepRangeAnswer,
  StepRangeRequest,
  ParticleIndexAnswer,
  ParticleOutputAnswer,
  ParticleOutputRequest,
  ParticleSetAnswer,
  ResultsOpenAnswer,
  ScanAnswer,
  ScanProgress,
  ScanRequest,
  StepAnswer,
  StepRequest,
} from "../results.ts";
import type { GeometryAnswer, LineageAnswer, ProvenanceAnswer, StageElementsAnswer } from "../zones.ts";

/** What `model:open` hands the page beside the model: its zones, read in the same visit to the file. */
export interface ModelOpened {
  model: ModelFile;
  provenance: ProvenanceAnswer;
  lineage: LineageAnswer;
  stageElements: StageElementsAnswer;
}

/** Each one-file read the loader makes, by its op, with the value a success answers. */
export interface LoaderOps {
  /** The model and its /provenance, /meta/lineage and /sequence element tags (`model:open`). */
  model: ModelOpened;
  /** The model's node and element counts, and nothing else of it (`compare:model`). */
  modelCounts: { nodes: number; elements: number };
  /** A /geometry sibling (`geometry:open`). A reader error is an `ok: false` answer, not a failure. */
  geometry: GeometryAnswer;
  /** A model's /provenance (`source:goto`). A reader error is an `ok: false` answer, not a failure. */
  provenance: ProvenanceAnswer;
  /** The model's cell count from the shape of each `/elements/<alias>/ids`, before the model is read (the memory warning, #3057). */
  cells: number;
  /** A `<stem>.particles.h5` (`particles:set`). A reader error is an `ok: false` answer. */
  particleSet: ParticleSetAnswer;
}

export type LoaderOp = keyof LoaderOps;

/** The two results hosts the loader keeps: the window set's and the compare set's (ADR 0132 D9.5). */
export type ResultsHostName = "window" | "compare";

/** A `ResultsHost` call the loader makes on one of its hosts, with its arguments and its answer. */
export interface ResultsCalls {
  open: { args: [path: string]; answer: ResultsOpenAnswer };
  readStep: { args: [req: StepRequest]; answer: StepAnswer };
  readHingeHistory: { args: [req: HingeHistoryRequest]; answer: HingeHistoryAnswer };
  readNodeHistory: { args: [req: NodeHistoryRequest]; answer: NodeHistoryAnswer };
  stepRange: { args: [req: StepRangeRequest]; answer: StepRangeAnswer };
  stepCounts: { args: []; answer: { hits: number; reads: number } };
  scan: { args: [req: ScanRequest, opts: ScanOptions]; answer: ScanAnswer };
  cancelScan: { args: [scanId: number]; answer: void };
  particleIndex: { args: [path: string]; answer: ParticleIndexAnswer };
  readParticleOutput: { args: [req: ParticleOutputRequest]; answer: ParticleOutputAnswer };
  close: { args: []; answer: void };
}

export type ResultsCall = keyof ResultsCalls;

/** `ResultsHost.scan`'s pacing (capture mode reads one step per chunk, 2 s apart). */
export interface ScanOptions {
  chunkSteps?: number;
  paceMs?: number;
  progressMs?: number;
}

/** A call on a results host: a union keyed by `call`, so each call's arguments are typed (finding 9 of the 2026-11-24 review). */
export type ResultsRequest = { [K in ResultsCall]: { id: number; op: "results"; host: ResultsHostName; call: K; args: ResultsCalls[K]["args"] } }[ResultsCall];

export type LoaderRequest = { id: number; op: LoaderOp; path: string } | ResultsRequest;

/** A failure is the reader's message, which main hands the page unchanged (a refusal names its zone). */
export type LoaderReply = { id: number; ok: true; value: unknown } | { id: number; ok: false; error: string };

/** A running scan's progress, sent between its chunks. */
export interface ScanProgressEvent {
  event: "scanProgress";
  host: ResultsHostName;
  progress: ScanProgress;
}

export type LoaderMessage = LoaderReply | ScanProgressEvent;
