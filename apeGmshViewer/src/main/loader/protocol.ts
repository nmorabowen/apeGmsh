// The messages between main and the loader process (ADR 0137 D2, #3058).
//
// Main asks for one read of one file and the loader answers it. Each request carries an id,
// and its answer carries the same id. Both cross a structured clone (Electron's
// `utilityProcess` port, or Node's `serialization: "advanced"` in the door tests), so the
// answers are plain data and typed arrays, as the IPC to the page already requires.

import type { ModelFile } from "../../model/types.ts";
import type { GeometryAnswer, LineageAnswer, ProvenanceAnswer, StageElementsAnswer } from "../zones.ts";

/** What `model:open` hands the page beside the model: its zones, read in the same visit to the file. */
export interface ModelOpened {
  model: ModelFile;
  provenance: ProvenanceAnswer;
  lineage: LineageAnswer;
  stageElements: StageElementsAnswer;
}

/** Each read the loader makes, by its op, with the value a success answers. */
export interface LoaderOps {
  /** The model and its /provenance, /meta/lineage and /sequence element tags (`model:open`). */
  model: ModelOpened;
  /** The model's node and element counts, and nothing else of it (`compare:model`). */
  modelCounts: { nodes: number; elements: number };
  /** A /geometry sibling (`geometry:open`). A reader error is an `ok: false` answer, not a failure. */
  geometry: GeometryAnswer;
  /** A model's /provenance (`source:goto`). A reader error is an `ok: false` answer, not a failure. */
  provenance: ProvenanceAnswer;
}

export type LoaderOp = keyof LoaderOps;

export interface LoaderRequest {
  id: number;
  op: LoaderOp;
  path: string;
}

/** A failure is the reader's message, which main hands the page unchanged (a refusal names its zone). */
export type LoaderReply = { id: number; ok: true; value: unknown } | { id: number; ok: false; error: string };
