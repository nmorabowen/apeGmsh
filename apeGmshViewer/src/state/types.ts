// The state and event types of the one store (ADR 0112 D6; the reconciled
// V0 design on #1283, decisions 15-17).
//
// Everything in `State` is plain data: records, arrays, strings, numbers,
// booleans and null. Large arrays live in the BlobStore (blobs.ts), outside
// the snapshot, and the state holds only a `BlobRef` to each. The
// reducer-purity test (test/store.test.ts) walks every state the reducer
// returns and fails on a TypedArray, an ArrayBuffer, a Map or a Set.
//
// This file only re-exports. Each slice's types live in types/<slice>.ts, and
// `State` and the `Event` union in types/state.ts and types/event.ts, so a
// change to one slice edits that slice's file. Importers keep this path: the
// panels and ui/ may import state/types.ts and nothing under types/. A slice
// file imports its siblings, never this file (the lint refuses the cycle).
// types/events/<slice>.ts (each slice's part of `Event`) and types/visibility.ts
// are not re-exported here.

// Panels import their chain types from here, never from src/chain (the
// panel allow-list: state/store, state/selectors, state/types, ui/).
export type { Chain, ChainNode, Field } from "../chain/resolve.ts";
// The model-definition data, as the reader read it (plain data: no typed array).
export type { AnalysisDef, DeclParams, DeclRecord, DeclValue, Definitions, DefObject, NeutralLoads, OpsFamily, ProgramDecls, SectionProfile, SequenceZone, StageDef, Table } from "../model/types.ts";

export * from "./types/model.ts";
export * from "./types/results.ts";
export * from "./types/user.ts";
export * from "./types/variants.ts";
export * from "./types/contour.ts";
export * from "./types/deform.ts";
export * from "./types/clip.ts";
export * from "./types/lineDiagram.ts";
export * from "./types/overlays.ts";
export * from "./types/windows.ts";
export * from "./types/offers.ts";
export * from "./types/camera.ts";
export * from "./types/state.ts";
export * from "./types/event.ts";
