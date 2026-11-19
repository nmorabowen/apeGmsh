// The variants slice (ADR 0132): a study's variants, the compare set and the difference.
// One slice of the store's types; src/state/types.ts re-exports it, and the
// panels import it from there. A slice file imports sibling files, never the barrel.

import type { UserInfo, UserReadIn, UserResultRead } from "./user.ts";

// -- VU10 (ADR 0132 D8, D9): a study's variants, the compare set, the difference ----

/** One row of the open manifest that passed D8, as main sends it: never its resolved path (the page names a row by index and label). */
export interface VariantRowIn {
  index: number;
  label: string;
  /** as the manifest writes it, relative to its folder: shown, never sent back */
  path: string;
  params: Readonly<Record<string, string | number | boolean>>;
}

/** A refused row of the manifest (reader/variants.ts `RowRefusal`). */
export interface VariantRefusalIn {
  index: number;
  label: string | null;
  code: string;
  reason: string;
}

/** The study main holds (main.ts `studyView`): the manifest the user opened, its rows, the row on screen and main's compare slot. */
export interface VariantsIn {
  manifest: string;
  version: string;
  /** the banner of a same-major newer manifest (read with its unknown keys dropped), or null */
  newer: string | null;
  rows: readonly VariantRowIn[];
  refused: readonly VariantRefusalIn[];
  /** the row whose model is the window set's, or null */
  active: number | null;
  /** the row in main's compare slot, or null */
  compare: { index: number; label: string } | null;
}

/** A stage of the compare set's results file, as the difference needs it: its id, names, nodal datasets and times. */
export interface CompareStageIn {
  id: string;
  /** shown only */
  name: string;
  /** what the difference matches the window's stage by (`ResultsStage.programName`) */
  programName: string;
  nodal: readonly string[];
  time: readonly number[];
}

/** The compare set main opened (`variants:compare`): the row, its results file (shown only) and its lineage. */
export interface CompareIn {
  index: number;
  label: string;
  results: string;
  /** the compare results' `fem_hash`, or null when it is not stamped */
  femHash: string | null;
  stages: readonly CompareStageIn[];
}

export type CompareState =
  | { index: number; label: string; status: "opening" }
  | ({ status: "open" } & CompareIn)
  | { index: number; label: string; status: "failed"; reason: string };

/** The read of the difference shown, under the key of selectors `differencePlanOf`. */
export type DifferenceRead =
  | { key: string; status: "loading" }
  | { key: string; status: "ready"; range: { min: number; max: number } | null; nodes: number; missing: number }
  | { key: string; status: "refused"; reason: string };

// -- VU19 (ADR 0132 D9.5): the compare set's model and `.user.h5`, through its own doors --

/** The compare slot a compare door read (main's `study.compare`), by row index and label: an answer for another slot is dropped. */
export interface CompareSlotIn {
  index: number;
  label: string;
}

/** The compare set's model as main read it (`compare:model`): its mesh's identity, never its path. */
export interface CompareModelIn {
  femHash: string;
  modelHash: string;
  nodes: number;
  elements: number;
}

/** What `compare:model` answers: the slot it read (null when main holds none) and the model, or why not. */
export type CompareModelAnswer = { ok: true; slot: CompareSlotIn } & CompareModelIn | { ok: false; slot: CompareSlotIn | null; reason: string };

/** What `compare:userRead` answers: the slot it read (null when main holds none), its `.user.h5` (shown only) and `user:read`'s answer for it. */
export interface CompareUserAnswer {
  slot: CompareSlotIn | null;
  path: string | null;
  read: UserReadIn;
}

export type CompareModelState = { status: "reading" } | ({ status: "ready" } & CompareModelIn) | { status: "failed"; reason: string };

/**
 * The compare set's `.user.h5`, each entry with A6's verdict against the
 * compare set's own model and results: being read, absent (the set has none,
 * which is no failure), refused with why, or read.
 */
export type CompareUserState =
  | { status: "reading" }
  | { status: "none"; path: string }
  | { status: "failed"; path: string | null; error: string }
  | { status: "ready"; path: string; info: UserInfo };

/** The compare user result drawn as the contour: one component of a state-free `ON_NODES` entry of the compare set's file. */
export interface CompareUserPick {
  name: string;
  component: string;
}

export interface VariantsState {
  /** the study main holds, or null with no manifest open */
  study: VariantsIn | null;
  /** a switch asked for and not yet answered */
  switching: { index: number; label: string } | null;
  /** why the last switch or compare was refused, for the panel; null otherwise */
  notice: string | null;
  /** the compare set (read-only, never the window set) */
  compare: CompareState | null;
  /** whether the contour shows the window variant minus the compare variant */
  difference: boolean;
  read: DifferenceRead | null;
  /** VU19: the open compare set's model (`compare:model`), null with no compare set open */
  compareModel: CompareModelState | null;
  /** VU19: the open compare set's `.user.h5` (`compare:userRead`), null with no compare set open */
  compareUser: CompareUserState | null;
  /** VU19: the compare user result drawn as the contour, or null; never with the difference on */
  shown: CompareUserPick | null;
  /** VU19: its read, under selectors `compareUserRequestOf`'s key */
  shownRead: UserResultRead | null;
}
