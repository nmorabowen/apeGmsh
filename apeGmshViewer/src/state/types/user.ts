// The user slice (ADR 0130): the `.user.h5` as main sends it (the `*In` shapes) and as the store holds it.
// One slice of the store's types; src/state/types.ts re-exports it, and the
// panels import it from there. A slice file imports sibling files, never the barrel.

import type { Vec3 } from "./camera.ts";

/**
 * ADR 0130: what `user:read` answers, as the renderer takes it. A structural
 * subset of reader/userFile.ts `UserRead` (state may not import the reader:
 * reader/read.ts imports state/results.ts, a cycle); test/state/user.test.ts
 * assigns the reader's type to this one, so the two cannot drift apart.
 */
export type UserReadIn = {
  ok: true;
  user: UserIndexIn;
  staleness: UserStalenessIn;
  /** VU9b: what a combination save made on this read sends back as `expect` (main/combinations.ts `combinationToken`: the path main resolved and `/meta@revision`); absent where main gives none (an agent mode) */
  token?: string;
} | { ok: false; error: string };
export type UserZoneIn<T> = { ok: true; value: T } | { ok: false; reason: string };
/** reader/userRules.ts `Combination`: a resolved row, its terms by source address (A3). VU7b-2 evaluates it, so its terms are typed. */
export interface UserCombinationIn {
  name: string;
  rule: string;
  terms: { kind: string; source: string; step: number; mode: number; factor: number }[];
  envelope: string;
  fraction: number | null;
  damping: number | null;
}
export interface UserRefusalIn {
  name: string;
  reason: string;
  detail: string;
  origin: string;
  /** the row's position in its table */
  row: number;
}
export interface UserResultEntryIn {
  name: string;
  target: string;
  basis: string;
  perState: boolean;
  linear: boolean;
  /** VU16: its columns in `DATA` order (`@COMPONENTS`); [] for an elements entry */
  components: string[];
}
export interface UserIndexIn {
  path: string;
  /** every root member but `/meta`, by name */
  zones: string[];
  warnings: string[];
  /** VU9b: `generation` is the `g<N>` read (`@current`) and `generations` every `g<N>` the zone holds, ascending; both absent from an index built before VU9b, and null / [] with no zone */
  combinations: UserZoneIn<{ rows: UserCombinationIn[]; modelRows: UserCombinationIn[]; refused: UserRefusalIn[]; generation?: number | null; generations?: number[] }>;
  results: UserZoneIn<{ entries: UserResultEntryIn[]; refused: UserRefusalIn[] }>;
  modes: UserZoneIn<UserModeTableIn> | null;
  spectra: UserZoneIn<{ cases: UserSpectrumCaseIn[]; refused: UserRefusalIn[] }> | null;
  /** VU8: `/entities`, or null when the file holds none (absent: an index built before VU8, read as none) */
  entities?: UserZoneIn<UserEntitiesIn> | null;
}
/** reader/userEntities.ts `Entities`: every row in table order (a row's position is its `ENTITY_ROW`), and the refused ones. */
export interface UserEntitiesIn {
  cuts: { name: string; side: string; point: Vec3; normal: Vec3; reference: Vec3; elementIds: number[]; polygon: Vec3[] }[];
  storeys: { name: string; level: number; reference: Vec3; nodeIds: number[] }[];
  pointSets: { name: string; xyz: Vec3[]; hostElement: number[] }[];
  refused: { kind: string; row: number; name: string; reason: string; detail: string }[];
}
/** reader/userModes.ts `ModeTable`. */
export interface UserModeTableIn {
  components: string[];
  rows: {
    host: string;
    kind: string;
    step: number;
    mode: number;
    lambda: number;
    period: number;
    generalizedMass: number;
    participation: number[];
    massRatio: number[];
    cumulativeRatio: number[];
  }[];
}
/** reader/userSpectra.ts `SpectrumCase`, the part the renderer lists. */
export interface UserSpectrumCaseIn {
  name: string;
  direction: number;
  scale: number;
  modes: { mode: number; damping: number }[];
  results: { label: string; target: string; components: string[] }[];
}
export interface StaleMark {
  stale: boolean;
  /** "" when current */
  reason: string;
}
export interface UserStalenessIn {
  zones: Record<string, StaleMark>;
  results: Record<string, StaleMark>;
  combinations: string[];
  /** A6's verdict on each `/modes` row, in table order */
  modes: StaleMark[];
}

/** The `.user.h5` zones the app knows by name (h5-schema.md "`<stem>.user.h5`"); any other root member is named in a warning. */
export const USER_ZONES = ["combinations", "results", "modes", "spectra", "entities"] as const;
export const USER_RESULT_TARGETS = ["nodes", "elements", "entities", "domain"] as const;
export const USER_RESULT_BASES = ["results", "model"] as const;

/** One User Combination (D4), from `.user.h5` or `model.h5` (`origin`). VU7b-2: picked from the phase bar, and evaluated by VU4's evaluator. */
export interface UserCombination {
  name: string;
  rule: string;
  origin: "user" | "model";
  terms: number;
  /** A6: a source of it is stale, directly or through another combination */
  stale: boolean;
}
/**
 * One User Result (D7), judged without its DATA. VU16: a state-free one on the
 * nodes is picked in the Contour card (`ContourSource` `user`) and drawn as a
 * contour; any other is listed only, with why (selectors `userResultPickOf`).
 */
export interface UserResult {
  name: string;
  target: (typeof USER_RESULT_TARGETS)[number];
  basis: (typeof USER_RESULT_BASES)[number];
  perState: boolean;
  linear: boolean;
  /** VU16: its columns in `DATA` order; [] for an elements entry */
  components: string[];
  stale: StaleMark;
}
/** One `/modes` row (D6) and its A6 verdict against the paired `.ladruno`. */
export interface UserModeRow {
  host: string;
  kind: string;
  step: number;
  /** 1-based */
  mode: number;
  lambda: number;
  period: number;
  /** NaN where every participation factor is zero */
  generalizedMass: number;
  /** one value per component of the table, in its order; the ratios are fractions */
  participation: number[];
  massRatio: number[];
  cumulativeRatio: number[];
  /** stale when its lambda differs from the recorded one, or cannot be matched to it */
  verdict: StaleMark;
}
/** One `/spectra` case (D5), listed: no rule is evaluated over it here (VU7b). */
export interface UserSpectrumCase {
  name: string;
  /** 1-based */
  direction: number;
  scale: number;
  /** the case's modes, 1-based */
  modes: number[];
  /** the case's damping when every mode gives the same one, else null (NaN for none) */
  damping: number | null;
  /** its results, `target/label` */
  results: string[];
  /** VU7b-2: its `ON_NODES` results, each with its columns as the file spells them (`DISPLACEMENT`, `Ux`): what a combination over it can colour by */
  nodal: { label: string; components: string[] }[];
}
/** A row the reader refused (A3), with its reason. */
export interface UserRefusal {
  name: string;
  reason: string;
  detail: string;
}
/** A zone's rows as listed, or why the zone was refused. */
export type UserGroup<T> = { ok: true; items: T[]; refused: UserRefusal[] } | { ok: false; reason: string };

/** The `ENTITY_KIND`s of `/entities` (h5-schema.md), one per table. */
export const USER_ENTITY_KINDS = ["cut", "storey", "point_set"] as const;
/** One standing `/entities/cuts` row (D8): the plane, the kept side, its polygon (`[]` for none), the point its moments are about. */
export interface UserCut {
  /** its `ENTITY_ROW` */
  row: number;
  name: string;
  side: "positive" | "negative";
  point: Vec3;
  normal: Vec3;
  reference: Vec3;
  /** OpenSees tags */
  elementIds: number[];
  polygon: Vec3[];
}
/** One standing `/entities/storeys` row (D8): a level, its nodes, its reference point (the centre of mass). */
export interface UserStorey {
  row: number;
  name: string;
  level: number;
  reference: Vec3;
  nodeIds: number[];
}
/** One standing `/entities/point_sets` row (D8): points, each with a host element (an OpenSees tag) or -1. */
export interface UserPointSet {
  row: number;
  name: string;
  xyz: Vec3[];
  hostElement: number[];
}
/** A refused `/entities` row, which keeps its position. */
export interface UserEntityRefusal extends UserRefusal {
  kind: (typeof USER_ENTITY_KINDS)[number];
  row: number;
}

/** The renderer's view of one `.user.h5`, derived from `UserReadIn` by the reducer. */
export interface UserInfo {
  /** the file's banners (a newer zone), and each root member this app does not know */
  warnings: string[];
  /** the User Combinations family: `.user.h5` rows, then `model.h5` rows; null when neither file has one */
  combinations: UserGroup<UserCombination> | null;
  /**
   * VU7b-2: the descriptor rows in force across both files and the refused ones,
   * as VU4's evaluator takes them (selectors `combinationTableOf`); null when
   * `/combinations` is refused or neither file has a row.
   */
  table: { rows: UserCombinationIn[]; modelRows: UserCombinationIn[]; refused: UserRefusalIn[] } | null;
  /**
   * VU9b: the `/combinations` generations, for the authoring panel's revert:
   * `current` is `@current`, `held` every `g<N>` ascending. Null when the zone
   * is refused or the file holds none (a first save creates it).
   */
  generations?: { current: number; held: number[] } | null;
  /** the User Results picker group; null when the file has no `/results` entry */
  results: UserGroup<UserResult> | null;
  /** the Spectra family: its cases and each refused one, or why the zone is refused; null when the file has none */
  spectra: { ok: true; items: UserSpectrumCase[]; refused: UserRefusal[]; stale: StaleMark } | { ok: false; reason: string; stale: StaleMark } | null;
  /** `/modes`, the modal table, each row with its verdict; null when the file has none */
  modes: { ok: true; components: string[]; rows: UserModeRow[]; stale: StaleMark } | { ok: false; reason: string; stale: StaleMark } | null;
  /**
   * VU8: `/entities`, its standing rows by table and each refused one, or why the
   * zone is refused; null (or absent: an info built without it) when the file has
   * none. A stale zone is drawn only on request (A6).
   */
  entities?:
    | { ok: true; cuts: UserCut[]; storeys: UserStorey[]; pointSets: UserPointSet[]; refused: UserEntityRefusal[]; stale: StaleMark }
    | { ok: false; reason: string; stale: StaleMark }
    | null;
}

/**
 * VU7b-2 (ADR 0130 D4): what the app read for the User Combination shown
 * (`PhaseKey` `combination`) and the quantity picked, under `key` (selectors
 * `combinationRequestOf`): being read, the evaluator's value, its refusal
 * (`CombinationRefused`, with its reason), or a failure the evaluator raised
 * (a case that repeats a node id raises in Python too). The values stay in
 * renderer/contour.ts `CombinationColumns`, as a results step's stay in the
 * effects; the state keeps what the legend and the deformed toggle need.
 */
export type CombinationRead =
  | { key: string; status: "loading" }
  | {
      key: string;
      status: "ready";
      /** the field's one label (`DISPLACEMENT/Ux`) */
      label: string;
      /** the finite min and max, null when no value is finite */
      range: { min: number; max: number } | null;
      /** how many nodes it gives a value for */
      nodes: number;
      /** whether it may draw as a deformed shape (ADR 0130 A3) */
      signed: boolean;
    }
  | { key: string; status: "refused"; reason: string; detail: string }
  | { key: string; status: "failed"; error: string };

/**
 * VU16 (ADR 0130 D7): what the app read for the user result picked
 * (`ContourSource` `user`), under `key` (selectors `userResultRequestOf`):
 * being read, its column's range, main's refusal (`stale` carries A6's
 * reason), or what the read raised. The values stay in renderer/contour.ts
 * `userResultHeld`, as a results step's stay in the effects. `read` counts the
 * feed's reads, so a re-read under the same key recolours.
 */
export type UserResultRead =
  | { key: string; status: "loading" }
  | {
      key: string;
      status: "ready";
      read: number;
      /** the finite min and max, null when no value is finite */
      range: { min: number; max: number } | null;
      /** how many node ids the column lists, and how many of them hold a finite value */
      nodes: number;
      finite: number;
    }
  | { key: string; status: "refused"; reason: "refused" | "stale"; detail: string }
  | { key: string; status: "failed"; error: string };

/** ADR 0130: the set's `.user.h5`. Read on request, after the model's first frame (renderer/app.ts). */
export type UserFileState =
  | { status: "reading"; path: string }
  | { status: "failed"; path: string; error: string }
  /** VU9b: `token` is the read's save token (`UserReadIn.token`), null where main gave none, so no save is made on it */
  | { status: "ready"; path: string; info: UserInfo; token?: string | null };

// -- VU9b (ADR 0130 A4, HG9 on #2693): authoring a User Combination in the app ---------------

/** One term of the draft, as the authoring panel holds it: a source address (A3) and a factor. */
export interface DraftTerm {
  kind: string;
  source: string;
  step: number;
  mode: number;
  factor: number;
}
/** The combination the human is writing (A3's row; null `fraction` / `damping` is NaN in the file, "" `envelope` none). */
export interface CombinationDraft {
  name: string;
  rule: string;
  envelope: string;
  fraction: number | null;
  damping: number | null;
  terms: DraftTerm[];
}
/**
 * The check of the draft under `key` (selectors `draftKeyOf`): running; standing,
 * with VU4's evaluator's value for one quantity (`preview`) or, with `preview`
 * null, why the app cannot evaluate it (`note`: a stage or mode term is read
 * from the results file); or refused, with the reader's or the evaluator's
 * reason. Only a standing draft is offered for save.
 */
export type DraftCheck =
  | { key: string; status: "checking" }
  | {
      key: string;
      status: "ok";
      preview: { label: string; nodes: number; range: { min: number; max: number } | null; signed: boolean } | null;
      note: string | null;
    }
  | { key: string; status: "refused"; reason: string; detail: string };
/**
 * What `combinations:save` takes: one edit, never a path (main writes the open
 * set's own `.user.h5`). `add` saves `row` as a new generation, `revert` puts
 * `@current` back to the generation before it (A7.3: one revert). `expect` is
 * the token of the read the edit was made on.
 */
export type CombinationEdit = { op: "add"; row: UserCombinationIn; expect: string } | { op: "revert"; expect: string };
/**
 * Main's answer to a combination save: the file read again after the replace
 * (with its new token), or a refusal that replaced nothing. `held` is the typed
 * refusal of a replace another program blocks (HG9 rule 3).
 */
export type CombinationSaveIn =
  | { ok: true; op: "add" | "revert"; file: string; generation: number; revision: number; read: UserReadIn }
  | { ok: false; reason: string; detail: string; file: string | null; held: boolean };
/**
 * VU9b: the authoring panel's own state (panels/combine.ts holds it; the store
 * holds the file it is made against). New events would need the store's event
 * ledger (state/spotlight.ts), which this slice does not own.
 */
export interface CombineState {
  draft: CombinationDraft;
  check: DraftCheck | null;
  /** the save in flight (its op), or null */
  saving: "add" | "revert" | null;
  /** the last refusal, shown until the next save */
  notice: { reason: string; detail: string; file: string | null; held: boolean } | null;
  /** the last save that landed: what it did, and the generation now in force */
  last: { op: "add" | "revert"; name: string | null; generation: number; file: string } | null;
}
