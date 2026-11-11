// The workspace slice (ADR 0132 D1-D7): the set's `<stem>.workspace.h5`, what
// the user makes in the window: named node and element sets, plot and view
// definitions. One slice of the store's types; src/state/types.ts re-exports
// it, and the panels import it from there. A slice file imports sibling
// files, never the barrel.
//
// Everything here is plain data. Main reads and writes the file
// (main/workspace.ts over reader/workspace.ts) and sends these shapes; the
// page never names the file's path (owner ruling option A, #2753): `file` is
// shown, never sent back.

/** The two kinds of member a set holds (a set holds one). */
export const SET_KINDS = ["nodes", "elements"] as const;
export type SetKind = (typeof SET_KINDS)[number];

/** A set's verdict against the paired model (h5-schema.md "The verdict"). */
export const SET_VERDICTS = ["current", "moved", "partial", "lost"] as const;
export type SetVerdictName = (typeof SET_VERDICTS)[number];

/** One set judged against the model main paired it with (Python's `SetVerdict`). */
export interface SetVerdictIn {
  verdict: SetVerdictName;
  matched: number;
  total: number;
  /** per member, in order, whether it matched; all true for `current` (no check ran) */
  members: readonly boolean[];
  /** whether the per-member check ran (only when an identity hash differs) */
  checked: boolean;
  /** the largest distance of a matched member, the figure a human confirming a `moved` set sees; null when nothing was measured */
  maxDistance: number | null;
}

/** One standing row of `/sets`, with its verdict (null when the verdict refused it: see `refused`). */
export interface WorkspaceSetIn {
  row: number;
  name: string;
  kind: SetKind;
  /** the members' OpenSees tags, in the user's order */
  ids: readonly number[];
  verdict: SetVerdictIn | null;
}

/** One standing row of `/plots` or `/views`: stored and read back; drawing a plot is a later slice (VU9b). */
export interface DefRowIn {
  table: "plots" | "views";
  row: number;
  name: string;
  grammar: string;
  /** the row's JSON text, as stored */
  body: string;
}

/** A row that refuses alone, with the schema's reason (`empty-name`, `duplicate`, `bad-members`, `no-tag-map`, ...). */
export interface RowRefusalIn {
  table: "sets" | "plots" | "views";
  row: number;
  name: string;
  reason: string;
  detail: string;
}

/**
 * Main's read of the open set's workspace (`workspace:read`). `file` is the
 * path main chose, shown and never sent back.
 *
 * - `none`: no set is open, or the set has no model to pair a workspace with
 * - `absent`: no file yet; the first save creates it
 * - `unreadable`: not HDF5, no version key, a bad version or a bad `/meta`;
 *   a save renames it aside first (D6), and the banner names the aside file
 * - `refused`: a file this app neither reads nor writes (another major, below
 *   the floor, a missing table, or no model beside it)
 * - `ready`: the tables, each set's verdict, and the rows that refuse alone.
 *   `readOnly` is the banner of a same-major newer file (ADR 0113 D7): it is
 *   shown and never written over (D2).
 *
 * `token` names the file's bytes as main read them (`absent` with no file).
 * A save sends it back as `expect`, and main refuses the save (`stale`) when
 * the window set's file is no longer those bytes: another set opened since
 * the read, or another writer saved. It is not a path.
 */
export type WorkspaceIn =
  | { status: "none"; reason: string }
  | { status: "absent"; file: string; token: string }
  | { status: "unreadable"; file: string; reason: string; detail: string; token: string }
  | { status: "refused"; file: string; reason: string; detail: string }
  | {
      status: "ready";
      file: string;
      token: string;
      version: string;
      revision: number;
      writer: string | null;
      saveId: string | null;
      readOnly: string | null;
      sets: readonly WorkspaceSetIn[];
      plots: readonly DefRowIn[];
      views: readonly DefRowIn[];
      refused: readonly RowRefusalIn[];
      /** tables whose columns are malformed, with the reason; a save refuses while one is broken */
      broken: readonly { table: string; reason: string }[];
    };

/**
 * What the page asks main to write (`workspace:save`), and all it may say:
 * never a path. A `create` names the members by their OpenSees tags; main
 * takes their coordinates and the identity hashes from the model itself.
 */
export type WorkspaceEdit = (
  | { op: "create"; name: string; kind: SetKind; ids: readonly number[] }
  | { op: "rename"; table: "sets" | "plots" | "views"; from: string; to: string }
  | { op: "delete"; table: "sets" | "plots" | "views"; name: string }
  /** a `partial` set keeps its matching members, in order, with the model's hashes */
  | { op: "accept"; name: string }
) & {
  /** the `token` of the read the edit was made on: main refuses the save when the file is no longer those bytes */
  expect: string;
};

/**
 * Main's answer to a save. A refusal replaced nothing: the old file is whole.
 * `held` is the typed refusal of a replace another program blocks (HG9 rule
 * 2): it names the file and asks the user to close it there, and the app never
 * retries on its own.
 */
export type WorkspaceSaveIn =
  | { ok: true; read: WorkspaceIn; aside: string | null }
  | { ok: false; reason: string; detail: string; file: string | null; held: boolean };

/** What the page asks for: a `create` takes its members from the selection when the effects carry it. */
export type WorkspacePageEdit =
  | { op: "create"; name: string }
  | { op: "rename"; table: "sets" | "plots" | "views"; from: string; to: string }
  | { op: "delete"; table: "sets" | "plots" | "views"; name: string }
  | { op: "accept"; name: string };

export interface WorkspaceState {
  /** main's last read; null before the first, and with no window set */
  read: WorkspaceIn | null;
  /** a read in flight */
  reading: boolean;
  /** what a new set takes from the selection: the clicked elements, or the node nearest each click */
  kind: SetKind;
  /**
   * The members a save would take from the selection now, in that kind, as
   * the effects compute them (the node table is in the BlobStore): the tags, or
   * why there are none. The panel lists them, so what is saved is what is shown
   * (the viewport highlights the clicked cells, also in Nodes). Null before the
   * first computation.
   */
  preview: { kind: SetKind; ids: readonly number[]; notes: readonly string[] } | { kind: SetKind; reason: string } | null;
  /** the latest save asked for (the effects act on a new `seq`); `seq` counts for the session */
  seq: number;
  /** `expect` is the token of the read on screen when the save was asked */
  request: { seq: number; edit: WorkspacePageEdit; expect: string } | null;
  /** the seq of a save in flight, or null */
  saving: number | null;
  /** the last save's refusal, for the panel; null after a save that landed */
  notice: { reason: string; detail: string; file: string | null; held: boolean } | null;
  /** D6, OQ5: the file an unreadable workspace was renamed aside to, once per rename, until dismissed */
  aside: string | null;
  /** `@save_id` of this window's last save: a watcher report of that save is recognised, not re-read as news */
  ownSaveId: string | null;
}
