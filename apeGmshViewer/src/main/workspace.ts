// The workspace doors (ADR 0132 D6, D7; HG9 on #2803): `workspace:read` and
// `workspace:save`, the app's one file write.
//
// Under owner ruling option A (#2753) the page never names a path: both doors
// take the window's set from main (`set()`), and the file is that set's
// `<stem>.workspace.h5` beside its model. `workspace:read` takes no argument;
// `workspace:save` takes one edit (`{op, ...}`, state/types/workspace.ts
// `WorkspaceEdit`) with closed keys, so a `path` (or any key it does not name)
// is refused before anything is read. Only the human's window writes (ADR 0128
// INV-1 as ADR 0132 narrowed it): every agent mode reads, none saves.
//
// A save is D6's, with the owner's three build rules (HG9, #2803 comments
// 6770969371 and 6773903480):
//   1. The app never keeps `<stem>.workspace.h5` open. It reads the bytes,
//      parses the file with h5wasm and closes it at once (reader/workspace.ts
//      `readWorkspaceFile`), and the save path holds no handle on it.
//   2. A replace another program blocks (Windows refuses to replace a file a
//      Python session holds open) is a typed refusal, `held`, naming the file
//      and asking the user to close it there: never a raw error, never a retry.
//   3. No rename-aside-then-move-in to get past a holder. The rename aside is
//      only D6's, for a file that is not a readable workspace (as Python's
//      `write_workspace` judges it: a file h5wasm cannot open, a truncated one
//      included, is `not-hdf5`); it is made as a hard link and an unlink, and a
//      holder that blocks the unlink takes the link back and refuses `held`.
// Every edit carries `expect`, the `token` of the read it was made on: the
// sha256 of the workspace path main resolved (case-folded where `samePath`
// folds), a NUL, and the bytes' sha256 or `absent`. It is opaque to the page.
// Main refuses the save, `stale`, when the window set's file is not that path
// with those bytes now: another set opened since the page read its workspace
// (even one whose file is byte-identical, or absent too), or another writer
// saved (#2890 reviews).
// Every refusal replaces nothing: the old file is whole and this writer's
// partial is gone.
//
// No Electron import (main.ts passes `ipcMain` in), so the tests run in Node.

import { createHash, randomBytes } from "node:crypto";
import { closeSync, fsyncSync, linkSync, openSync, readdirSync, readFileSync, renameSync, statSync, unlinkSync } from "node:fs";
import { hostname } from "node:os";
import { basename, dirname, join } from "node:path";
import { inspect } from "node:util";
import { buildWrite, ModelSide, readWorkspaceFile, setsIn, UNREADABLE, verdictsOf, verifyPartial, WorkspaceRefusal, writePartial, WriteRefusal, type Change, type WorkspaceData, type WorkspaceH5 } from "../reader/workspace.ts";
import { SET_KINDS, type WorkspaceEdit, type WorkspaceIn, type WorkspaceSaveIn } from "../state/types/workspace.ts";
import { resolve } from "node:path";
import { candidates, FOLD_CASE, type OpenSet } from "./pairing.ts";
import { heldRefusal } from "../state/workspace.ts";

export { heldRefusal };

/** The disk as a save sees it; the tests pass one that fails where they choose. */
export interface WorkspaceDisk {
  /** the file's bytes, or null when no file is there */
  readFile(path: string): Uint8Array | null;
  isFile(path: string): boolean;
  /** replace `to` with `from` in one rename */
  rename(from: string, to: string): void;
  /** a second name for `from`; refuses (EEXIST) when `to` exists */
  link(from: string, to: string): void;
  unlink(path: string): void;
  fsync(path: string): void;
  list(dir: string): string[];
}

const code = (err: unknown): string | undefined => (err as NodeJS.ErrnoException | null)?.code;

export const nodeDisk: WorkspaceDisk = {
  readFile: (path) => {
    try {
      return readFileSync(path);
    } catch (err) {
      if (code(err) === "ENOENT") return null;
      throw err;
    }
  },
  isFile: (path) => {
    try {
      return statSync(path).isFile();
    } catch {
      return false;
    }
  },
  rename: (from, to) => renameSync(from, to),
  link: (from, to) => linkSync(from, to),
  unlink: (path) => unlinkSync(path),
  // Windows refuses fsync on a read-only handle.
  fsync: (path) => {
    const fd = openSync(path, "r+");
    try {
      fsyncSync(fd);
    } finally {
      closeSync(fd);
    }
  },
  list: (dir) => readdirSync(dir),
};

export interface WorkspaceDeps {
  h5: WorkspaceH5;
  ready: Promise<unknown>;
  disk: WorkspaceDisk;
  /** this machine's name as a partial carries it (`.user.h5`'s rule) */
  host: string;
  pid: number;
  alive(pid: number): boolean;
  /** 8 hex digits for a partial's name */
  token(): string;
  /** a fresh `@save_id`, 8 to 64 of A-Za-z0-9_- */
  saveId(): string;
  now(): Date;
  /** `@writer`, e.g. "apeGmshViewer 0.0.1" */
  writer: string;
  /** a test's hook between the steps of a save (`partial-written`, `verified`, `before-replace`); never set in the app */
  step?(name: string): void;
}

export const UNKNOWN_HOST = "unknownhost";

/** This machine's name, safe in a file name and free of the `-` separator (Python's `_host_token`). */
export function hostToken(name: string = safeHostname()): string {
  const token = name.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "").slice(0, 63);
  return token || UNKNOWN_HOST;
}

function safeHostname(): string {
  try {
    return hostname();
  } catch {
    return "";
  }
}

/** Whether `pid` runs. Unsure reads as alive, so a sweep leaves the file (libuv's signal 0 is a liveness test on Windows too). */
export function pidAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch (err) {
    return code(err) !== "ESRCH";
  }
}

export function nodeDeps(h5: WorkspaceH5, ready: Promise<unknown>, writer: string): WorkspaceDeps {
  return {
    h5,
    ready,
    disk: nodeDisk,
    host: hostToken(),
    pid: process.pid,
    alive: pidAlive,
    token: () => randomBytes(4).toString("hex"),
    saveId: () => randomBytes(16).toString("base64url"),
    now: () => new Date(),
    writer,
  };
}

const shown = (x: unknown): string => {
  try {
    return inspect(x, { depth: 3, breakLength: Infinity, maxArrayLength: 10, maxStringLength: 300 });
  } catch {
    return Object.prototype.toString.call(x);
  }
};

/** The window's set's workspace file and model, as main chose them, or why there is none. */
export function workspaceOf(set: OpenSet | { reason: string }): { file: string; model: string } | { reason: string } {
  if ("reason" in set) return { reason: set.reason };
  if (set.model === null) return { reason: "the open set has no model to pair a workspace with" };
  return { file: set.workspace ?? candidates(set.model).workspace, model: set.model };
}

const digestOf = (bytes: Uint8Array | null): string | null => (bytes === null ? null : createHash("sha256").update(bytes).digest("hex"));
/** The token a read hands the page and a save expects back: the bytes' digest, or `absent`. */
export const ABSENT = "absent";
/** The token of `file` holding bytes of `digest` (null: no file): bound to the path main resolves, so another set's identical file never matches. */
export function tokenOf(file: string, digest: string | null, fold: boolean = FOLD_CASE): string {
  const path = resolve(file);
  return createHash("sha256").update(`${fold ? path.toLowerCase() : path}\0${digest ?? ABSENT}`).digest("hex");
}


const HELD_CODES = new Set(["EPERM", "EACCES", "EBUSY"]);
const isHeld = (err: unknown) => HELD_CODES.has(code(err) ?? "");

/** One read of the file on disk: its bytes' digest and what they hold (`unreadable` when D6 renames it aside). */
type Previous = { digest: string | null; data: WorkspaceData | null; unreadable: { reason: string; detail: string } | null };

/** Read the target whole, and close it; a file that changed between the byte read and the parse refuses (`revision-moved`). */
function readPrevious(deps: WorkspaceDeps, file: string): Previous {
  const bytes = deps.disk.readFile(file);
  if (bytes === null) return { digest: null, data: null, unreadable: null };
  if (!deps.disk.isFile(file)) throw new WriteRefusal("not-a-file", `${file} is not a file`);
  const digest = digestOf(bytes);
  let data: WorkspaceData | null = null;
  let unreadable: Previous["unreadable"] = null;
  try {
    data = readWorkspaceFile(deps.h5, file);
  } catch (err) {
    if (!(err instanceof WorkspaceRefusal)) throw err;
    if (!UNREADABLE.has(err.reason)) throw new WriteRefusal(err.reason, `${err.detail}; nothing was replaced`);
    unreadable = { reason: err.reason, detail: err.detail };
  }
  if (digestOf(deps.disk.readFile(file)) !== digest) throw new WriteRefusal("revision-moved", `${file} changed while this save read it; nothing was replaced. Save again.`);
  return { digest, data, unreadable };
}

/** Read the set's workspace for the page: the tables and each set's verdict against the set's model. */
export async function readWorkspace(deps: WorkspaceDeps, set: OpenSet | { reason: string }): Promise<WorkspaceIn> {
  const where = workspaceOf(set);
  if ("reason" in where) return { status: "none", reason: where.reason };
  const { file, model } = where;
  await deps.ready;
  let bytes: Uint8Array | null;
  try {
    bytes = deps.disk.readFile(file);
  } catch (err) {
    return { status: "refused", file, reason: isHeld(err) ? "held" : "cannot-read", detail: `${file} cannot be read: ${err instanceof Error ? err.message : String(err)}` };
  }
  if (bytes === null) return { status: "absent", file, token: tokenOf(file, null) };
  const token = tokenOf(file, digestOf(bytes));
  let data: WorkspaceData;
  try {
    data = readWorkspaceFile(deps.h5, file);
  } catch (err) {
    if (!(err instanceof WorkspaceRefusal)) throw err;
    return UNREADABLE.has(err.reason) ? { status: "unreadable", file, reason: err.reason, detail: err.detail, token } : { status: "refused", file, reason: err.reason, detail: err.detail };
  }
  if (!deps.disk.isFile(model)) return { status: "refused", file, reason: "unpaired", detail: `${file}: no model archive ${model} beside it` };
  let side: ModelSide;
  try {
    side = new ModelSide(deps.h5, model);
  } catch (err) {
    if (err instanceof WorkspaceRefusal) return { status: "refused", file, reason: err.reason, detail: err.detail };
    throw err;
  }
  const { verdicts, refused } = verdictsOf(data.sets, side);
  return {
    status: "ready",
    file,
    token,
    version: data.version,
    revision: data.revision,
    writer: data.writer,
    saveId: data.saveId,
    readOnly: data.newer,
    sets: setsIn(data.sets, verdicts),
    plots: data.plots,
    views: data.views,
    refused: [...data.refused, ...refused],
    broken: Object.entries(data.broken).map(([table, reason]) => ({ table, reason: reason! })),
  };
}

const EDIT_KEYS: Record<WorkspaceEdit["op"], readonly string[]> = {
  create: ["expect", "ids", "kind", "name", "op"],
  rename: ["expect", "from", "op", "table", "to"],
  delete: ["expect", "name", "op", "table"],
  accept: ["expect", "name", "op"],
};
const TABLE_NAMES = ["sets", "plots", "views"];

/** The edit the page sent, or why it is refused. Closed keys: a `path` (or any key the op does not name) refuses. */
export function parseEdit(raw: unknown): WorkspaceEdit | string {
  const form = "workspace:save takes one edit, { op: create | rename | delete | accept, ... }, and never a path or a file";
  if (typeof raw !== "object" || raw === null || Array.isArray(raw)) return `${form}; refused ${shown(raw)}`;
  const e = raw as Record<string, unknown>;
  const op = e["op"];
  if (typeof op !== "string" || !Object.hasOwn(EDIT_KEYS, op)) return `${form}; refused ${shown(raw)}`;
  const keys = Object.keys(e).sort();
  const want = EDIT_KEYS[op as WorkspaceEdit["op"]];
  if (keys.length !== want.length || keys.some((k, i) => k !== want[i])) return `${form}; ${op} takes { ${want.join(", ")} }; refused ${shown(raw)}`;
  const str = (k: string) => typeof e[k] === "string";
  if (!str("expect")) return `${form}; expect is the token of the read the edit was made on; refused ${shown(raw)}`;
  const expect = e["expect"] as string;
  switch (op) {
    case "create":
      if (!str("name") || !(SET_KINDS as readonly unknown[]).includes(e["kind"]) || !Array.isArray(e["ids"])) return `${form}; create takes a name, a kind (nodes or elements) and ids; refused ${shown(raw)}`;
      return { op, name: e["name"] as string, kind: e["kind"] as "nodes" | "elements", ids: [...(e["ids"] as unknown[])] as number[], expect };
    case "rename":
      if (!TABLE_NAMES.includes(e["table"] as string) || !str("from") || !str("to")) return `${form}; rename takes a table and two names; refused ${shown(raw)}`;
      return { op, table: e["table"] as "sets", from: e["from"] as string, to: e["to"] as string, expect };
    case "delete":
      if (!TABLE_NAMES.includes(e["table"] as string) || !str("name")) return `${form}; delete takes a table and a name; refused ${shown(raw)}`;
      return { op, table: e["table"] as "sets", name: e["name"] as string, expect };
    default:
      if (!str("name")) return `${form}; accept takes a set's name; refused ${shown(raw)}`;
      return { op: "accept", name: e["name"] as string, expect };
  }
}

/** What one edit changes in the file, or a refusal. */
function changeOf(edit: WorkspaceEdit, previous: WorkspaceData | null, model: ModelSide, file: string): Change {
  switch (edit.op) {
    case "create": {
      // The app's create never replaces: a name the file holds is renamed or deleted first (no silent loss, INV-2).
      const held = previous?.raw.sets?.some((r) => r.name === edit.name) ?? false;
      if (held) throw new WriteRefusal("duplicate", `${file} already holds a set named ${JSON.stringify(edit.name)}; rename or delete it first`);
      return { sets: [{ name: edit.name, kind: edit.kind, ids: edit.ids }] };
    }
    case "rename":
      return { rename: [{ table: edit.table, from: edit.from, to: edit.to }] };
    case "delete":
      return { remove: { [edit.table]: [edit.name] } };
    case "accept": {
      const s = previous?.sets.find((x) => x.name === edit.name);
      if (!s) throw new WriteRefusal("bad-param", `${file} has no standing set named ${JSON.stringify(edit.name)}`);
      const v = verdictsOf([s], model).verdicts.get(s.name);
      if (v?.verdict !== "partial") throw new WriteRefusal("bad-param", `set ${JSON.stringify(s.name)} reads ${v?.verdict ?? "refused"}; only a partial set has survivors to accept`);
      return { sets: [{ name: s.name, kind: s.kind, ids: Array.from(s.ids).filter((_, i) => v.members[i]) }] };
    }
  }
}

function utcStamp(d: Date): string {
  const p = (n: number, w = 2) => String(n).padStart(w, "0");
  return `${p(d.getUTCFullYear(), 4)}${p(d.getUTCMonth() + 1)}${p(d.getUTCDate())}T${p(d.getUTCHours())}${p(d.getUTCMinutes())}${p(d.getUTCSeconds())}Z`;
}

/** Move an unreadable target to `<name>.unreadable-<UTC>[-n]`, never over a file; a holder refuses `held` and leaves it in place. */
function renameAside(deps: WorkspaceDeps, file: string): string {
  const stamp = utcStamp(deps.now());
  for (let n = 0; n < 1000; n++) {
    const aside = join(dirname(file), `${basename(file)}.unreadable-${stamp}${n ? `-${n}` : ""}`);
    try {
      deps.disk.link(file, aside);
    } catch (err) {
      if (code(err) === "EEXIST") continue;
      throw err;
    }
    try {
      deps.disk.unlink(file);
    } catch (err) {
      try {
        deps.disk.unlink(aside);
      } catch {
        // The aside name stays beside the whole old file; the refusal below says nothing was replaced.
      }
      throw err;
    }
    return aside;
  }
  throw new WriteRefusal("not-a-file", `${file}: no free name to rename it aside`);
}

/**
 * Remove this host's partials of `file` whose writer has died; a failure here never fails the save (Python's
 * `_sweep_dead_partials`). VU9b's combination writer (./combinations.ts) sweeps its `.user.h5` partials with it too.
 */
export function sweepDeadPartials(deps: Pick<WorkspaceDeps, "host" | "alive"> & { disk: Pick<WorkspaceDisk, "list" | "unlink"> }, file: string): void {
  if (deps.host === UNKNOWN_HOST) return;
  const esc = basename(file).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const pattern = new RegExp(`^${esc}\\.([a-z0-9_]+)-(\\d+)-[0-9a-f]+\\.partial$`);
  let names: string[];
  try {
    names = deps.disk.list(dirname(file));
  } catch {
    return;
  }
  for (const name of names) {
    const m = pattern.exec(name);
    if (m === null || m[1] !== deps.host || deps.alive(Number(m[2]))) continue;
    try {
      deps.disk.unlink(join(dirname(file), name));
    } catch {
      // A dead writer's lock outlives it for a moment; the next save tries again.
    }
  }
}

/** Save `edit` into the set's workspace: a whole new file through this writer's partial, or a refusal that replaced nothing. */
export async function saveWorkspace(deps: WorkspaceDeps, set: OpenSet | { reason: string }, raw: unknown): Promise<WorkspaceSaveIn> {
  const edit = parseEdit(raw);
  if (typeof edit === "string") return { ok: false, reason: "bad-request", detail: edit, file: null, held: false };
  const where = workspaceOf(set);
  if ("reason" in where) return { ok: false, reason: "unpaired", detail: `nothing to save into: ${where.reason}`, file: null, held: false };
  const { file, model } = where;
  if (!deps.disk.isFile(model)) return { ok: false, reason: "unpaired", detail: `${file}: no ${basename(model)} beside it`, file, held: false };
  await deps.ready;
  let partial: string | null = null;
  let aside: string | null = null;
  let replaced = false;
  try {
    const previous = readPrevious(deps, file);
    if (tokenOf(file, previous.digest) !== edit.expect) {
      throw new WriteRefusal("stale", `${file} is not the file the edit was made on (another set is open now, or another writer saved since it was read); it is read again: check it and save again. Nothing was replaced.`);
    }
    const side = new ModelSide(deps.h5, model);
    const planned = buildWrite(previous.data, side, changeOf(edit, previous.data, side, file), file);
    const meta = { revision: planned.revision, saveId: deps.saveId(), writer: deps.writer };
    partial = join(dirname(file), `${basename(file)}.${deps.host}-${deps.pid}-${deps.token()}.partial`);
    writePartial(deps.h5, partial, meta, planned.tables);
    deps.step?.("partial-written");
    verifyPartial(deps.h5, partial, meta, planned.tables);
    deps.disk.fsync(partial);
    deps.step?.("verified");
    // The second of two writers refuses here: the file must still be the bytes this save read, or still be absent.
    let now: string | null;
    try {
      now = digestOf(deps.disk.readFile(file));
    } catch {
      now = "unreadable now";
    }
    if (now !== previous.digest) throw new WriteRefusal("revision-moved", `${file} ${previous.digest === null ? "was created" : "changed"} while this save ran; nothing was replaced. Save again.`);
    if (previous.unreadable !== null) aside = renameAside(deps, file);
    deps.step?.("before-replace");
    deps.disk.rename(partial, file);
    replaced = true;
  } catch (err) {
    if (partial !== null) {
      try {
        deps.disk.unlink(partial);
      } catch {
        // gone already, or the sweep of a later save removes it
      }
    }
    if (aside !== null && !replaced) {
      // The target name is free (the old file sits aside); put it back, so a failed save leaves the folder as it was.
      try {
        deps.disk.link(aside, file);
        deps.disk.unlink(aside);
        aside = null;
      } catch {
        // The old file stays whole under the aside name; the refusal names it.
      }
    }
    if (err instanceof WriteRefusal) return { ok: false, reason: err.reason, detail: err.detail, file, held: false };
    if (err instanceof WorkspaceRefusal) return { ok: false, reason: err.reason, detail: err.detail, file, held: false };
    if (isHeld(err)) return heldRefusal(file);
    return { ok: false, reason: "failed", detail: `the save of ${file} failed; nothing was replaced: ${err instanceof Error ? err.message : String(err)}${aside ? ` (the old file is whole as ${aside})` : ""}`, file, held: false };
  }
  sweepDeadPartials(deps, file);
  return { ok: true, read: await readWorkspace(deps, set), aside };
}

/** What main.ts gives the doors: the window's set now, and why this mode may not write (null in the human's window). */
export interface WorkspaceDoors {
  set(): OpenSet | { reason: string };
  cannotWrite(): string | null;
  deps: WorkspaceDeps;
}

/** Register `workspace:read` and `workspace:save` on `ipc` (main.ts passes `ipcMain`). */
export function registerWorkspace(ipc: { handle(channel: string, fn: (event: unknown, ...args: unknown[]) => unknown): void }, doors: WorkspaceDoors): void {
  ipc.handle("workspace:read", async (_e, ...args): Promise<WorkspaceIn> => {
    if (args.length) return { status: "none", reason: `workspace:read takes no argument (main reads the open set's workspace); refused ${shown(args)}` };
    return readWorkspace(doors.deps, doors.set());
  });
  ipc.handle("workspace:save", async (_e, ...args): Promise<WorkspaceSaveIn> => {
    if (args.length !== 1) return { ok: false, reason: "bad-request", detail: `workspace:save takes one edit and never a path; refused ${shown(args)}`, file: null, held: false };
    const why = doors.cannotWrite();
    if (why !== null) return { ok: false, reason: "read-only", detail: why, file: null, held: false };
    return saveWorkspace(doors.deps, doors.set(), args[0]);
  });
}
