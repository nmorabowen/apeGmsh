// The combination door (ADR 0130 A4, A7.3; HG9 on #2693): `combinations:save`,
// the app's one write into the set's `<stem>.user.h5`.
//
// As the workspace door (./workspace.ts), the page never names a path: the door
// writes the window's set's own user file (`target()`, main.ts `pairedUser`).
// It takes one edit with closed keys (`{op: "add", row, expect}` or
// `{op: "revert", expect}`), so a `path`, or any key the op does not name, is
// refused before anything is read. Only the human's window writes: every agent
// mode reads, none saves (ADR 0128 INV-1).
//
// A save is ADR 0130 A4: copy the file to this writer's own partial, open the
// partial in `a` mode, add `g<N+1>` (or, for a revert, point `@current` back),
// set `@current` and `/meta@revision`, reopen and verify, refuse if the
// on-disk revision moved, and replace the original. The append-mode probe
// (#2693, comment 6770969917) proved `a` on the live file corrupts it when the
// writer is killed, so the owner's five build rules hold here (HG9, comment
// 6776817023):
//   1. Never open the live `.user.h5` for writing. It is copied (read-only), and
//      only the partial is opened for writing (reader/h5open.ts `appendH5`
//      refuses any other name).
//   2. The app closes its own `.user.h5` handles before the replace and reads
//      the file again after it (`release`; reader/h5open.ts `closeHandlesOn`).
//   3. A holder outside the app blocks the replace: a typed refusal, `held`,
//      naming the file. No retry, no rename-aside.
//   4. This writer's dead `<stem>.user.h5.<host>-<pid>-<random>.partial` files
//      are swept on open (`user:read`, main.ts) and on save; a live pid's, or
//      another host's, is left alone (./workspace.ts `sweepDeadPartials`).
//   5. Free disk space of at least the file's size is checked before the copy.
// A save is bound to the read it was made on: `expect` is that read's token
// (`combinationToken`, over the path main resolved and `/meta@revision`).
// The descriptor written is checked by the readers' rules first (`resolve`, as
// Python's writer `_check_combinations`): a refused row never reaches disk.
// Every refusal replaces nothing: the old file is whole and this writer's
// partial is gone.
//
// No Electron import (main.ts passes `ipcMain` in), so the tests run in Node.

import { randomBytes } from "node:crypto";
import { closeSync, constants as fsConstants, fsyncSync, openSync, readdirSync, readSync, renameSync, statfsSync, statSync, unlinkSync, writeSync } from "node:fs";
import { basename, dirname, join } from "node:path";
import { inspect } from "node:util";
import { combinationHeldRefusal } from "../combination/author.ts";
import { appendH5, closeHandlesOn } from "../reader/h5open.ts";
import { modelCombinationRows, readForWrite, readRevision, readUserSet, UserWriteRefused, type UserH5, type UserWriteBase } from "../reader/userFile.ts";
import { resolve as resolveRows, type Combination } from "../reader/userRules.ts";
import type { CombinationEdit, CombinationSaveIn, UserCombinationIn, UserReadIn } from "../state/types/user.ts";
import { hostToken, pidAlive, sweepDeadPartials, tokenOf } from "./workspace.ts";

export { combinationHeldRefusal };

/** The disk as a save sees it; the tests pass one that fails where they choose. */
export interface CombinationDisk {
  isFile(path: string): boolean;
  size(path: string): number;
  /** free bytes this process may write in `dir` */
  free(dir: string): number;
  /** copy `from` to `to`, creating `to` exclusively; `chunk` runs after each block written (a test kills the writer there) */
  copy(from: string, to: string, chunk: () => void): void;
  /** replace `to` with `from` in one rename */
  rename(from: string, to: string): void;
  unlink(path: string): void;
  fsync(path: string): void;
  list(dir: string): string[];
}

const code = (err: unknown): string | undefined => (err as NodeJS.ErrnoException | null)?.code;
/** One copy block: a multi-GB file is copied in steps, so a kill lands inside the copy (the crash test). */
export const COPY_BLOCK = 8 * 1024 * 1024;

export const nodeCombinationDisk: CombinationDisk = {
  isFile: (path) => {
    try {
      return statSync(path).isFile();
    } catch {
      return false;
    }
  },
  size: (path) => statSync(path).size,
  free: (dir) => {
    const s = statfsSync(dir);
    return Number(s.bavail) * Number(s.bsize);
  },
  copy: (from, to, chunk) => {
    // The live file is only ever read (rule 1); the copy is this writer's own, created exclusively.
    const src = openSync(from, "r");
    let dst: number | null = null;
    try {
      dst = openSync(to, fsConstants.O_WRONLY | fsConstants.O_CREAT | fsConstants.O_EXCL);
      const buf = Buffer.allocUnsafe(COPY_BLOCK);
      for (;;) {
        const n = readSync(src, buf, 0, COPY_BLOCK, null);
        if (n === 0) break;
        let off = 0;
        while (off < n) off += writeSync(dst, buf, off, n - off);
        chunk();
      }
    } finally {
      if (dst !== null) closeSync(dst);
      closeSync(src);
    }
  },
  rename: (from, to) => renameSync(from, to),
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

/** What h5wasm gives the append (`h5wasm/node`, after `await h5wasm.ready`). */
interface H5WriteNode {
  path: string;
  attrs: Record<string, unknown>;
  keys(): string[];
  get(name: string): unknown;
  create_group(name: string): unknown;
  create_dataset(args: { name: string; data: unknown; shape: number[]; dtype: string }): unknown;
  create_attribute(name: string, data: unknown, shape: number[] | null, dtype: string): unknown;
  delete_attribute(name: string): number;
}
interface H5WriteFile extends H5WriteNode {
  file_id: bigint | number;
  flush(): unknown;
  close(): unknown;
}
/** What this module needs of h5wasm: the readers' interface, and an appending File. */
export type CombinationH5 = UserH5 & { File: new (filename: string, mode: "r" | "a") => never };

export interface CombinationDeps {
  h5: CombinationH5;
  ready: Promise<unknown>;
  disk: CombinationDisk;
  /** this machine's name as a partial carries it (./workspace.ts `hostToken`) */
  host: string;
  pid: number;
  alive(pid: number): boolean;
  /** 8 hex digits for a partial's name */
  token(): string;
  /** `/meta@writer`, e.g. "apeGmshViewer 0.0.1" */
  writer: string;
  /** close this app's own handles on `file` before the replace (rule 2); the count it closed */
  release(file: string): number;
  /** a directory listing for A6's staleness read after the save */
  listDir(dir: string): readonly string[];
  /** a test's hook between the steps of a save (`copying`, `copied`, `appending`, `appended`, `verified`, `before-replace`); never set in the app */
  step?(name: string): void;
}

export function nodeCombinationDeps(h5: CombinationH5, ready: Promise<unknown>, writer: string, listDir: (dir: string) => readonly string[]): CombinationDeps {
  return {
    h5,
    ready,
    disk: nodeCombinationDisk,
    host: hostToken(),
    pid: process.pid,
    alive: pidAlive,
    token: () => randomBytes(4).toString("hex"),
    writer,
    release: closeHandlesOn,
    listDir,
  };
}

/** The set's user file, as main resolves it, and the model and results its rows and staleness are judged with; or why there is none. */
export type CombinationTarget = { user: string; model: string | null; results: string | null } | { reason: string };

/** The token a `user:read` hands the page and a save expects back: the path main resolved and the file's `/meta@revision`. */
export function combinationToken(file: string, revision: number): string {
  return tokenOf(file, `revision ${revision}`);
}

const shown = (x: unknown): string => {
  try {
    return inspect(x, { depth: 4, breakLength: Infinity, maxArrayLength: 10, maxStringLength: 300 });
  } catch {
    return Object.prototype.toString.call(x);
  }
};

const ROW_KEYS = ["damping", "envelope", "fraction", "name", "rule", "terms"];
const TERM_KEYS = ["factor", "kind", "mode", "source", "step"];
const EDIT_KEYS: Record<CombinationEdit["op"], readonly string[]> = { add: ["expect", "op", "row"], revert: ["expect", "op"] };
const closed = (o: unknown, want: readonly string[]): o is Record<string, unknown> => {
  if (typeof o !== "object" || o === null || Array.isArray(o)) return false;
  const keys = Object.keys(o).sort();
  return keys.length === want.length && keys.every((k, i) => k === want[i]);
};
const numOrNull = (x: unknown) => x === null || typeof x === "number";

/** The edit the page sent, or why it is refused. Closed keys at every level: a `path` (or any key the op does not name) refuses. */
export function parseCombinationEdit(raw: unknown): CombinationEdit | string {
  const form = "combinations:save takes one edit, { op: add, row, expect } or { op: revert, expect }, and never a path or a file";
  const op = (raw as { op?: unknown } | null)?.op;
  if (typeof op !== "string" || !Object.hasOwn(EDIT_KEYS, op)) return `${form}; refused ${shown(raw)}`;
  if (!closed(raw, EDIT_KEYS[op as CombinationEdit["op"]])) return `${form}; ${op} takes { ${EDIT_KEYS[op as CombinationEdit["op"]].join(", ")} }; refused ${shown(raw)}`;
  if (typeof raw["expect"] !== "string") return `${form}; expect is the token of the read the edit was made on; refused ${shown(raw)}`;
  if (op === "revert") return { op: "revert", expect: raw["expect"] };
  const row = raw["row"];
  if (!closed(row, ROW_KEYS)) return `${form}; a row is { ${ROW_KEYS.join(", ")} }; refused ${shown(raw)}`;
  const terms = row["terms"];
  const typed =
    typeof row["name"] === "string" &&
    typeof row["rule"] === "string" &&
    typeof row["envelope"] === "string" &&
    numOrNull(row["fraction"]) &&
    numOrNull(row["damping"]) &&
    Array.isArray(terms) &&
    terms.every((t) => closed(t, TERM_KEYS) && typeof t["kind"] === "string" && typeof t["source"] === "string" && [t["step"], t["mode"], t["factor"]].every((v) => typeof v === "number"));
  if (!typed) return `${form}; a row holds a name, a rule, an envelope, a fraction and a damping (a number or null), and terms of { ${TERM_KEYS.join(", ")} }; refused ${shown(raw)}`;
  const r = row as unknown as UserCombinationIn;
  return {
    op: "add",
    expect: raw["expect"],
    row: { name: r.name, rule: r.rule, envelope: r.envelope, fraction: r.fraction, damping: r.damping, terms: r.terms.map((t) => ({ kind: t.kind, source: t.source, step: t.step, mode: t.mode, factor: t.factor })) },
  };
}

/** A refusal of this save, with the schema's word. */
class SaveRefused extends Error {
  readonly reason: string;
  readonly detail: string;
  constructor(reason: string, detail: string) {
    super(`${reason}: ${detail}`);
    this.reason = reason;
    this.detail = detail;
  }
}

const HELD_CODES = new Set(["EPERM", "EACCES", "EBUSY"]);
const isHeld = (err: unknown) => HELD_CODES.has(code(err) ?? "");

/** What a save writes: the rows of the new generation (an add), or the generation `@current` goes back to (a revert). */
type Plan = { op: "add"; generation: number; rows: Combination[]; name: string } | { op: "revert"; generation: number; rows: Combination[] };

/** The plan of `edit` on `base`, or a refusal: a row the readers refuse never reaches disk. */
function planOf(edit: CombinationEdit, base: UserWriteBase, modelRows: Combination[], file: string): Plan {
  if (edit.op === "revert") {
    // A7.3: one revert. `@current` goes back to the generation before it, once: a revert of a revert is refused.
    if (base.current === null) throw new SaveRefused("nothing-to-revert", `${file} holds no /combinations to revert`);
    const top = base.held[base.held.length - 1]!;
    if (base.current !== top) throw new SaveRefused("nothing-to-revert", `${file}: the last save was reverted already (@current is g${base.current}, below g${top}); the app gives one revert (ADR 0130 A7.3)`);
    if (base.previous === null) throw new SaveRefused("nothing-to-revert", `${file}: /combinations holds no generation before g${base.current}`);
    return { op: "revert", generation: base.previous.generation, rows: base.previous.rows };
  }
  const row: Combination = { ...edit.row, terms: edit.row.terms.map((t) => ({ ...t })) };
  const rows = [...base.rows, row];
  const judged = resolveRows(rows, modelRows);
  const mine = judged.refused.filter((r) => r.origin === "user");
  if (mine.length) {
    // The new row's own refusal first: it is the one the human is writing.
    const lines = [...mine].sort((a, b) => Number(b.row === rows.length - 1) - Number(a.row === rows.length - 1)).map((r) => `${JSON.stringify(r.name)}: ${r.reason}: ${r.detail}`);
    throw new SaveRefused("refused", `the readers refuse ${lines.join("; ")}; nothing was written`);
  }
  const next = base.held.length ? base.held[base.held.length - 1]! + 1 : 0;
  return { op: "add", generation: next, rows, name: row.name };
}

/** Write the plan into the partial (opened `a`, never the live file), with `@current` and `/meta@revision`. */
function appendPlan(deps: CombinationDeps, partial: string, plan: Plan, base: UserWriteBase): void {
  const f = appendH5(deps.h5 as unknown as Parameters<typeof appendH5<H5WriteFile>>[0], partial);
  try {
    const meta = f.get("meta") as H5WriteNode;
    let zone: H5WriteNode;
    if (base.current === null) {
      // A first save creates the zone and stamps its version, as Python's writer does.
      f.create_group("combinations");
      zone = f.get("combinations") as H5WriteNode;
      meta.create_attribute("combinations_schema_version", "1.0.0", null, "S");
    } else {
      zone = f.get("combinations") as H5WriteNode;
    }
    if (plan.op === "add") writeGeneration(zone, `g${plan.generation}`, plan.rows);
    deps.step?.("appending");
    setAttr(zone, "current", plan.generation, "<i4");
    setAttr(meta, "revision", base.revision + 1, "<i4");
    setAttr(meta, "writer", deps.writer, "S");
    f.flush();
  } finally {
    f.close();
  }
}

function setAttr(node: H5WriteNode, name: string, value: unknown, dtype: string): void {
  if (Object.hasOwn(node.attrs, name) && node.delete_attribute(name) !== 0) throw new Error(`${node.path}@${name} cannot be replaced`);
  node.create_attribute(name, value, null, dtype);
}

/** One `g<N>`, in Python's `write_generation` layout: text columns vlen UTF-8, `i4` and `f8` columns, NaN for none. */
function writeGeneration(zone: H5WriteNode, link: string, rows: readonly Combination[]): void {
  zone.create_group(link);
  const g = zone.get(link) as H5WriteNode;
  const terms = rows.flatMap((r) => r.terms);
  const text = (node: H5WriteNode, name: string, data: string[]) => node.create_dataset({ name, data, shape: [data.length], dtype: "S" });
  // "<d", not "<f8": h5wasm 0.10.3 reads the width of a float dtype from its letter, and "<f8" writes f4 (reader/workspace.ts).
  const f8 = (node: H5WriteNode, name: string, data: number[]) => node.create_dataset({ name, data: Float64Array.from(data), shape: [data.length], dtype: "<d" });
  const i4 = (node: H5WriteNode, name: string, data: number[]) => node.create_dataset({ name, data: Int32Array.from(data), shape: [data.length], dtype: "<i4" });
  text(g, "name", rows.map((r) => r.name));
  text(g, "rule", rows.map((r) => r.rule));
  text(g, "envelope", rows.map((r) => r.envelope));
  f8(g, "fraction", rows.map((r) => (r.fraction === null ? Number.NaN : r.fraction)));
  f8(g, "damping", rows.map((r) => (r.damping === null ? Number.NaN : r.damping)));
  const offsets = [0];
  for (const r of rows) offsets.push(offsets[offsets.length - 1]! + r.terms.length);
  i4(g, "term_offsets", offsets);
  g.create_group("terms");
  const t = g.get("terms") as H5WriteNode;
  text(t, "kind", terms.map((x) => x.kind));
  text(t, "source", terms.map((x) => x.source));
  i4(t, "step", terms.map((x) => x.step));
  i4(t, "mode", terms.map((x) => x.mode));
  f8(t, "factor", terms.map((x) => x.factor));
}

/** Bit for bit, so a NaN or a signed zero compares as stored. */
const sameNumber = (a: number | null, b: number | null): boolean => {
  if (a === null || b === null) return a === b;
  const x = new Float64Array([a, b]);
  const u = new Uint32Array(x.buffer);
  return u[0] === u[2] && u[1] === u[3];
};
const sameRows = (a: readonly Combination[], b: readonly Combination[]): boolean =>
  a.length === b.length &&
  a.every((r, i) => {
    const s = b[i]!;
    return (
      r.name === s.name &&
      r.rule === s.rule &&
      r.envelope === s.envelope &&
      sameNumber(r.fraction, s.fraction) &&
      sameNumber(r.damping, s.damping) &&
      r.terms.length === s.terms.length &&
      r.terms.every((t, j) => {
        const u = s.terms[j]!;
        return t.kind === u.kind && t.source === u.source && t.step === u.step && t.mode === u.mode && sameNumber(t.factor, u.factor);
      })
    );
  });

/** Read the partial back through the reader: the revision, `@current`, every row bit for bit, and every other zone still there. */
function verifyPartial(deps: CombinationDeps, partial: string, plan: Plan, base: UserWriteBase): void {
  let back: UserWriteBase;
  try {
    back = readForWrite(deps.h5, partial);
  } catch (err) {
    throw new SaveRefused("partial-mismatch", `${partial} does not read back: ${err instanceof Error ? err.message : String(err)}`);
  }
  if (back.revision !== base.revision + 1) throw new SaveRefused("partial-mismatch", `${partial}: /meta@revision reads ${back.revision}, not ${base.revision + 1}`);
  if (back.current !== plan.generation) throw new SaveRefused("partial-mismatch", `${partial}: @current reads g${back.current}, not g${plan.generation}`);
  if (!sameRows(back.rows, plan.rows)) throw new SaveRefused("partial-mismatch", `${partial}: the rows of g${plan.generation} did not read back`);
  const zones = new Set([...base.zones, "combinations"]);
  if (back.zones.length !== zones.size || back.zones.some((z) => !zones.has(z))) throw new SaveRefused("partial-mismatch", `${partial}: its zones read ${back.zones.join(", ")}, not ${[...zones].sort().join(", ")}`);
}

/** Save `raw` into the set's `.user.h5` by A4, or a refusal that replaced nothing. */
export async function saveCombination(deps: CombinationDeps, target: CombinationTarget, raw: unknown): Promise<CombinationSaveIn> {
  const edit = parseCombinationEdit(raw);
  if (typeof edit === "string") return { ok: false, reason: "bad-request", detail: edit, file: null, held: false };
  if ("reason" in target) return { ok: false, reason: "unpaired", detail: `nothing to save into: ${target.reason}`, file: null, held: false };
  const file = target.user;
  if (!deps.disk.isFile(file)) {
    return { ok: false, reason: "no-user-file", detail: `${file} does not exist. The app adds to a user file and never creates one: write it with Python (apeGmsh.results.user.write_user_file)`, file, held: false };
  }
  await deps.ready;
  sweepDeadPartials(deps, file);
  let partial: string | null = null;
  try {
    let base: UserWriteBase;
    try {
      base = readForWrite(deps.h5, file);
    } catch (err) {
      if (err instanceof UserWriteRefused) throw new SaveRefused(err.reason, err.detail);
      throw err;
    }
    if (combinationToken(file, base.revision) !== edit.expect) {
      throw new SaveRefused("stale", `${file} is not the file the edit was made on (another set is open now, or another writer saved since it was read); it is read again: check it and save again. Nothing was replaced.`);
    }
    let modelRows: Combination[] = [];
    if (target.model !== null && deps.disk.isFile(target.model)) {
      try {
        modelRows = modelCombinationRows(deps.h5, target.model);
      } catch (err) {
        if (err instanceof UserWriteRefused) throw new SaveRefused("refused", `${err.detail}; the rows cannot be judged against it, so nothing was written`);
        throw err;
      }
    }
    const plan = planOf(edit, base, modelRows, file);
    // Rule 5: the copy needs a second file's worth of disk.
    const size = deps.disk.size(file);
    const free = deps.disk.free(dirname(file));
    if (free < size) {
      throw new SaveRefused("no-space", `saving needs a copy of ${file} (${size} bytes) beside it, and the disk has ${free} bytes free. Free some space and save again. Nothing was written.`);
    }
    partial = join(dirname(file), `${basename(file)}.${deps.host}-${deps.pid}-${deps.token()}.partial`);
    let first = true;
    deps.disk.copy(file, partial, () => {
      if (first) deps.step?.("copying");
      first = false;
    });
    deps.step?.("copied");
    appendPlan(deps, partial, plan, base);
    deps.step?.("appended");
    verifyPartial(deps, partial, plan, base);
    deps.disk.fsync(partial);
    deps.step?.("verified");
    // The second of two writers refuses here: the revision on disk must still be the one this save read.
    let now: number | string;
    try {
      now = readRevision(deps.h5, file);
    } catch (err) {
      now = `unreadable (${err instanceof Error ? err.message : String(err)})`;
    }
    if (now !== base.revision) throw new SaveRefused("revision-moved", `${file} changed while this save ran (revision ${base.revision} is now ${now}); nothing was replaced. Save again.`);
    // Rule 2: this app's own handles on the file never block, nor outlive, the replace.
    deps.release(file);
    deps.step?.("before-replace");
    deps.disk.rename(partial, file);
    partial = null;
  } catch (err) {
    if (partial !== null) {
      try {
        deps.disk.unlink(partial);
      } catch {
        // gone already, or the sweep of a later save removes it
      }
    }
    if (err instanceof SaveRefused) return { ok: false, reason: err.reason, detail: err.detail, file, held: false };
    // Rule 3: an outside holder blocks the replace (Windows: EPERM); the typed refusal names the file, at once.
    if (isHeld(err)) return combinationHeldRefusal(file);
    return { ok: false, reason: "failed", detail: `the save of ${file} failed; nothing was replaced: ${err instanceof Error ? err.message : String(err)}`, file, held: false };
  }
  sweepDeadPartials(deps, file);
  // The file read again, as main reads it for the page, with its new token.
  const read = readUserSet(deps.h5, file, target.model, target.results, deps.listDir);
  const answer: UserReadIn = read.ok ? { ...read, token: combinationToken(file, read.user.revision) } : read;
  const revision = read.ok ? read.user.revision : -1;
  const generation = read.ok && read.user.combinations.ok ? (read.user.combinations.value.generation ?? -1) : -1;
  return { ok: true, op: edit.op, file, generation, revision, read: answer };
}

/** What main.ts gives the door: the window's set's user file now, and why this mode may not write (null in the human's window). */
export interface CombinationDoors {
  target(): CombinationTarget;
  cannotWrite(): string | null;
  deps: CombinationDeps;
}

/** Register `combinations:save` on `ipc` (main.ts passes `ipcMain`). */
export function registerCombinations(ipc: { handle(channel: string, fn: (event: unknown, ...args: unknown[]) => unknown): void }, doors: CombinationDoors): void {
  ipc.handle("combinations:save", async (_e, ...args): Promise<CombinationSaveIn> => {
    if (args.length !== 1) return { ok: false, reason: "bad-request", detail: `combinations:save takes one edit and never a path; refused ${shown(args)}`, file: null, held: false };
    const why = doors.cannotWrite();
    if (why !== null) return { ok: false, reason: "read-only", detail: why, file: null, held: false };
    return saveCombination(doors.deps, doors.target(), args[0]);
  });
}

/** Rule 4 on open: the sweep `user:read` runs in the human's window before it reads the file. */
export function sweepOnOpen(deps: Pick<CombinationDeps, "host" | "alive" | "disk">, file: string): void {
  sweepDeadPartials(deps, file);
}
