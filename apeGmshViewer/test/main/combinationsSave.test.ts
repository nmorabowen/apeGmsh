// VU9b (ADR 0130 A4, A7.3; HG9 on #2693): the app's combination save, on a real disk.
//
// Each test copies the VU9b fixture (test/fixtures/vu9b: plate.h5 and its
// plate.user.h5, written by Python with one /spectra case and g0 of three rows)
// into a folder of its own and saves through main/combinations.ts
// `saveCombination`, the function behind the `combinations:save` door. The
// oracles, one per HG9 build rule and per gate of the card:
//   - a save adds g<N+1> with the file's rows and the new one, sets @current and
//     /meta@revision, and changes no other zone (a digest of every zone);
//   - one revert puts @current back, by the same copy-then-replace; a second is refused;
//   - rule 1: no h5wasm handle on the live file is ever opened for writing;
//   - rule 2: this app's own handle on the file is closed before the replace;
//   - rule 3: a holder outside the app is the typed refusal naming the file;
//   - rule 4: this host's dead partials are swept on open and on save, a live
//     pid's and another host's are left;
//   - rule 5: a disk without a second file's worth of space refuses before the copy;
//   - a killed save leaves the old file byte-identical, at every step;
//   - a refused descriptor never reaches disk; a save made on another read is `stale`.

import assert from "node:assert/strict";
import { execFileSync, spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { copyFileSync, existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { hostname, tmpdir } from "node:os";
import { join } from "node:path";
import { after, test } from "node:test";
import { fileURLToPath } from "node:url";
import * as h5wasm from "h5wasm/node";
import {
  combinationHeldRefusal,
  combinationToken,
  nodeCombinationDeps,
  nodeCombinationDisk,
  parseCombinationEdit,
  registerCombinations,
  saveCombination,
  sweepOnOpen,
  type CombinationDeps,
  type CombinationDisk,
  type CombinationH5,
} from "../../src/main/combinations.ts";
import { hostToken } from "../../src/main/workspace.ts";
import { listDir } from "../../src/main/pairing.ts";
import { appendH5, openH5, openHandlesOn } from "../../src/reader/h5open.ts";
import { readRevision, readUserFile, type UserH5 } from "../../src/reader/userFile.ts";
import type { CombinationSaveIn, UserCombinationIn } from "../../src/state/types/user.ts";

await h5wasm.ready;
const H5 = h5wasm as unknown as CombinationH5;
const UH5 = h5wasm as unknown as UserH5;
const FIX = (p: string) => fileURLToPath(new URL(`../fixtures/vu9b/${p}`, import.meta.url));
const DRIVER = fileURLToPath(new URL("./combinationsDriver.ts", import.meta.url));
const root = mkdtempSync(join(tmpdir(), "vu9b-save-"));
after(() => rmSync(root, { recursive: true, force: true }));

let n = 0;
/** A folder holding a copy of the plate and Python's user file of it. */
function folder(): { dir: string; file: string; model: string; target: { user: string; model: string; results: null } } {
  const dir = join(root, `f${n++}`);
  mkdirSync(dir);
  copyFileSync(FIX("plate.h5"), join(dir, "plate.h5"));
  copyFileSync(FIX("plate.user.h5"), join(dir, "plate.user.h5"));
  const file = join(dir, "plate.user.h5");
  return { dir, file, model: join(dir, "plate.h5"), target: { user: file, model: join(dir, "plate.h5"), results: null } };
}

const deps = (over: Partial<CombinationDeps> = {}): CombinationDeps => ({ ...nodeCombinationDeps(H5, h5wasm.ready, "apeGmshViewer test", listDir), ...over });
/** The token main hands the page for the file as it is now. */
const tokenNow = (file: string) => combinationToken(file, readRevision(UH5, file));
const ROW: UserCombinationIn = { name: "SRSS Ex modes 1-2", rule: "srss", envelope: "", fraction: null, damping: null, terms: [1, 2].map((k) => ({ kind: "spectrum", source: "Ex (synthetic)", step: -1, mode: k, factor: 1 })) };
const add = (d: CombinationDeps, f: ReturnType<typeof folder>, row: UserCombinationIn = ROW) => saveCombination(d, f.target, { op: "add", row, expect: tokenNow(f.file) });
const revert = (d: CombinationDeps, f: ReturnType<typeof folder>) => saveCombination(d, f.target, { op: "revert", expect: tokenNow(f.file) });
const partials = (dir: string) => readdirSync(dir).filter((x) => x.endsWith(".partial"));
const saved = (r: CombinationSaveIn) => {
  assert.ok(r.ok, JSON.stringify(r));
  return r as Extract<CombinationSaveIn, { ok: true }>;
};
const refused = (r: CombinationSaveIn) => {
  assert.ok(!r.ok, `the save landed: ${JSON.stringify(r)}`);
  return r as Extract<CombinationSaveIn, { ok: false }>;
};
const combos = (file: string, model: string | null) => {
  const c = readUserFile(UH5, file, model).combinations;
  assert.ok(c.ok, JSON.stringify(c));
  return c.value;
};

// -- zone digests: every path, attribute and raw value of a root zone ------------------------

type Node = { path: string; attrs: Record<string, { value: unknown; metadata: unknown }>; keys?: () => string[]; get?: (n: string) => unknown; value?: unknown; metadata?: unknown; shape?: number[] };
const plain = (v: unknown): unknown => (ArrayBuffer.isView(v) ? Array.from(v as unknown as ArrayLike<number | bigint>, (x) => (typeof x === "bigint" ? `${x}n` : Object.is(x, -0) ? "-0" : Number.isNaN(x) ? "NaN" : x)) : typeof v === "bigint" ? `${v}n` : v);
function walk(node: Node, h: ReturnType<typeof createHash>): void {
  h.update(`path ${node.path}\n`);
  for (const k of Object.keys(node.attrs).sort()) h.update(`@${k} ${JSON.stringify(node.attrs[k]!.metadata)} ${JSON.stringify(plain(node.attrs[k]!.value))}\n`);
  if (typeof node.keys === "function") {
    for (const k of node.keys().sort()) walk(node.get!(k) as Node, h);
  } else {
    h.update(`data ${JSON.stringify(node.metadata)} ${JSON.stringify(node.shape)} ${JSON.stringify(plain(node.value))}\n`);
  }
}
/** A sha256 per root zone (and per `/meta`), over its paths, attributes and data. */
function zoneDigests(file: string): Record<string, string> {
  const f = openH5(UH5 as unknown as Parameters<typeof openH5<Node & { close(): void }>>[0], file);
  try {
    const out: Record<string, string> = {};
    for (const k of f.keys!()) {
      const h = createHash("sha256");
      walk(f.get!(k) as Node, h);
      out[k] = h.digest("hex");
    }
    return out;
  } finally {
    f.close();
  }
}
const changed = (a: Record<string, string>, b: Record<string, string>) => [...new Set([...Object.keys(a), ...Object.keys(b)])].filter((k) => a[k] !== b[k]).sort();

// -- the save -----------------------------------------------------------------------------------

test("a save adds g1 with the file's three rows and the new one, @current 1, revision 2, and the readers take it", async () => {
  const f = folder();
  const before = combos(f.file, f.model);
  assert.equal(before.generation, 0);
  const r = saved(await add(deps(), f));
  assert.equal(r.op, "add");
  assert.equal(r.generation, 1);
  assert.equal(r.revision, 2);
  assert.ok(r.read.ok);
  assert.equal(r.read.ok && r.read.token, tokenNow(f.file), "the answer carries the token of the file as it is now");
  const after = combos(f.file, f.model);
  assert.deepEqual(after.generations, [0, 1]);
  assert.deepEqual(after.rows.map((x) => x.name), [...before.rows.map((x) => x.name), ROW.name]);
  assert.deepEqual(after.rows.at(-1), ROW);
  assert.deepEqual(after.refused, []);
  assert.equal(readUserFile(UH5, f.file, null).writer, "apeGmshViewer test");
  assert.deepEqual(partials(f.dir), []);
});

test("zone digests: a save and a revert change /combinations and /meta, and no other zone", async () => {
  const f = folder();
  const d0 = zoneDigests(f.file);
  assert.deepEqual(Object.keys(d0).sort(), ["combinations", "meta", "spectra"]);
  saved(await add(deps(), f));
  const d1 = zoneDigests(f.file);
  assert.deepEqual(changed(d0, d1), ["combinations", "meta"]);
  saved(await revert(deps(), f));
  const d2 = zoneDigests(f.file);
  assert.deepEqual(changed(d1, d2), ["combinations", "meta"]);
  assert.equal(d2["spectra"], d0["spectra"], "/spectra is the bytes Python wrote, through a save and a revert");
});

test("one revert: @current goes back to g0 by copy-then-replace, a second revert is refused, and a save after it adds g2 on g0's rows", async () => {
  const f = folder();
  const g0 = combos(f.file, f.model).rows;
  saved(await add(deps(), f));
  const r = saved(await revert(deps(), f));
  assert.equal(r.op, "revert");
  assert.equal(r.generation, 0);
  assert.equal(r.revision, 3);
  const back = combos(f.file, f.model);
  assert.equal(back.generation, 0);
  assert.deepEqual(back.generations, [0, 1], "g1 stays in the file: earlier generations are the backup (A4)");
  assert.deepEqual(back.rows, g0);
  const bytes = readFileSync(f.file);
  const again = refused(await revert(deps(), f));
  assert.equal(again.reason, "nothing-to-revert");
  assert.match(again.detail, /one revert/);
  assert.deepEqual(readFileSync(f.file), bytes, "a refused revert writes nothing");
  saved(await add(deps(), f, { ...ROW, name: "after the revert" }));
  const g2 = combos(f.file, f.model);
  assert.equal(g2.generation, 2);
  assert.deepEqual(g2.rows.map((x) => x.name), [...g0.map((x) => x.name), "after the revert"]);
});

test("a file with nothing to revert, and a set with no user file, are refused and nothing is written", async () => {
  const f = folder();
  const bytes = readFileSync(f.file);
  assert.equal(refused(await revert(deps(), f)).reason, "nothing-to-revert");
  assert.deepEqual(readFileSync(f.file), bytes);
  rmSync(f.file);
  const r = refused(await saveCombination(deps(), f.target, { op: "add", row: ROW, expect: "x" }));
  assert.equal(r.reason, "no-user-file");
  assert.equal(existsSync(f.file), false, "the app never creates a user file");
});

test("a refused descriptor never reaches disk: a duplicate name, a directional row with no fraction, a cycle", async () => {
  const f = folder();
  const bytes = readFileSync(f.file);
  const cases: [UserCombinationIn, string][] = [
    [{ ...ROW, name: "SRSS Ex" }, "duplicate"],
    [{ ...ROW, name: "dir", rule: "directional" }, "bad-param"],
    [{ ...ROW, name: "loop", rule: "linear", terms: [{ kind: "combination", source: "loop", step: -1, mode: 0, factor: 1 }] }, "cycle"],
    [{ ...ROW, name: " padded" }, "empty-name"],
  ];
  for (const [row, reason] of cases) {
    const r = refused(await add(deps(), f, row));
    assert.equal(r.reason, "refused", JSON.stringify(r));
    assert.match(r.detail, new RegExp(`${reason}`), r.detail);
    assert.deepEqual(readFileSync(f.file), bytes, `${row.name}: the file is untouched`);
    assert.deepEqual(partials(f.dir), [], `${row.name}: no copy was made`);
  }
});

test("a save made on another read is stale; a writer that lands during the save makes it refuse revision-moved, and theirs stands", async () => {
  const f = folder();
  const bytes = readFileSync(f.file);
  const r = refused(await saveCombination(deps(), f.target, { op: "add", row: ROW, expect: combinationToken(f.file, 7) }));
  assert.equal(r.reason, "stale");
  assert.deepEqual(readFileSync(f.file), bytes);
  let theirs: Buffer | null = null;
  const late = refused(
    await add(
      deps({
        step: (name) => {
          if (name !== "verified") return;
          // Another writer's save lands now: the app-written fixture, three revisions ahead.
          writeFileSync(f.file, readFileSync(FIX("app_written/plate.user.h5")));
          theirs = readFileSync(f.file);
        },
      }),
      f,
    ),
  );
  assert.equal(late.reason, "revision-moved");
  assert.deepEqual(readFileSync(f.file), theirs);
  assert.deepEqual(partials(f.dir), []);
});

test("a file newer than this app writes is never written over: /combinations at 1.1.0", async () => {
  const f = folder();
  const w = new (h5wasm as unknown as { File: new (p: string, m: "a") => { get(p: string): { delete_attribute(n: string): number; create_attribute(n: string, v: unknown, s: null, d: string): void }; close(): void } }).File(f.file, "a");
  const meta = w.get("meta");
  meta.delete_attribute("combinations_schema_version");
  meta.create_attribute("combinations_schema_version", "1.1.0", null, "S");
  w.close();
  const bytes = readFileSync(f.file);
  const r = refused(await add(deps(), f));
  assert.equal(r.reason, "newer");
  assert.deepEqual(readFileSync(f.file), bytes);
});

// -- HG9 rule 1: the live file is never opened for writing ----------------------------------------

test("rule 1: a save opens the live file read-only only; every write goes to this writer's own partial", async () => {
  const f = folder();
  const opens: [string, string][] = [];
  const File = (h5wasm as unknown as { File: new (p: string, m: string) => unknown }).File;
  const spy = new Proxy(H5, {
    get(t, k, rcv) {
      if (k !== "File") return Reflect.get(t, k, rcv);
      return new Proxy(File, {
        construct(target, args) {
          opens.push([String(args[0]), String(args[1])]);
          return Reflect.construct(target, args);
        },
      });
    },
  }) as CombinationH5;
  saved(await add(deps({ h5: spy }), f));
  saved(await revert(deps({ h5: spy }), f));
  const writes = opens.filter(([, m]) => m !== "r");
  assert.ok(writes.length >= 2, `the save opened its partials for writing: ${JSON.stringify(opens)}`);
  for (const [p, m] of writes) assert.ok(p.endsWith(".partial") && p !== f.file, `${p} was opened "${m}"`);
  assert.ok(opens.some(([p, m]) => p === f.file && m === "r"), "the live file is read");
  assert.throws(() => appendH5(H5 as unknown as Parameters<typeof appendH5<{ file_id: number }>>[0], f.file), /never .*plate\.user\.h5/);
});

// -- HG9 rule 2: the app's own handles are closed before the replace -------------------------------

test("rule 2: a handle this app holds on the file is closed before the replace, and the save lands", async () => {
  const f = folder();
  const mine = openH5(UH5, f.file);
  assert.equal(openHandlesOn(f.file), 1);
  const r = await add(deps(), f);
  saved(r);
  assert.equal(openHandlesOn(f.file), 0, "the save closed the app's own handle");
  void mine;
});

test("rule 2, Windows: without the release, the app's own handle blocks its replace (why the rule exists)", { skip: process.platform !== "win32" && "only Windows refuses to replace a file a handle holds open" }, async () => {
  const f = folder();
  const mine = openH5(UH5, f.file) as { close(): void };
  try {
    const r = refused(await add(deps({ release: () => 0 }), f));
    assert.equal(r.reason, "held");
  } finally {
    mine.close();
  }
});

// -- HG9 rule 3: an outside holder is a typed refusal naming the file ------------------------------

/** A disk that refuses the replace as Windows does when another program holds the file. */
const holding: CombinationDisk = {
  ...nodeCombinationDisk,
  rename: () => {
    throw Object.assign(new Error("EPERM: operation not permitted, rename"), { code: "EPERM" });
  },
};

test("rule 3: a replace another program blocks is the typed refusal: it names the file, the old file is whole, no partial is left", async () => {
  const f = folder();
  const bytes = readFileSync(f.file);
  const r = refused(await add(deps({ disk: holding }), f));
  assert.deepEqual(r, combinationHeldRefusal(f.file));
  assert.match(r.detail, /open in another program.*Close it there and save again\. Nothing was replaced\./);
  assert.deepEqual(readFileSync(f.file), bytes);
  assert.deepEqual(partials(f.dir), []);
});

test("rule 3, Windows: another process holding the file open gets held at once (no retry), and the save lands once it lets go", { skip: process.platform !== "win32" && "only Windows refuses to replace a file another process holds open" }, async () => {
  const f = folder();
  const bytes = readFileSync(f.file);
  // An outside holder: another process, as a Python session holding h5py.File(..., "r") would be.
  const holder = spawn(process.execPath, ["-e", `require("fs").openSync(${JSON.stringify(f.file)}, "r"); process.stdout.write("HOLDING\\n"); setInterval(() => {}, 1000);`], { stdio: ["ignore", "pipe", "inherit"] });
  try {
    await new Promise<void>((res, rej) => {
      holder.stdout.once("data", () => res());
      holder.once("exit", () => rej(new Error("the holder exited")));
    });
    const t = Date.now();
    const r = refused(await add(deps(), f));
    assert.equal(r.reason, "held");
    assert.equal(r.file, f.file);
    assert.ok(Date.now() - t < 5000, "no retry loop");
    assert.deepEqual(readFileSync(f.file), bytes);
    assert.deepEqual(partials(f.dir), []);
  } finally {
    holder.kill();
    await new Promise((res) => holder.once("exit", res));
  }
  saved(await add(deps(), f));
});

// -- HG9 rule 4: sweep this writer's dead partials ----------------------------------------------

/** A pid that ran and has exited. */
function deadPid(): number {
  const out = execFileSync(process.execPath, ["-e", "process.stdout.write(String(process.pid))"]);
  return Number(out.toString());
}

test("rule 4: on open and on save, this host's dead partials go; a live pid's and another host's stay", async () => {
  const f = folder();
  const host = hostToken(hostname());
  const dead = `plate.user.h5.${host}-${deadPid()}-0badf00d.partial`;
  const live = `plate.user.h5.${host}-${process.pid}-0badf00d.partial`;
  const other = `plate.user.h5.otherhost-${deadPid()}-0badf00d.partial`;
  for (const p of [dead, live, other]) writeFileSync(join(f.dir, p), "x");
  sweepOnOpen(deps(), f.file);
  assert.deepEqual(partials(f.dir).sort(), [live, other].sort(), "the open swept the dead partial only");
  writeFileSync(join(f.dir, dead), "x");
  saved(await add(deps(), f));
  assert.deepEqual(partials(f.dir).sort(), [live, other].sort(), "the save swept the dead partial only");
});

// -- HG9 rule 5: free disk space before the copy --------------------------------------------------

test("rule 5: a disk with less free space than the file refuses before the copy, with a message, and writes nothing", async () => {
  const f = folder();
  const bytes = readFileSync(f.file);
  let copied = false;
  const r = refused(
    await add(
      deps({
        disk: {
          ...nodeCombinationDisk,
          free: () => bytes.length - 1,
          copy: (...a) => {
            copied = true;
            nodeCombinationDisk.copy(...a);
          },
        },
      }),
      f,
    ),
  );
  assert.equal(r.reason, "no-space");
  assert.match(r.detail, new RegExp(`${bytes.length} bytes.*${bytes.length - 1} bytes free`));
  assert.equal(copied, false);
  assert.deepEqual(readFileSync(f.file), bytes);
  assert.deepEqual(partials(f.dir), []);
  // Exactly the file's size is enough.
  saved(await add(deps({ disk: { ...nodeCombinationDisk, free: () => bytes.length } }), f));
});

// -- the crash test --------------------------------------------------------------------------------

test("crash: a save killed during the copy, during the append, or before the replace leaves the old file byte-identical, and the next open sweeps its partial", async () => {
  for (const [step, op] of [
    ["copying", "add"],
    ["copied", "add"],
    ["appending", "add"],
    ["appended", "add"],
    ["verified", "add"],
    ["before-replace", "add"],
    ["appending", "revert"],
    ["before-replace", "revert"],
  ] as const) {
    const f = folder();
    if (op === "revert") saved(await add(deps(), f));
    const old = readFileSync(f.file);
    let died = false;
    try {
      execFileSync(process.execPath, [DRIVER, `--dir=${f.dir}`, `--op=${op}`, `--kill-at=${step}`, "--block=4096"], { stdio: ["ignore", "pipe", "pipe"], timeout: 60_000 });
    } catch (err) {
      const e = err as { stdout?: Buffer; stderr?: Buffer };
      // Killed, not failed: it printed nothing, and no error of its own.
      assert.ok(!String(e.stdout ?? "").includes("NOT KILLED") && !/Error:/.test(String(e.stderr ?? "").replace(/HDF5-DIAG: Error/g, "")), `the driver failed rather than died at ${step}: ${String(e.stderr ?? "")}`);
      died = true;
    }
    assert.ok(died, `the driver was not killed at ${op} ${step}`);
    assert.deepEqual(readFileSync(f.file), old, `${op} ${step}: the target is the old bytes`);
    assert.ok(readUserFile(UH5, f.file, f.model).combinations.ok, `${op} ${step}: the file reads whole`);
    assert.equal(partials(f.dir).length, 1, `${op} ${step}: one dead partial`);
    if (step === "copying") assert.ok(readFileSync(join(f.dir, partials(f.dir)[0]!)).length < old.length, "killed inside the copy: the partial is short");
    sweepOnOpen(deps(), f.file);
    assert.deepEqual(partials(f.dir), [], `${op} ${step}: the next open swept the dead writer's partial`);
  }
});

test("crash: a saving loop killed at random moments leaves a file that reads whole every time", async () => {
  const f = folder();
  for (let i = 0; i < 6; i++) {
    const child = spawn(process.execPath, [DRIVER, `--dir=${f.dir}`, "--loop"], { stdio: ["ignore", "pipe", "pipe"] });
    let stderr = "";
    child.stderr.on("data", (d: Buffer) => (stderr += d.toString()));
    const exited = new Promise((res) => child.once("exit", res));
    await Promise.race([new Promise<void>((res) => child.stdout.once("data", () => res())), exited]);
    await new Promise((res) => setTimeout(res, 150 + 97 * i));
    assert.equal(child.exitCode, null, `kill ${i}: the saving loop stopped on its own: ${stderr}`);
    child.kill("SIGKILL");
    await exited;
    const c = readUserFile(UH5, f.file, f.model).combinations;
    assert.ok(c.ok && c.value.rows.some((r) => r.name === "SRSS Ex"), `kill ${i}: the file reads whole`);
    sweepOnOpen(deps(), f.file);
  }
});

// -- the door ----------------------------------------------------------------------------------------

test("the door: one edit with closed keys, never a path; an agent mode is refused read-only", async () => {
  const f = folder();
  const handlers = new Map<string, (e: unknown, ...a: unknown[]) => unknown>();
  let mode = "view";
  registerCombinations({ handle: (c, fn) => void handlers.set(c, fn) }, { target: () => f.target, cannotWrite: () => (mode === "view" ? null : `--mode=${mode} never writes`), deps: deps() });
  const door = handlers.get("combinations:save")!;
  const bytes = readFileSync(f.file);
  const expect = tokenNow(f.file);
  for (const bad of [
    { op: "add", row: ROW, expect, path: f.file },
    { op: "add", row: { ...ROW, file: f.file }, expect },
    { op: "add", row: { ...ROW, terms: [{ ...ROW.terms[0]!, path: f.file }] }, expect },
    { op: "revert", expect, file: f.file },
    { op: "add", row: ROW },
    { op: "write", expect },
    f.file,
  ]) {
    const r = (await door(null, bad)) as CombinationSaveIn;
    assert.ok(!r.ok && r.reason === "bad-request", JSON.stringify(bad));
  }
  assert.equal(((await door(null, { op: "revert", expect }, f.file)) as CombinationSaveIn & { reason: string }).reason, "bad-request");
  assert.deepEqual(readFileSync(f.file), bytes);
  for (const m of ["render", "capture", "measure"]) {
    mode = m;
    const r = (await door(null, { op: "add", row: ROW, expect })) as CombinationSaveIn;
    assert.ok(!r.ok && r.reason === "read-only", `${m}: ${JSON.stringify(r)}`);
  }
  assert.deepEqual(readFileSync(f.file), bytes);
  mode = "view";
  saved((await door(null, { op: "add", row: ROW, expect })) as CombinationSaveIn);
  assert.equal(typeof parseCombinationEdit({ op: "add", row: ROW, expect }), "object");
});
