// VU9a (ADR 0132 D6; HG9 on #2803): the app's workspace save, on a real disk.
//
// Each test copies the worked-case frame (test/fixtures/vu9a/frame.h5, written
// by Python) into a folder of its own and saves through main/workspace.ts
// `saveWorkspace`, the function behind the `workspace:save` door. The oracles:
//   - a killed save leaves the old file whole: a child process kills itself
//     hard between the steps of a save, and the target's bytes are the old ones;
//   - a held target is a typed refusal (`held`, naming the file), the old file
//     whole and no partial left: on any platform through a disk that refuses
//     the replace as Windows does (EPERM), and on Windows with a real open
//     handle on the target;
//   - the second of two writers refuses (`revision-moved`), an unreadable file
//     is renamed aside (D6) and never overwritten, a newer file is never
//     written over (D2), a tag the model lacks is refused (`bad-row`);
//   - the same file reads back in the app with the verdicts Python gives.

import assert from "node:assert/strict";
import { execFileSync, spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { closeSync, copyFileSync, existsSync, mkdirSync, mkdtempSync, openSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { after, test } from "node:test";
import { fileURLToPath } from "node:url";
import * as h5wasm from "h5wasm/node";
import { heldRefusal, nodeDeps, nodeDisk, readWorkspace, saveWorkspace, type WorkspaceDeps, type WorkspaceDisk } from "../../src/main/workspace.ts";
import { readWorkspaceFile, writePartial, type WorkspaceH5 } from "../../src/reader/workspace.ts";
import type { OpenSet } from "../../src/main/pairing.ts";
import type { WorkspaceIn, WorkspaceSaveIn } from "../../src/state/types/workspace.ts";

await h5wasm.ready;
const H5 = h5wasm as unknown as WorkspaceH5;
const FIX = (p: string) => fileURLToPath(new URL(`../fixtures/vu9a/${p}`, import.meta.url));
const DRIVER = fileURLToPath(new URL("./workspaceKillDriver.ts", import.meta.url));
const root = mkdtempSync(join(tmpdir(), "vu9a-save-"));
after(() => rmSync(root, { recursive: true, force: true }));

let n = 0;
/** A folder holding a copy of the frame (and, when asked, Python's workspace of it). */
function folder(withWorkspace = false): { dir: string; set: OpenSet; ws: string } {
  const dir = join(root, `f${n++}`);
  rmSync(dir, { recursive: true, force: true });
  mkdirSync(dir);
  copyFileSync(FIX("frame.h5"), join(dir, "frame.h5"));
  if (withWorkspace) copyFileSync(FIX("frame.workspace.h5"), join(dir, "frame.workspace.h5"));
  return { dir, set: { model: join(dir, "frame.h5"), geometry: null, results: null, user: null, workspace: null }, ws: join(dir, "frame.workspace.h5") };
}

const deps = (over: Partial<WorkspaceDeps> = {}): WorkspaceDeps => ({ ...nodeDeps(H5, h5wasm.ready, "apeGmshViewer test"), ...over });
/** The token main hands the page for the file as it is now (the bytes' sha256, or "absent"), computed here on its own. */
const tokenNow = (file: string) => {
  const bytes = existsSync(file) ? createHash("sha256").update(readFileSync(file)).digest("hex") : "absent";
  const path = resolve(file);
  return createHash("sha256").update(`${process.platform === "win32" || process.platform === "darwin" ? path.toLowerCase() : path}\0${bytes}`).digest("hex");
};
/** A save made on a read of the file as it is now: the edit carries that read's token, as the page sends it. */
const save = (d: WorkspaceDeps, set: OpenSet, edit: Record<string, unknown>) => saveWorkspace(d, set, { ...edit, expect: tokenNow(join(set.model!, "..", "frame.workspace.h5")) });
const partials = (dir: string) => readdirSync(dir).filter((f) => f.endsWith(".partial"));
const ready = (r: WorkspaceIn) => {
  assert.equal(r.status, "ready", JSON.stringify(r));
  return r as Extract<WorkspaceIn, { status: "ready" }>;
};
const saved = (r: WorkspaceSaveIn) => {
  assert.ok(r.ok, JSON.stringify(r));
  return r as Extract<WorkspaceSaveIn, { ok: true }>;
};
const refused = (r: WorkspaceSaveIn) => {
  assert.ok(!r.ok, `the save landed: ${JSON.stringify(r)}`);
  return r as Extract<WorkspaceSaveIn, { ok: false }>;
};

test("a first save creates the file: the set reads current, with the model's positions, and revision 1", async () => {
  const f = folder();
  const r = saved(await save(deps(), f.set, { op: "create", name: "roof", kind: "nodes", ids: [31, 32, 33] }));
  assert.equal(r.aside, null);
  const read = ready(r.read);
  assert.equal(read.revision, 1);
  assert.equal(read.writer, "apeGmshViewer test");
  assert.deepEqual(read.sets.map((s) => [s.name, s.verdict?.verdict]), [["roof", "current"]]);
  const back = readWorkspaceFile(H5, f.ws);
  assert.deepEqual([...back.sets[0]!.xyz], [0, 0, 9, 6, 0, 9, 12, 0, 9], "positions come from the model, not the page");
  assert.equal(back.sets[0]!.fem_hash, "f".repeat(32));
  assert.deepEqual(partials(f.dir), []);
});

test("rename, delete and an element set: every other row is copied verbatim, the revision counts", async () => {
  const f = folder(true);
  const before = readWorkspaceFile(H5, f.ws);
  saved(await save(deps(), f.set, { op: "rename", table: "sets", from: "roof", to: "roof line" }));
  saved(await save(deps(), f.set, { op: "delete", table: "plots", name: "roof drift" }));
  const r = saved(await save(deps(), f.set, { op: "create", name: "two columns", kind: "elements", ids: [204, 205] }));
  const after_ = readWorkspaceFile(H5, f.ws);
  assert.equal(after_.revision, before.revision + 3);
  assert.deepEqual(after_.sets.map((s) => s.name), ["roof line", "base", "column line", "springs", "columns", "two columns"]);
  // The renamed row keeps its members, positions and hashes bit for bit.
  assert.deepEqual([...after_.raw.sets![0]!.xyz], [...before.raw.sets![0]!.xyz]);
  assert.deepEqual(after_.raw.sets!.slice(1, 5), before.raw.sets!.slice(1, 5));
  assert.deepEqual(after_.plots, []);
  assert.deepEqual(after_.raw.views, before.raw.views);
  assert.equal(ready(r.read).sets.find((s) => s.name === "two columns")?.verdict?.verdict, "current");
});

test("a moved set takes the model's hashes on the next save; a partial set keeps its survivors on accept", async () => {
  const f = folder(true);
  copyFileSync(FIX("cases/base3_moved.h5"), f.set.model!);
  const read = ready(await readWorkspace(deps(), f.set));
  const v = (r: typeof read, name: string) => r.sets.find((s) => s.name === name)?.verdict;
  assert.equal(v(read, "roof")?.verdict, "moved");
  assert.equal(v(read, "base")?.verdict, "partial");
  const r = ready(saved(await save(deps(), f.set, { op: "accept", name: "base" })).read);
  assert.equal(v(r, "roof")?.verdict, "current", "the save wrote the model's hashes into the moved row");
  assert.equal(v(r, "base")?.verdict, "current");
  assert.deepEqual(r.sets.find((s) => s.name === "base")?.ids, [1, 2], "the survivors, in order");
  assert.equal(refused(await save(deps(), f.set, { op: "accept", name: "roof" })).reason, "bad-param");
});

test("a killed save leaves the old file whole, at every step before the replace", async () => {
  for (const step of ["partial-written", "verified", "before-replace"]) {
    const f = folder(true);
    const old = readFileSync(f.ws);
    let died = false;
    try {
      execFileSync(process.execPath, [DRIVER, `--dir=${f.dir}`, `--kill-at=${step}`], { stdio: ["ignore", "pipe", "pipe"], timeout: 60_000 });
    } catch (err) {
      const e = err as { stdout?: Buffer; stderr?: Buffer };
      // Killed, not failed: it printed nothing, and no error of its own.
      assert.ok(!String(e.stdout ?? "").includes("NOT KILLED") && !/Error:/.test(String(e.stderr ?? "").replace(/HDF5-DIAG: Error/g, "")), `the driver failed rather than died at ${step}: ${String(e.stderr ?? "")}`);
      died = true;
    }
    assert.ok(died, `the driver was not killed at ${step}`);
    assert.deepEqual(readFileSync(f.ws), old, `${step}: the target is the old bytes`);
    assert.deepEqual(readWorkspaceFile(H5, f.ws).sets.map((s) => s.name), ["roof", "base", "column line", "springs", "columns"]);
    // The dead writer's partial is left; the next save on this host sweeps it.
    assert.equal(partials(f.dir).length, 1, `${step}: one dead partial`);
    saved(await save(deps(), f.set, { op: "create", name: "after", kind: "nodes", ids: [1] }));
    assert.deepEqual(partials(f.dir), [], `${step}: the next save swept the dead writer's partial`);
  }
});

test("a save killed at a random moment leaves a whole file every time", async () => {
  const f = folder(true);
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
    const back = readWorkspaceFile(H5, f.ws);
    assert.ok(back.sets.some((s) => s.name === "roof"), `kill ${i}: the file reads whole`);
  }
});

/** A disk that refuses one operation as Windows does when another program holds the file. */
function holding(op: keyof WorkspaceDisk, when: (path: string) => boolean = () => true): WorkspaceDisk {
  return {
    ...nodeDisk,
    [op]: (...args: string[]) => {
      if (when(args[1] ?? args[0]!)) throw Object.assign(new Error(`EPERM: operation not permitted, ${op}`), { code: "EPERM" });
      return (nodeDisk[op] as (...a: string[]) => unknown)(...args);
    },
  } as WorkspaceDisk;
}

test("a replace another program blocks is the typed refusal: it names the file, the old file is whole, no partial is left", async () => {
  const f = folder(true);
  const old = readFileSync(f.ws);
  const r = refused(await save(deps({ disk: holding("rename") }), f.set, { op: "create", name: "blocked", kind: "nodes", ids: [1] }));
  assert.deepEqual(r, heldRefusal(f.ws));
  assert.equal(r.held, true);
  assert.match(r.detail, /open in another program.*Close it there and save again\. Nothing was replaced\./);
  assert.deepEqual(readFileSync(f.ws), old);
  assert.deepEqual(partials(f.dir), []);
});

test("Windows: a real handle on the target blocks the replace, and the save refuses held at once (no retry loop)", { skip: process.platform !== "win32" && "only Windows refuses to replace a file another process holds open" }, async () => {
  const f = folder(true);
  const old = readFileSync(f.ws);
  const fd = openSync(f.ws, "r");
  try {
    const t = Date.now();
    const r = refused(await save(deps(), f.set, { op: "create", name: "blocked", kind: "nodes", ids: [1] }));
    assert.equal(r.reason, "held");
    assert.equal(r.file, f.ws);
    assert.ok(Date.now() - t < 5000, "no retry loop");
  } finally {
    closeSync(fd);
  }
  assert.deepEqual(readFileSync(f.ws), old);
  assert.deepEqual(partials(f.dir), []);
  // Closed there, the same save lands.
  saved(await save(deps(), f.set, { op: "create", name: "blocked", kind: "nodes", ids: [1] }));
});

test("the second of two writers refuses: the file changed between the read and the replace", async () => {
  const f = folder(true);
  let other: Buffer | null = null;
  const r = refused(
    await save(
      deps({
        step: (name) => {
          if (name !== "verified") return;
          // Another writer's save lands now: Python's, a revision ahead.
          writeFileSync(f.ws, readFileSync(FIX("plate/plate.workspace.h5")));
          other = readFileSync(f.ws);
        },
      }),
      f.set,
      { op: "create", name: "late", kind: "nodes", ids: [1] },
    ),
  );
  assert.equal(r.reason, "revision-moved");
  assert.deepEqual(readFileSync(f.ws), other, "the other writer's file stands");
  assert.deepEqual(partials(f.dir), []);
});

test("an unreadable file is renamed aside, never overwritten, and the answer names the aside file", async () => {
  const f = folder();
  writeFileSync(f.ws, "not a workspace");
  const read = await readWorkspace(deps(), f.set);
  assert.equal(read.status, "unreadable");
  const r = saved(await save(deps({ now: () => new Date(Date.UTC(2026, 10, 11, 9, 8, 7)) }), f.set, { op: "create", name: "roof", kind: "nodes", ids: [31] }));
  assert.equal(r.aside, `${f.ws}.unreadable-20261111T090807Z`);
  assert.equal(readFileSync(r.aside!, "utf8"), "not a workspace");
  assert.equal(ready(r.read).sets[0]!.name, "roof");
  // A second unreadable file in the same second takes the next free name.
  writeFileSync(f.ws, "again");
  const r2 = saved(await save(deps({ now: () => new Date(Date.UTC(2026, 10, 11, 9, 8, 7)) }), f.set, { op: "create", name: "roof", kind: "nodes", ids: [31] }));
  assert.equal(r2.aside, `${f.ws}.unreadable-20261111T090807Z-1`);
  assert.equal(readFileSync(r.aside!, "utf8"), "not a workspace", "the first aside file is untouched");
});

test("an unreadable file held by another program refuses held and stays in place", async () => {
  const f = folder();
  writeFileSync(f.ws, "not a workspace");
  const r = refused(await save(deps({ disk: holding("unlink", (p) => p === f.ws) }), f.set, { op: "create", name: "roof", kind: "nodes", ids: [31] }));
  assert.equal(r.reason, "held");
  assert.equal(readFileSync(f.ws, "utf8"), "not a workspace");
  assert.deepEqual(readdirSync(f.dir).filter((x) => x.includes(".unreadable-") || x.endsWith(".partial")), [], "no aside name and no partial left");
});

test("a newer file opens read-only and is never written over (D2)", async () => {
  const f = folder();
  // A 1.1.0 file: this app's writer, then the version key moved on.
  writePartial(H5, f.ws, { revision: 3, saveId: "newerfile", writer: "a later apeGmsh" }, { sets: [], plots: [], views: [] });
  const b = readFileSync(f.ws);
  const patched = Buffer.from(b.toString("latin1").replace("1.0.0", "1.1.0"), "latin1");
  writeFileSync(f.ws, patched);
  const read = ready(await readWorkspace(deps(), f.set));
  assert.match(read.readOnly ?? "", /newer than this app's 1\.0\.0: it is shown read-only/);
  const r = refused(await save(deps(), f.set, { op: "create", name: "roof", kind: "nodes", ids: [31] }));
  assert.equal(r.reason, "newer");
  assert.deepEqual(readFileSync(f.ws), patched);
});

test("refusals that write nothing: a tag the model lacks, a name the file holds, an empty set, a bad name", async () => {
  const f = folder(true);
  const old = readFileSync(f.ws);
  const cases: [Record<string, unknown>, string][] = [
    [{ op: "create", name: "ghost", kind: "nodes", ids: [31, 999] }, "bad-row"],
    [{ op: "create", name: "roof", kind: "nodes", ids: [1] }, "duplicate"],
    [{ op: "create", name: "empty", kind: "nodes", ids: [] }, "bad-members"],
    [{ op: "create", name: " padded", kind: "nodes", ids: [1] }, "empty-name"],
    [{ op: "create", name: "twice", kind: "nodes", ids: [1, 1] }, "bad-members"],
    [{ op: "rename", table: "sets", from: "roof", to: "base" }, "duplicate"],
    [{ op: "delete", table: "views", name: "nope" }, "bad-param"],
  ];
  for (const [edit, reason] of cases) assert.equal(refused(await save(deps(), f.set, edit)).reason, reason, JSON.stringify(edit));
  assert.deepEqual(readFileSync(f.ws), old);
  assert.deepEqual(partials(f.dir), []);
});

test("the edit never carries a path: any key the op does not name refuses before anything is read", async () => {
  const f = folder();
  const elsewhere = join(f.dir, "elsewhere.workspace.h5");
  for (const edit of [
    { op: "create", name: "roof", kind: "nodes", ids: [31], path: elsewhere },
    { op: "create", name: "roof", kind: "nodes", ids: [31], file: elsewhere },
    { op: "delete", table: "sets", name: "roof", target: elsewhere },
    elsewhere,
    { op: "write", path: elsewhere },
    null,
  ]) {
    const r = refused(await saveWorkspace(deps(), f.set, edit));
    assert.equal(r.reason, "bad-request", JSON.stringify(edit));
    assert.match(r.detail, /never a path/);
  }
  assert.ok(!existsSync(f.ws) && !existsSync(elsewhere), "nothing was written");
});

test("a save made on a read the file no longer matches refuses stale, and replaces nothing", async () => {
  const f = folder(true);
  const read = ready(await readWorkspace(deps(), f.set));
  assert.equal(read.token, tokenNow(f.ws), "the read's token binds the file's path to its bytes");
  // Another writer saves after the page read the file.
  writeFileSync(f.ws, readFileSync(FIX("plate/plate.workspace.h5")));
  const other = readFileSync(f.ws);
  const r = refused(await saveWorkspace(deps(), f.set, { op: "delete", table: "sets", name: "roof", expect: read.token }));
  assert.equal(r.reason, "stale");
  assert.equal(r.file, f.ws);
  assert.deepEqual(readFileSync(f.ws), other);
  assert.deepEqual(partials(f.dir), []);
  // The token of "no file yet" does not stand for a file that appeared.
  const absent = folder();
  const none = (await readWorkspace(deps(), absent.set)) as { token: string };
  writeFileSync(absent.ws, readFileSync(FIX("plate/plate.workspace.h5")));
  assert.equal(refused(await saveWorkspace(deps(), absent.set, { op: "create", name: "x", kind: "nodes", ids: [1], expect: none.token })).reason, "stale");
});

test("a truncated workspace (half of a real one) is unreadable, as Python reads it, and a save moves it aside (D6)", async () => {
  const f = folder();
  const half = readFileSync(FIX("truncated.workspace.h5"));
  writeFileSync(f.ws, half);
  const read = await readWorkspace(deps(), f.set);
  assert.equal(read.status, "unreadable");
  assert.equal((read as { reason: string }).reason, "not-hdf5");
  const r = saved(await save(deps(), f.set, { op: "create", name: "roof", kind: "nodes", ids: [31] }));
  assert.ok(r.aside !== null && r.aside.includes(".unreadable-"));
  assert.deepEqual(readFileSync(r.aside!), half, "the truncated file is kept whole under the aside name");
  assert.equal(ready(r.read).sets[0]!.name, "roof");
});
