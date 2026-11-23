// ADR 0137 D2 (#3058): main opens a model, its zones and a /geometry sibling in the loader
// process, never on the thread that owns the windows.
//
// Two oracles. The client (src/main/loader/client.ts) against a stand-in process: answers
// matched by id, the reader's message passed on unchanged, and a stopped process failing only
// its own reads, with the next read starting a new one. Then main.ts itself, run in a child
// Node process with `electron` stubbed (./loaderDriver.ts), whose `utilityProcess.fork` forks
// the real src/main/loader/process.ts: `model:open` answers as it did in-process (the same
// model with its typed arrays, the same three zones, the same refusal text as `openModel`
// throws), and a loader killed between or during reads costs one refused read, never the app.
// That main imports no model reader at all is `npm run lint`'s rule `model-read-in-loader`
// (test/lint.test.ts plants a violation).

import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { copyFileSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, test } from "node:test";
import { fileURLToPath } from "node:url";
import { openModel } from "../../src/reader/node.ts";
import { Loader, type LoaderChild } from "../../src/main/loader/client.ts";
import type { LoaderMessage, LoaderReply, LoaderRequest } from "../../src/main/loader/protocol.ts";

const FIX = fileURLToPath(new URL("../../fixtures/", import.meta.url));
const DRIVER = fileURLToPath(new URL("./loaderDriver.ts", import.meta.url));

/** A stand-in process: it records what it is sent and answers or exits when the test says. */
function standIn() {
  const spawned: { sent: LoaderRequest[]; reply(m: LoaderReply): void; event(m: LoaderMessage): void; exit(code: number): void }[] = [];
  const spawn = (): LoaderChild => {
    let onMessage: (m: LoaderMessage) => void = () => undefined;
    let onExit: (code: number) => void = () => undefined;
    const c = { sent: [] as LoaderRequest[], reply: (m: LoaderReply) => onMessage(m), event: (m: LoaderMessage) => onMessage(m), exit: (code: number) => onExit(code) };
    spawned.push(c);
    return { postMessage: (m) => c.sent.push(m), onMessage: (fn) => (onMessage = fn), onExit: (fn) => (onExit = fn) };
  };
  return { spawned, loader: new Loader(spawn) };
}

test("the client starts the process on the first read, keeps it, and matches answers to reads by id", async () => {
  const { spawned, loader } = standIn();
  assert.equal(spawned.length, 0, "no process before a read");
  const a = loader.modelCounts("a.h5");
  const b = loader.geometry("b.geometry.h5");
  assert.equal(spawned.length, 1, "one process for both reads");
  assert.deepEqual(spawned[0]!.sent.map((r) => [r.op, "path" in r ? r.path : null]), [["modelCounts", "a.h5"], ["geometry", "b.geometry.h5"]]);
  const [ra, rb] = spawned[0]!.sent;
  spawned[0]!.reply({ id: rb!.id, ok: true, value: { ok: true, geometry: null, sizeBytes: 1, readMs: 0 } });
  spawned[0]!.reply({ id: ra!.id, ok: true, value: { nodes: 15, elements: 24 } });
  assert.deepEqual(await a, { nodes: 15, elements: 24 });
  assert.deepEqual(await b, { ok: true, geometry: null, sizeBytes: 1, readMs: 0 });
  assert.equal(loader.waiting(), 0);
});

test("a reader's failure rejects with the reader's own message", async () => {
  const { spawned, loader } = standIn();
  const a = loader.model("a.h5");
  spawned[0]!.reply({ id: spawned[0]!.sent[0]!.id, ok: false, error: "the /mesh zone was written by a newer apeGmsh (v9)" });
  await assert.rejects(a, { message: "the /mesh zone was written by a newer apeGmsh (v9)" });
});

test("a process that exits fails only its unanswered reads, and the next read starts a new one", async () => {
  const { spawned, loader } = standIn();
  const model = loader.model("a.h5");
  const geometry = loader.geometry("a.geometry.h5");
  const provenance = loader.provenance("a.h5");
  spawned[0]!.exit(3221225477);
  await assert.rejects(model, { message: "the loader process stopped (exit code 3221225477) before it finished reading" });
  assert.deepEqual(await geometry, { ok: false, error: "the loader process stopped (exit code 3221225477) before it finished reading" }, "geometry answers ok: false, as a reader error does");
  assert.equal((await provenance).ok, false);
  const next = loader.modelCounts("a.h5");
  assert.equal(spawned.length, 2, "the next read starts a new process");
  spawned[1]!.reply({ id: spawned[1]!.sent[0]!.id, ok: true, value: { nodes: 1, elements: 1 } });
  assert.deepEqual(await next, { nodes: 1, elements: 1 });
  // A late answer from the stopped process reaches nobody.
  spawned[0]!.reply({ id: spawned[0]!.sent[0]!.id, ok: true, value: "late" });
  assert.equal(loader.waiting(), 0);
});

// ---------------------------------------------------------------------------
// main.ts, with the real loader process.

const root = mkdtempSync(join(tmpdir(), "agv-loader-"));
after(() => rmSync(root, { recursive: true, force: true }));

function run(which: "good" | "unreadable", dir: string): Record<string, unknown> {
  const stdout = execFileSync(process.execPath, [DRIVER, `--case=${which}`, "--mode=describe", `--file=${join(dir, "s.h5")}`, `--out=${join(dir, "out")}`], {
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
    timeout: 60_000,
  });
  const line = stdout.split(/\r?\n/).find((l) => l.startsWith("LOADER "));
  assert.ok(line, `the driver printed no answer:\n${stdout}`);
  return JSON.parse(line.slice("LOADER ".length)) as Record<string, unknown>;
}

type Seen = { ok: boolean; error?: string; nodeIds: string | null; nodes: number | null; blocks: number | null; zones: (boolean | undefined)[] | null };

test("model:open and geometry:open answer through the loader process as they did in-process", async () => {
  const dir = join(root, "good");
  mkdirSync(dir);
  copyFileSync(join(FIX, "bridge_provenance.h5"), join(dir, "s.h5"));
  copyFileSync(join(FIX, "zones.geometry.h5"), join(dir, "s.geometry.h5"));
  const inProcess = await openModel(join(dir, "s.h5"));
  const got = run("good", dir);

  const open = got["open"] as Seen;
  assert.equal(open.ok, true, open.error);
  assert.equal(got["forksAfterOpen"], 1, "model:open forked the loader");
  assert.match(String(got["entry"]), /[\\/]src[\\/]main[\\/]loader[\\/]process\.ts$/);
  assert.equal(open.nodeIds, inProcess.nodeIds.constructor.name, "typed arrays cross the process boundary as typed arrays");
  assert.equal(open.nodes, inProcess.nodeIds.length);
  assert.equal(open.blocks, inProcess.blocks.length);
  assert.deepEqual(open.zones, [true, true, true], "provenance, lineage and stage elements ride with the model");

  const g = got["geometry"] as { ok: boolean; error?: string; drawn: boolean };
  assert.equal(g.ok, true, g.error);
  assert.equal(g.drawn, true, "the /geometry sibling is read");
  assert.equal(got["forksAfterGeometry"], 1, "the loader is kept between reads");
});

test("a loader killed between reads or during one costs that read only: the next read starts another", () => {
  const got = run("good", join(root, "good"));
  assert.equal((got["afterStop"] as Seen).ok, true, (got["afterStop"] as Seen).error);
  assert.equal(got["forksAfterStop"], 2, "the read after the kill forked a new loader");
  const during = got["stoppedDuring"] as Seen;
  assert.equal(during.ok, false, "a read whose loader died is refused");
  assert.match(during.error ?? "", /^the loader process stopped \(exit code -?\d+\) before it finished reading$/);
  assert.equal((got["afterStoppedDuring"] as Seen).ok, true, (got["afterStoppedDuring"] as Seen).error);
  assert.equal(got["forksAtEnd"], 3);
});

test("an unreadable model is refused with the same message openModel throws in-process", async () => {
  const dir = join(root, "unreadable");
  mkdirSync(dir);
  writeFileSync(join(dir, "s.h5"), "this is not an HDF5 file\n");
  const expected = await openModel(join(dir, "s.h5")).then(
    () => assert.fail("precondition: openModel must refuse the file"),
    (err: unknown) => (err instanceof Error ? err.message : String(err)),
  );
  const open = run("unreadable", dir)["open"] as Seen;
  assert.equal(open.ok, false);
  assert.equal(open.error, expected);
});

test("warm() starts the process before the first read, once", async () => {
  const { spawned, loader } = standIn();
  loader.warm();
  loader.warm();
  assert.equal(spawned.length, 1);
  const a = loader.modelCounts("a.h5");
  assert.equal(spawned.length, 1, "the read uses the warm process");
  spawned[0]!.reply({ id: spawned[0]!.sent[0]!.id, ok: true, value: { nodes: 2, elements: 1 } });
  assert.deepEqual(await a, { nodes: 2, elements: 1 });
});

// ---------------------------------------------------------------------------
// The results hosts in the loader (#3058, second slice).

test("a stopped loader's results files are opened again before the next read of them", async () => {
  const { spawned, loader } = standIn();
  const host = loader.results("window");
  const open = host.open("r.ladruno");
  spawned[0]!.reply({ id: spawned[0]!.sent[0]!.id, ok: true, value: { ok: true } });
  assert.deepEqual(await open, { ok: true });
  spawned[0]!.exit(1);
  const step = host.readStep({ path: "r.ladruno", stage: "s", step: 3, seq: 9 });
  await new Promise((r) => setImmediate(r));
  assert.equal(spawned.length, 2, "the read started a new process");
  const sent = () => spawned[1]!.sent.map((r) => ("call" in r ? `${r.host}:${r.call}` : r.op));
  assert.deepEqual(sent(), ["window:open"], "the open is sent first, and the read waits for it");
  spawned[1]!.reply({ id: spawned[1]!.sent[0]!.id, ok: true, value: { ok: true } });
  await new Promise((r) => setImmediate(r));
  assert.deepEqual(sent(), ["window:open", "window:readStep"]);
  spawned[1]!.reply({ id: spawned[1]!.sent[1]!.id, ok: true, value: { ok: true, seq: 9 } });
  assert.deepEqual(await step, { ok: true, seq: 9 });
});

test("a closed host is not opened again in a new process, and the other host is", () => {
  const { spawned, loader } = standIn();
  void loader.results("window").open("w.ladruno");
  const compare = loader.results("compare");
  void compare.open("c.ladruno");
  compare.close();
  spawned[0]!.exit(1);
  loader.warm();
  const opens = spawned[1]!.sent.map((r) => ("call" in r ? [r.host, r.call, r.args[0]] : null));
  assert.deepEqual(opens, [["window", "open", "w.ladruno"]]);
});

test("a results read whose loader stopped answers ok: false with why, never rejects", async () => {
  const { spawned, loader } = standIn();
  const host = loader.results("compare");
  const step = host.readStep({ path: "c", stage: "s", step: 0, seq: 4 });
  const scan = host.scan({ path: "c", stage: "s", components: ["x"], magnitude: false, scanId: 5 }, () => undefined);
  const open = host.open("c");
  await new Promise((r) => setImmediate(r));
  spawned[0]!.exit(7);
  const msg = "the loader process stopped (exit code 7) before it finished reading";
  assert.deepEqual(await step, { ok: false, seq: 4, error: msg });
  assert.deepEqual(await scan, { ok: false, scanId: 5, error: msg });
  assert.deepEqual(await open, { ok: false, error: msg });
});

test("a scan's progress reaches its own listener, by host and scan id", async () => {
  const { spawned, loader } = standIn();
  const seen: string[] = [];
  const w = loader.results("window").scan({ path: "w", stage: "s", components: ["x"], magnitude: false, scanId: 1 }, (p) => seen.push(`w${p.done}`));
  const c = loader.results("compare").scan({ path: "c", stage: "s", components: ["x"], magnitude: false, scanId: 1 }, (p) => seen.push(`c${p.done}`));
  await new Promise((r) => setImmediate(r));
  const child = spawned[0]!;
  const fire = (host: "window" | "compare", done: number) => child.event({ event: "scanProgress", host, progress: { scanId: 1, done, total: 9 } });
  fire("window", 2);
  fire("compare", 5);
  fire("window", 4);
  assert.deepEqual(seen, ["w2", "c5", "w4"]);
  for (const r of child.sent) child.reply({ id: r.id, ok: true, value: { ok: true } });
  await Promise.all([w, c]);
  fire("window", 6);
  assert.deepEqual(seen, ["w2", "c5", "w4"], "a finished scan hears nothing more");
});

test("the results doors answer through the loader: open, step, progress, a cancel and a step read mid-scan, a stop mid-scan", () => {
  const dir = join(root, "results");
  mkdirSync(dir);
  copyFileSync(join(FIX, "contour_staged.h5"), join(dir, "s.h5"));
  copyFileSync(join(FIX, "contour_staged.results.h5"), join(dir, "s.results.h5"));
  const stdout = execFileSync(process.execPath, [DRIVER, "--case=results", "--mode=capture", `--file=${join(dir, "s.h5")}`, `--out=${join(dir, "out.png")}`], {
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
    timeout: 90_000,
  });
  const line = stdout.split(/\r?\n/).find((l) => l.startsWith("LOADER "));
  assert.ok(line, `the driver printed no answer:\n${stdout}`);
  type Said = { ok: boolean; error?: string; stages?: number; nodal?: string[]; cancelled?: boolean };
  const got = JSON.parse(line.slice("LOADER ".length)) as Record<string, unknown>;
  const said = (k: string) => got[k] as Said;
  assert.equal(said("open").ok, true, said("open").error);
  assert.equal(said("open").stages, 3);
  assert.equal(said("step").ok, true, said("step").error);
  assert.deepEqual(said("step").nodal, ["displacement_x", "displacement_y", "displacement_z"]);
  assert.deepEqual(said("cancelled"), { ok: false, error: "cancelled", cancelled: true }, "the cancel reached the running scan");
  const progress = got["progress"] as { scanId: number; done: number; total: number }[];
  assert.ok(progress.length >= 1 && progress.every((p) => p.scanId === 7 && p.total === 4), JSON.stringify(progress));
  assert.equal(said("stepDuringScan").ok, true, "a step read is served while a scan runs");
  assert.equal(said("stoppedScan").ok, false);
  assert.match(said("stoppedScan").error ?? "", /^the loader process stopped \(exit code -?\d+\) before it finished reading$/);
  assert.equal(said("stepAfterStop").ok, true, `the file is opened again in the new loader: ${said("stepAfterStop").error}`);
  assert.equal(got["newForks"], 1);
});

test("the first open starts the process and sends that one open, not a replay of it too", () => {
  const { spawned, loader } = standIn();
  void loader.results("window").open("r.ladruno");
  assert.equal(spawned.length, 1);
  assert.deepEqual(spawned[0]!.sent.map((r) => ("call" in r ? `${r.host}:${r.call}` : r.op)), ["window:open"]);
});

// ---------------------------------------------------------------------------
// ADR 0137 D4 stage 1 (#3057): main's memory status for a model load, with an injected reading.

type Status = { path: string; phase: string; cells: number | null; estimateMB: number | null; availableMB: number; commitHeadroomMB: number | null; reasons: string[] };

function runMemory(dir: string, freeMb: number, commitMb: number): { open: Seen; memory: Status[] } {
  const stdout = execFileSync(process.execPath, [DRIVER, "--case=memory", "--mode=view", `--file=${join(dir, "s.h5")}`, `--free-mb=${freeMb}`, `--commit-mb=${commitMb}`], {
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
    timeout: 60_000,
  });
  const line = stdout.split(/\r?\n/).find((l) => l.startsWith("LOADER "));
  assert.ok(line, `the driver printed no answer:\n${stdout}`);
  return JSON.parse(line.slice("LOADER ".length)) as { open: Seen; memory: Status[] };
}

test("view mode: a model opened with little memory sends the page its memory status before and after the read, and still opens", async () => {
  const dir = join(root, "memory");
  mkdirSync(dir);
  copyFileSync(join(FIX, "bridge_provenance.h5"), join(dir, "s.h5"));
  const model = await openModel(join(dir, "s.h5"));
  const cells = model.blocks.reduce((n, b) => n + b.ids.length, 0);
  const got = runMemory(dir, 1500, 3000);
  assert.equal(got.open.ok, true, `the warning never refuses: ${got.open.error}`);
  const phases = got.memory.map((m) => m.phase);
  assert.equal(phases[0], "before");
  assert.equal(phases.at(-1), "after");
  for (const m of got.memory) {
    assert.equal(m.path, join(dir, "s.h5"));
    assert.equal(m.cells, cells, "the cell count from the dataset shapes is the model's");
    assert.equal(m.availableMB, 1500);
    // Electron reports commit headroom only on Windows (src/main/memory.ts).
    assert.equal(m.commitHeadroomMB, process.platform === "win32" ? 3000 : null);
    assert.deepEqual(m.reasons, ["1.5 GB of RAM is available, under 2.0 GB"]);
  }
});

test("view mode: a model opened with memory to spare sends a status with no reason", () => {
  const got = runMemory(join(root, "memory"), 16_000, 60_000);
  assert.equal(got.open.ok, true);
  assert.ok(got.memory.length >= 2);
  assert.ok(got.memory.every((m) => m.reasons.length === 0), JSON.stringify(got.memory));
});

test("an agent mode (describe) sends no memory status: a still never carries the warning", () => {
  const got = run("good", join(root, "good"));
  assert.equal(got["memorySent"], 0);
});
