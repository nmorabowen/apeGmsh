// VA2h (#2665), end to end: no headless mode hangs. `--render`, `--describe`
// and `capture` are started as real processes with a fault forced into them,
// and each must exit non-zero within seconds with its reason on stderr. Not
// part of `npm test`: it starts Electron, so it needs a display. Run it after
// a build:
//
//   node scripts/build.mjs && node test/spec/headless-exit.e2e.ts [<outdir>]
//
// The faults, for each of the three modes:
//   throw  an error thrown inside the `whenReady` callback, after the hidden
//          window exists (Electron's answer is a modal, and the process stays);
//   crash  the renderer crashed (`webContents.forcefullyCrashRenderer()`) when
//          the page asks main for its model, before the run can finish;
//   hang   the renderer's main thread spinning forever from that same moment
//          (Electron's `unresponsive`).
// (a hang uses `--hang-ms=5000`: Electron's own `unresponsive` never fires for
// a hidden window that gets no input, so main's page watchdog is what ends it),
// and:
//   - capture: a throw inside the capture sequence itself (renderer/app.ts,
//     after its first still) exits 1 with `<out>.error.txt`; it used to exit 0
//     (`app.quit()` ignores process.exitCode: a D7 run stopped at 38 of 59);
//   - measure-results on a file that is not HDF5 exits 1 (the same exit 0);
//   - `--hang-ms` that is not a whole number above 0 is refused, exit 2;
// and, for `--render` only:
//   - INV-6 after a crash part way: a still already written has no manifest
//     entry (the renderer held them), so it is removed, and manifest.json
//     refuses the run naming it and the reason;
//   - a write that throws part way (a full disk) leaves no partial PNG beside
//     the shot's refusal (#2681 review);
// and the #2708 review's four, each failing on the first cut (73be21dc5):
//   - render: a crash while an app-chrome shot's capturePage is in flight names
//     that shot, and no still as written;
//   - describe with main itself blocked 8 s (`--hang-ms=5000`) finishes, exit 0,
//     and is not reported as a hung renderer;
//   - capture: capturePage returning an empty image exits 1 with error.txt;
//   - measure: a failing `.ladruno` read row exits 1;
// and a normal run of each mode still exits 0 and writes only into `--out`: the
// run's own `--user-data-dir`, where Chromium would write by default, stays
// empty (a check no window on the machine can disturb, unlike %APPDATA%).
//
// Every process starts through a generated entry that forces the fault and
// turns `dialog.showErrorBox` into a stderr line, so nothing stops on the
// desktop of whoever runs this; the run fails on any such line. It installs no
// uncaught-error listener of its own, so Electron's default path is what a
// fault meets when the app has none. A process still running at its limit is
// stopped by its own PID (and that PID's tree), never by name.

import assert from "node:assert/strict";
import { execFileSync, spawn } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const ROOT = fileURLToPath(new URL("../../", import.meta.url));
const FIX = join(ROOT, "fixtures");
const electron = createRequire(import.meta.url)("electron") as string;
const OUT = resolve(process.argv[2] ?? mkdtempSync(join(tmpdir(), "agv-headless-e2e-")));
mkdirSync(OUT, { recursive: true });
const log = (s: string) => process.stdout.write(`${s}\n`);
/** `AGV_ONLY=render.crash,capture.hang` runs only those cases (for probing one). */
const ONLY = process.env["AGV_ONLY"]?.split(",") ?? null;
const wanted = (name: string) => ONLY === null || ONLY.includes(name);

const MODEL = join(FIX, "shoebuckle.h5");
/** How long a run may take before it counts as hung: it is stopped, and the case fails. */
const LIMIT_MS = 60000;
/** A fault must end the run this soon after the run starts (a normal run of these takes a few seconds). */
const PROMPT_MS = 30000;

const ENTRY = join(OUT, "fault-entry.mjs");
writeFileSync(
  ENTRY,
  `import { app, BrowserWindow, dialog, ipcMain, nativeImage } from "electron";
import fs from "node:fs";
import { syncBuiltinESMExports } from "node:module";
const nl = String.fromCharCode(10);
dialog.showErrorBox = (t, c) => process.stderr.write("MODAL " + t + " " + c + nl);
const fault = process.env.AGV_FAULT ?? "";
// The fault waits for this file (a still written) when one is named, else for the page's load.
const after = process.env.AGV_AFTER ?? "";
const same = (a, b) => String(a).toLowerCase() === String(b).toLowerCase();
if (fault === "throw") {
  BrowserWindow.prototype.setMenuBarVisibility = function () {
    throw new Error("AGV_FAULT: a forced throw inside whenReady");
  };
}
const strike = (wc) => (fault === "crash" ? wc.forcefullyCrashRenderer() : void wc.executeJavaScript("for (;;) {}").catch(() => {}));
if ((fault === "crash" || fault === "hang") && !after) {
  // When the page asks for its model, which every one of these modes does, before it can finish.
  const handle = ipcMain.handle.bind(ipcMain);
  ipcMain.handle = (channel, fn) => handle(channel, channel !== "model:open" ? fn : (e, ...a) => (strike(e.sender), fn(e, ...a)));
}
if ((fault === "crash" || fault === "hang") && after) {
  app.on("web-contents-created", (_e, wc) => {
    const t = setInterval(() => {
      if (fs.existsSync(after)) {
        clearInterval(t);
        strike(wc);
      }
    }, 5);
  });
}
if (fault === "sequence") {
  // A throw inside the capture sequence itself (renderer/app.ts): its first still written, the
  // #inspector it reads next is gone, so its own next line throws a TypeError.
  const handle = ipcMain.handle.bind(ipcMain);
  ipcMain.handle = (channel, fn) =>
    handle(channel, channel !== "capture:still" ? fn : async (e, suffix, ...a) => {
      const r = await fn(e, suffix, ...a);
      if (suffix === "") await e.sender.executeJavaScript('document.getElementById("inspector").remove()');
      return r;
    });
}
// The #2708 review's four (each its own case below).
const wrap = (name, make) => {
  const handle = ipcMain.handle.bind(ipcMain);
  ipcMain.handle = (channel, fn) => handle(channel, channel !== name ? fn : make(fn));
};
// A crash while main awaits capturePage for an app-chrome shot: the shot's path is handed out, its PNG never written.
// capturePage itself crashes the renderer and stays pending, so the crash is seen while it is in flight.
if (fault === "crash-capture") {
  app.on("web-contents-created", (_e, wc) => {
    wc.capturePage = () => {
      wc.forcefullyCrashRenderer();
      return new Promise(() => {});
    };
  });
}
// A capture that resolves after the run began ending: the real image is taken, the renderer crashed,
// and the image handed back only from a render-process-gone listener that runs after main's (added
// below, after main's import), so after headlessEnd wrote the manifest and before its deferred exit.
let release = null;
if (fault === "crash-late") {
  app.on("web-contents-created", (_e, wc) => {
    const real = wc.capturePage.bind(wc);
    wc.capturePage = async (...a) => {
      const img = await real(...a);
      const late = new Promise((r) => (release = () => r(img)));
      wc.forcefullyCrashRenderer();
      return late;
    };
  });
}
// Main itself blocked for 8 s (with --hang-ms=5000) while the renderer is fine.
if (fault === "main-block") wrap("model:open", (fn) => (e, ...a) => {
  const until = Date.now() + 8000;
  while (Date.now() < until) {}
  return fn(e, ...a);
});
// capturePage returns an empty image.
if (fault === "empty") app.on("web-contents-created", (_e, wc) => (wc.capturePage = async () => nativeImage.createEmpty()));
// measure's .ladruno read row fails: a .ladruno that is not HDF5 appears beside the model once the orbit is done.
if (fault === "measure-read") wrap("measure:done", (fn) => (e, ...a) => {
  fs.writeFileSync(after, "not hdf5");
  return fn(e, ...a);
});
if (fault === "partial") {
  const real = fs.writeFileSync;
  fs.writeFileSync = function (path, data, ...rest) {
    if (same(path, after)) {
      real.call(this, path, data.subarray(0, 64));
      throw new Error("AGV_FAULT: ENOSPC, a forced partial write");
    }
    return real.call(this, path, data, ...rest);
  };
  syncBuiltinESMExports();
}
await import(${JSON.stringify(pathToFileURL(join(ROOT, "dist", "main.mjs")).href)});
if (fault === "crash-late") app.on("render-process-gone", () => release?.());
`,
);

interface Ran {
  code: number | null;
  ms: number;
  text: string;
  hung: boolean;
}

/** One app process with its own profile folder; stopped by its PID tree at the limit. */
function run(user: string, env: Record<string, string>, ...args: string[]): Promise<Ran> {
  const t = performance.now();
  return new Promise((done) => {
    const c = spawn(electron, [ENTRY, `--user-data-dir=${user}`, ...args], { stdio: ["ignore", "pipe", "pipe"], env: { ...process.env, ...env } });
    let text = "";
    let hung = false;
    c.stdout.on("data", (d) => (text += String(d)));
    c.stderr.on("data", (d) => (text += String(d)));
    const killer = setTimeout(() => {
      hung = true;
      try {
        if (process.platform === "win32") execFileSync("taskkill", ["/PID", String(c.pid), "/T", "/F"], { stdio: "ignore" });
        else c.kill("SIGKILL");
      } catch {
        // It exited between the timer and the kill.
      }
    }, LIMIT_MS);
    c.on("exit", (code) => {
      clearTimeout(killer);
      done({ code, ms: performance.now() - t, text, hung });
    });
  });
}

const noModal = (r: Ran, what: string) => assert.doesNotMatch(r.text, /MODAL/, `${what}: a modal would stop on the desktop: ${r.text}`);
/** What is under a folder, `.chromium` (the run's own profile, INV-1) left out. */
const listing = (dir: string) => (existsSync(dir) ? readdirSync(dir).filter((f) => f !== ".chromium").sort() : []);
const emptyOrAbsent = (dir: string) => !existsSync(dir) || readdirSync(dir).length === 0;

let seq = 0;
/** A fresh folder per case: its profile, its `--out`, its spec. */
function caseDir(name: string): string {
  const d = join(OUT, `${String(++seq).padStart(2, "0")}-${name}`);
  mkdirSync(d, { recursive: true });
  return d;
}

/** Two model-chrome stills of the mesh, so a fault can fall between them. */
function specIn(dir: string): string {
  const p = join(dir, "spec.json");
  writeFileSync(p, JSON.stringify({ spec: "apegmsh-view/1", files: { model: MODEL }, size: [640, 480], shots: [{ name: "a", chrome: "model", camera: "iso" }, { name: "b", chrome: "model", camera: "top" }] }));
  return p;
}

type Mode = "render" | "describe" | "capture";
const MODES: Mode[] = ["render", "describe", "capture"];

/** The arguments of one mode, run in `dir`; `out` is the folder (render, describe) or the PNG (capture) it may write. */
function argsOf(mode: Mode, dir: string): { out: string; args: string[] } {
  if (mode === "render") {
    const out = join(dir, "out");
    return { out, args: ["--mode=render", `--spec=${specIn(dir)}`, `--out=${out}`] };
  }
  if (mode === "describe") {
    const out = join(dir, "out");
    return { out, args: ["--mode=describe", `--file=${MODEL}`, `--out=${out}`] };
  }
  const out = join(dir, "still.png");
  return { out, args: ["--mode=capture", `--file=${MODEL}`, `--out=${out}`] };
}

/** A fault ends the run non-zero, promptly, with its reason on stderr and no modal. */
function failedPromptly(r: Ran, what: string, reason: RegExp): void {
  noModal(r, what);
  assert.equal(r.hung, false, `${what}: still running after ${LIMIT_MS / 1000} s (stopped by its PID): ${r.text}`);
  assert.notEqual(r.code, 0, `${what}: exit 0 after a fault: ${r.text}`);
  assert.ok(r.ms < PROMPT_MS, `${what}: took ${Math.round(r.ms)} ms`);
  assert.match(r.text, reason, `${what}: the reason on stderr`);
}

const REASON = {
  throw: /failed: an (uncaught error|unhandled rejection) in the main process: .*AGV_FAULT: a forced throw inside whenReady/,
  crash: /failed: the renderer process is gone \(crashed/,
  hang: /failed: the renderer stopped responding: its page left main's question unanswered for 5000 ms \(--hang-ms\)/,
};

// ---- normal runs: exit 0, and nothing outside --out ------------------------------------------

for (const mode of MODES) {
  if (!wanted(`${mode}.ok`)) continue;
  const dir = caseDir(`${mode}-ok`);
  const user = join(dir, "user");
  const { out, args } = argsOf(mode, dir);
  const r = await run(user, {}, ...args);
  noModal(r, `${mode} normal`);
  assert.equal(r.code, 0, `${mode}: a normal run exits 0: ${r.text}`);
  if (mode === "render") {
    assert.deepEqual(listing(out), ["a.png", "b.png", "manifest.json"]);
    const m = JSON.parse(readFileSync(join(out, "manifest.json"), "utf8")) as { shots: { status: string }[] };
    assert.deepEqual(m.shots.map((s) => s.status), ["ok", "ok"]);
  } else if (mode === "describe") {
    assert.deepEqual(listing(out), ["describe.json"]);
  } else {
    assert.ok(existsSync(out), `capture wrote ${out}`);
  }
  // INV-1 for the agent modes: their profile is under --out, so the run's own profile folder stays empty.
  if (mode !== "capture") assert.ok(emptyOrAbsent(user), `${mode}: INV-1: nothing written to the default profile folder ${user}: ${existsSync(user) ? readdirSync(user).join(", ") : ""}`);
  log(`${mode}: a normal run exits 0 in ${Math.round(r.ms)} ms${mode === "capture" ? "" : "; nothing outside --out"}`);
}

// ---- faults in each mode: non-zero, promptly, the reason on stderr --------------------------------

for (const fault of ["throw", "crash", "hang"] as const) {
  for (const mode of MODES) {
    if (!wanted(`${mode}.${fault}`)) continue;
    const dir = caseDir(`${mode}-${fault}`);
    const { out, args } = argsOf(mode, dir);
    // The watchdog's limit is the caller's: 5 s here, so a hang is seen well inside PROMPT_MS.
    const r = await run(join(dir, "user"), { AGV_FAULT: fault }, ...args, ...(fault === "hang" ? ["--hang-ms=5000"] : []));
    failedPromptly(r, `${mode} ${fault}`, REASON[fault]);
    assert.equal(r.code, 1, `${mode} ${fault}: exit 1, the run failed`);
    if (mode === "render") {
      // INV-6: no PNG stands without its entry, and the manifest says why the run stopped, and on which shot.
      assert.deepEqual(listing(out).filter((f) => f.endsWith(".png")), [], `${mode} ${fault}: no PNG`);
      const m = JSON.parse(readFileSync(join(out, "manifest.json"), "utf8")) as { refused: { key: string; reason: string } | null; shots: unknown[] };
      assert.equal(m.refused?.key, "render");
      assert.match(m.refused?.reason ?? "", /the run did not finish: [^]*; it stopped on shot a; none of the 2 stills had been written$/);
    }
    if (mode === "capture") assert.match(readFileSync(`${out}.error.txt`, "utf8"), REASON[fault], "capture: the reason in <out>.error.txt");
    log(`${mode} ${fault}: exit ${r.code} in ${Math.round(r.ms)} ms: ${(r.text.match(/apeGmshViewer: \w+ failed: [^\n]*/) ?? [""])[0].slice(0, 140)}`);
  }
}

// ---- render: INV-6 after a crash part way, and a partial write ---------------------------------

if (wanted("render.crash-after-a")) {
  const dir = caseDir("render-crash-after-a");
  const { out, args } = argsOf("render", dir);
  const r = await run(join(dir, "user"), { AGV_FAULT: "crash", AGV_AFTER: join(out, "a.png") }, ...args);
  failedPromptly(r, "render crash after a.png", REASON.crash);
  assert.equal(r.code, 1);
  assert.deepEqual(listing(out), ["manifest.json"], "INV-6: the still written before the crash is removed; only the manifest stays");
  const m = JSON.parse(readFileSync(join(out, "manifest.json"), "utf8")) as { refused: { key: string; reason: string }; shots: unknown[]; app: { version: unknown } };
  assert.equal(m.refused.key, "render");
  assert.match(m.refused.reason, /renderer process is gone/);
  assert.match(m.refused.reason, /1 of 2 stills had been written \(a\); they are removed/);
  assert.deepEqual(m.shots, []);
  assert.equal(typeof m.app.version, "string");
  log(`render crash after a.png: exit 1, a.png removed, manifest: ${m.refused.reason.slice(0, 160)}`);
}

// ---- capture: a throw inside the capture sequence (the D7 repro: exit 0 at 38 of 59 stills) -------------

if (wanted("capture.sequence")) {
  const dir = caseDir("capture-sequence");
  const { out, args } = argsOf("capture", dir);
  const r = await run(join(dir, "user"), { AGV_FAULT: "sequence" }, ...args);
  failedPromptly(r, "capture throw in the sequence", /capture failed: the page failed: .*inspector|capture failed: the page failed: .*null/);
  assert.equal(r.code, 1);
  const said = readFileSync(`${out}.error.txt`, "utf8");
  assert.match(said, /the page failed: /, "the reason in <out>.error.txt");
  assert.match(said, /the last still written was .*still\.png/, "and where the capture stopped");
  log(`capture throw in the sequence: exit 1 in ${Math.round(r.ms)} ms, ${out}.error.txt: ${said.split("\n")[0]!.slice(0, 140)}`);
}

// ---- measure-results: a file it cannot read ---------------------------------------------------

if (wanted("measure-results.bad")) {
  const dir = caseDir("measure-results-bad");
  const bad = join(dir, "garbage.results.h5");
  writeFileSync(bad, "not hdf5");
  const r = await run(join(dir, "user"), {}, "--mode=measure-results", `--file=${bad}`, `--out=${join(dir, "m.json")}`);
  failedPromptly(r, "measure-results on a file that is not HDF5", /measure-results failed: /);
  assert.equal(r.code, 1);
  log(`measure-results on garbage: exit 1 in ${Math.round(r.ms)} ms`);
}

// ---- the watchdog's limit is checked like any argument ------------------------------------------

if (wanted("hang-ms.bad")) {
  for (const v of ["0", "-5", "5s", ""]) {
    const dir = caseDir(`hang-ms-${v || "empty"}`);
    const { out, args } = argsOf("render", dir);
    const r = await run(join(dir, "user"), {}, ...args, `--hang-ms=${v}`);
    noModal(r, `--hang-ms=${v}`);
    assert.equal(r.code, 2, `--hang-ms=${v}: exit 2: ${r.text}`);
    assert.match(r.text, /--hang-ms=.*expected a whole number of milliseconds above 0/);
    assert.equal(existsSync(out), false, `--hang-ms=${v}: nothing written`);
  }
  log("--hang-ms: 0, -5, 5s and empty are refused (exit 2, nothing written)");
}

if (wanted("render.partial")) {
  const dir = caseDir("render-partial");
  const { out, args } = argsOf("render", dir);
  const r = await run(join(dir, "user"), { AGV_FAULT: "partial", AGV_AFTER: join(out, "b.png") }, ...args);
  noModal(r, "render partial write");
  assert.equal(r.code, 1, `a shot refused: exit 1: ${r.text}`);
  assert.deepEqual(listing(out), ["a.png", "manifest.json"], "the partial b.png is removed");
  const m = JSON.parse(readFileSync(join(out, "manifest.json"), "utf8")) as { shots: { name: string; status: string; png: string | null; reason: string | null }[] };
  const b = m.shots.find((s) => s.name === "b")!;
  assert.equal(b.status, "refused");
  assert.equal(b.png, null);
  assert.match(b.reason ?? "", /AGV_FAULT: ENOSPC/);
  assert.equal(m.shots.find((s) => s.name === "a")!.status, "ok");
  log(`render partial write of b.png: exit 1, b.png removed, b refused: ${(b.reason ?? "").slice(0, 120)}`);
}

// ---- the #2708 review: four ways the first cut still lied -----------------------------------------

/** Two app-chrome stills (main captures the page itself, `render:capture`). */
function appSpecIn(dir: string): string {
  const p = join(dir, "spec.json");
  writeFileSync(p, JSON.stringify({ spec: "apegmsh-view/1", files: { model: MODEL }, size: [640, 480], shots: [{ name: "a", chrome: "app", camera: "iso" }, { name: "b", chrome: "app", camera: "top" }] }));
  return p;
}

if (wanted("render.crash-mid-a")) {
  // 1: a crash while a's capture is in flight names shot a, and no still as written.
  const dir = caseDir("render-crash-mid-a");
  const out = join(dir, "out");
  const r = await run(join(dir, "user"), { AGV_FAULT: "crash-capture" }, "--mode=render", `--spec=${appSpecIn(dir)}`, `--out=${out}`);
  failedPromptly(r, "render crash during a's capture", REASON.crash);
  assert.equal(r.code, 1);
  assert.deepEqual(listing(out), ["manifest.json"]);
  const m = JSON.parse(readFileSync(join(out, "manifest.json"), "utf8")) as { refused: { key: string; reason: string } };
  assert.match(m.refused.reason, /; it stopped on shot a; none of the 2 stills had been written$/, `the shot in flight is named: ${m.refused.reason}`);
  log(`render crash during a's capture: exit 1, ${m.refused.reason.slice(-80)}`);
}

if (wanted("render.capture-after-end")) {
  // 1b (review round 2): a capture that resolves once the run is ending writes no PNG beside the refusal.
  const runs = Number(process.env["AGV_REPEAT"] ?? 1);
  for (let i = 0; i < runs; i++) {
    const dir = caseDir("render-capture-after-end");
    const out = join(dir, "out");
    const r = await run(join(dir, "user"), { AGV_FAULT: "crash-late" }, "--mode=render", `--spec=${appSpecIn(dir)}`, `--out=${out}`);
    failedPromptly(r, "render capture resolving after the end", REASON.crash);
    assert.equal(r.code, 1);
    assert.deepEqual(listing(out), ["manifest.json"], `run ${i + 1}: no PNG written after the manifest refused the run`);
    const m = JSON.parse(readFileSync(join(out, "manifest.json"), "utf8")) as { refused: { reason: string } };
    assert.match(m.refused.reason, /; it stopped on shot a; none of the 2 stills had been written$/);
  }
  log(`render capture resolving after the run began ending: ${runs} run(s), exit 1, only manifest.json each time`);
}

if (wanted("describe.main-block")) {
  // 2: main blocked for 8 s with --hang-ms=5000 is main's stall, not a hung renderer: the run finishes, exit 0.
  const dir = caseDir("describe-main-block");
  const { out, args } = argsOf("describe", dir);
  const r = await run(join(dir, "user"), { AGV_FAULT: "main-block" }, ...args, "--hang-ms=5000");
  noModal(r, "describe main-block");
  assert.doesNotMatch(r.text, /renderer stopped responding/, `main's own stall blamed on the renderer: ${r.text}`);
  assert.equal(r.code, 0, `describe with main blocked 8 s: exit 0: ${r.text}`);
  assert.deepEqual(listing(out), ["describe.json"]);
  log(`describe with main blocked 8 s (--hang-ms=5000): exit 0 in ${Math.round(r.ms)} ms, no hang reported`);
}

if (wanted("capture.empty")) {
  // 3: capturePage returning an empty image is a failed capture: exit 1, the reason in <out>.error.txt, no still.
  const dir = caseDir("capture-empty");
  const { out, args } = argsOf("capture", dir);
  const r = await run(join(dir, "user"), { AGV_FAULT: "empty" }, ...args);
  failedPromptly(r, "capture of an empty image", /capture failed: capturePage returned an empty image/);
  assert.equal(r.code, 1);
  assert.equal(existsSync(out), false, "no still.png");
  assert.match(readFileSync(`${out}.error.txt`, "utf8"), /capturePage returned an empty image/);
  log(`capture of an empty image: exit 1 in ${Math.round(r.ms)} ms, reason in ${out}.error.txt`);
}

if (wanted("measure.read-row")) {
  // 4: measure's .ladruno read row failing exits 1, not 0.
  const dir = caseDir("measure-read-row");
  const model = join(dir, "m.h5");
  writeFileSync(model, readFileSync(MODEL));
  const r = await run(join(dir, "user"), { AGV_FAULT: "measure-read", AGV_AFTER: join(dir, "m.ladruno") }, "--mode=measure", `--file=${model}`, `--out=${join(dir, "m.json")}`);
  failedPromptly(r, "measure with a .ladruno read row that fails", /measure failed: the \.ladruno read row: /);
  assert.equal(r.code, 1);
  log(`measure with a failing .ladruno read row: exit 1 in ${Math.round(r.ms)} ms`);
}

log(`headless exit e2e: all checks passed; runs in ${OUT}`);
process.exit(0);
