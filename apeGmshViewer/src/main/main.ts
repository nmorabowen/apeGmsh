// Electron main process. It reads files and nothing else (ADR 0112 D4):
// no network, no writes except the measurement / capture outputs a script
// asked for, and no child process except the user's editor for go-to-source
// (D3, ./source.ts). The one socket is a local pipe, private to one agent call:
// the running window's acknowledgement of `--present` or `--snapshot`.
//
// Modes (argv, set by scripts/launch.mjs):
//   --mode=view     open a window on --file, or on the first positional
//                   argument (a double-clicked file), or an empty drop target
//   --mode=measure  open, orbit, pick, write a JSON measurement to --out
//   --mode=capture  open hidden, select a beam, write a PNG still to --out
//                   (and, with a results sibling, one still per results stage)
//   --mode=measure-results  time the results.h5 read path (V4a, #1707; the
//                   block at the end of this file), write a JSON to --out and
//                   print a MEASUREMENTS row. Started directly, after
//                   `npm run build`: `electron . --mode=measure-results
//                   --file=<x.results.h5> --out=<json>`, or with an `.mpco`
//                   file and `--dataset=<path>` for the raw MPCO point.
//   --mode=render   ADR 0128 D4: take every shot of the view spec `--spec`
//                   in a hidden window; write `<name>.png` per shot and
//                   `manifest.json` into the folder `--out`, then exit (0 every
//                   shot drawn, 1 a shot refused or the run failed, 2 the
//                   spec refused: no window opens and no PNG is written).
//                   An app-chrome still is captured until two consecutive
//                   captures match, at most `--capture-tries` (default 10)
//   --mode=describe ADR 0128 D4: write `describe.json` into `--out`: what the
//                   set of `--file` offers a spec (stages, steps, fields, groups)
//   --mode=present  ADR 0128 D5: offer the view spec `--spec` to the running
//                   window over the second-instance path, which lists it and
//                   changes nothing else (INV-5). Exit 0 only when the window
//                   acknowledged it, 1 no window answered (nothing opens), 2 the
//                   spec refused
//   --mode=snapshot ADR 0128 D6: ask the running window for its view; it writes
//                   `view.json`, `view.png` and `manifest.json` into the folder
//                   `--out`, the manifest last. Exit 0 written, 1 no window is
//                   running or it could not read back, 2 `--out` refused
// The ADR's spellings work too: `--render spec.json --out d`, `--describe f --out d`,
// `--present spec.json`, `--snapshot --out d` (src/spec/snapshot.ts `agentArgsOf`);
// a request for an agent mode that cannot be carried out exits 2, never view mode.
// No mode but view hangs (VA2h, #2665): an uncaught error, a failure the page
// reports, a crashed renderer or one that stops answering for `--hang-ms`
// (default 60000) exits 1 with the reason on stderr; a render's manifest.json
// then refuses the run (key `render`) and keeps no PNG without its entry, and a
// capture writes the reason to `<out>.error.txt`.
// Neither agent mode takes the single-instance lock, so both run beside the
// human's window and never reach it (ADR 0128 INV-5). Both refuse an `--out`
// that is not empty (exit 2, nothing written), and keep Chromium's profile
// and crash dumps in `<out>/.chromium` (INV-1: nothing written outside it).
//
// In view mode the app is single-instance: a second launch forwards its file
// here. Opening a file opens its set (./pairing.ts), and the set is watched
// (./watch.ts). The renderer hears of a new set or a rewritten file through
// `onOpen` / `onFileChanged` once it subscribes; a renderer that has not
// subscribed (the P0 page) is reloaded instead, and reads the new model from
// `app:config`.

import { app, BrowserWindow, dialog, ipcMain, type IpcMainEvent, type WebContents } from "electron";
import { randomUUID } from "node:crypto";
import { connect, createServer } from "node:net";
import { existsSync, mkdirSync, readdirSync, readFileSync, realpathSync, renameSync, rmSync, statSync, unwatchFile, watchFile, writeFileSync } from "node:fs";
import { cpus, tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { inspect } from "node:util";
import * as h5wasm from "h5wasm/node";
import { openModel } from "../reader/node.ts";
import { openH5, type H5Openable } from "../reader/h5open.ts";
import { LadrunoFile, openLadruno } from "../reader/ladruno.ts";
import { openResults, type ResultsFile, type ResultsH5 } from "../reader/results.ts";
import { readUserFile, readUserSet, type UserH5 } from "../reader/userFile.ts";
import { readCaseValues, SPECTRA_FAMILIES, type CaseRead, type SpectrumTarget } from "../reader/userSpectra.ts";
import { candidates, classify, FOLD_CASE, fileFromArgv, listDir, pairSet, samePath, type Kind, type OpenSet } from "./pairing.ts";
import { compareStagesOf, ResultsHost, type ScanRequest, type StepRequest } from "./results.ts";
import { checkPlan } from "../state/extrapolation.ts";
import { OpenSession, Study } from "./session.ts";
import { confinedToStudy, isManifestPath, readVariants, resolveRow, type VariantsFs } from "../reader/variants.ts";
import { goToSource, provenanceSources, type SourceScope } from "./source.ts";
import { SetWatcher } from "./watch.ts";
import { openGeometry, openLineage, openProvenance, openStageElements } from "./zones.ts";
import { resolveSpecFiles } from "../spec/files.ts";
import { outRefusal, prepareRender, PROFILE_DIR, type JobFs, type RenderJobFiles } from "../spec/job.ts";
import { MANIFEST_FILE, refusedManifest } from "../spec/manifest.ts";
import { agentArgsOf, presentDocument, SNAPSHOT_PNG, SNAPSHOT_VIEW, type OfferPayload, type PresentedDocument } from "../spec/snapshot.ts";
import type { OfferIn } from "../state/types.ts";

const here = dirname(fileURLToPath(import.meta.url));

// Unpackaged, argv[1] is the app path; packaged, the file follows the exe.
const argvSkip = process.defaultApp ? 2 : 1;
/** The ADR's spellings as `--mode=` arguments; null when the command line uses none. */
const spelled = agentArgsOf(process.argv.slice(argvSkip));
const argv = spelled && "argv" in spelled ? [...spelled.argv, ...process.argv] : process.argv;

function arg(name: string): string | undefined {
  const pre = `--${name}=`;
  return argv.find((a) => a.startsWith(pre))?.slice(pre.length);
}

const MODES = ["view", "measure", "capture", "measure-results", "render", "describe", "present", "snapshot"] as const;
/** Why this command line is refused before anything starts: said on stderr, exit 2. Never a thrown error (a modal), never view mode. */
const badArgs =
  spelled && "refused" in spelled
    ? spelled.refused
    : !(MODES as readonly string[]).includes(arg("mode") ?? "view")
      ? `unknown --mode=${arg("mode")}; expected ${MODES.join(", ")}`
      : arg("hang-ms") !== undefined && !/^[1-9][0-9]*$/.test(arg("hang-ms")!)
        ? `--hang-ms=${arg("hang-ms")}: expected a whole number of milliseconds above 0`
        : arg("capture-tries") !== undefined && !(/^[1-9][0-9]*$/.test(arg("capture-tries")!) && Number(arg("capture-tries")) >= 2)
          ? `--capture-tries=${arg("capture-tries")}: expected a whole number of captures, 2 or more`
          : null;
const mode = badArgs === null ? (arg("mode") ?? "view") : "refused";
if (badArgs !== null) {
  process.stderr.write(`apeGmshViewer: ${badArgs}\n`);
  app.exit(2);
}
/** ADR 0128 D4: the two agent modes; `out` is a folder for them, and they write nothing outside it (INV-1). */
const agentMode = mode === "render" || mode === "describe";
/** ADR 0128 D5, D6: these reach the window already running, and never open one. */
const reachMode = mode === "present" || mode === "snapshot";
const file = mode === "view" ? fileFromArgv(process.argv, argvSkip, process.cwd()) : arg("file");
const out = arg("out");
const t0 = Number(arg("t0") ?? Date.now());

// ---------------------------------------------------------------------------
// VA2h (#2665): no mode but view can hang. Electron answers an uncaught error
// in main with a modal dialog, and a crashed or hung renderer leaves the
// hidden window's process alive, so a caller waiting on the exit code would
// wait forever. Every other mode ends in `headlessEnd`: the reason on stderr,
// exit 1. A render also leaves its `--out` true to INV-6 (`abandonRender`),
// and a capture writes the reason to `<out>.error.txt`.
//
// Electron's `unresponsive` needs input the renderer leaves unanswered, which
// a hidden window never gets, so main also asks each window's page a trivial
// question every second; no answer for `--hang-ms` (default 60 s) ends the run.
// A page that is merely slow still answers between its tasks.
// ---------------------------------------------------------------------------

/** Every mode but view runs for a caller that waits on its exit code. */
const headless = mode !== "view";
/** Set once the run is ending, so a late event (a renderer killed by the exit itself) changes nothing. */
let ending = false;
const errText = (err: unknown) => (err instanceof Error ? (err.stack ?? err.message) : String(err));
/** How long a page may leave main's question unanswered (`--hang-ms`). */
const HANG_MS = Number(arg("hang-ms") ?? 60000);
/** The last still a capture wrote, to name where a failed capture stopped. */
let lastCaptured: string | null = null;

/** Where the run stopped, for its refusal: the shot in progress (render) or the last still written (capture). */
function whereStopped(): string {
  if (mode === "render" && renderJob) {
    // A shot taken from `unwritten` but not yet in `written` (its capture in flight) is the one in progress.
    const next = renderJob.spec.shots.find((s) => !written.has(s.name));
    return next ? `it stopped on shot ${next.name}` : "every still had been written";
  }
  if (mode === "capture") return lastCaptured ? `the last still written was ${lastCaptured}` : "no still had been written";
  return "";
}

function headlessEnd(reason: string): void {
  if (ending) return;
  ending = true;
  let where = "";
  try {
    where = whereStopped();
  } catch {
    // Before the job is known: nothing was drawn.
  }
  const full = where ? `${reason}; ${where}` : reason;
  const line = `apeGmshViewer: ${mode} failed: ${full}\n`;
  process.stderr.write(line);
  try {
    if (mode === "render") abandonRender(full);
    if (mode === "capture" && out) writeFileSync(`${out}.error.txt`, line);
  } catch (err) {
    process.stderr.write(`apeGmshViewer: ${mode}: the failure could not be recorded in --out: ${errText(err)}\n`);
  }
  // Out of the failing callback, and the windows gone first: one run that exited from inside it ended in an access violation, not 1.
  setImmediate(() => {
    for (const w of BrowserWindow.getAllWindows()) w.destroy();
    app.exit(1);
  });
}

const TICK_MS = 1000;

/**
 * Ask the page a trivial question every second; a page that leaves one
 * question unanswered for HANG_MS is hung. The wait is timed from when the
 * question was sent, and a tick that itself fires late (main was busy, so the
 * answer may sit unread in main's queue) gives no verdict: it restarts the
 * wait, so main's own stall is never blamed on the renderer (#2708 review).
 */
function watchPage(w: BrowserWindow): void {
  let asking = false;
  let sentAt = 0;
  let lastTick = Date.now();
  const timer = setInterval(() => {
    const now = Date.now();
    const late = now - lastTick > 2 * TICK_MS;
    lastTick = now;
    if (w.isDestroyed() || ending) {
      clearInterval(timer);
      return;
    }
    if (late) {
      if (asking) sentAt = now;
      return;
    }
    if (asking) {
      if (now - sentAt > HANG_MS) {
        clearInterval(timer);
        headlessEnd(`the renderer stopped responding: its page left main's question unanswered for ${HANG_MS} ms (--hang-ms)`);
      }
      return;
    }
    asking = true;
    sentAt = now;
    w.webContents.executeJavaScript("0").then(
      () => {
        asking = false;
      },
      () => {
        asking = false;
      },
    );
  }, TICK_MS);
}

if (headless) {
  // A listener of its own also stops Electron's default handler, which is the modal.
  process.on("uncaughtException", (err) => headlessEnd(`an uncaught error in the main process: ${errText(err)}`));
  process.on("unhandledRejection", (err) => headlessEnd(`an unhandled rejection in the main process: ${errText(err)}`));
  app.on("render-process-gone", (_e, _contents, details) => headlessEnd(`the renderer process is gone (${details.reason}, exit code ${details.exitCode})`));
  app.on("browser-window-created", (_e, w) => {
    w.on("unresponsive", () => headlessEnd("the renderer stopped responding (Electron's unresponsive)"));
    watchPage(w);
  });
}

// Measurement of GPU throughput: lift the vsync and frame-rate caps.
if (arg("uncapped") === "1") {
  app.commandLine.appendSwitch("disable-frame-rate-limit");
  app.commandLine.appendSwitch("disable-gpu-vsync");
}

if ((mode === "measure" || mode === "capture" || mode === "measure-results") && (!file || !out)) {
  throw new Error(`--mode=${mode} needs --file and --out`);
}
// A still is the spec's `size` in pixels, whatever the display's scale (ADR 0128 D9: a model still is exactly `size`).
if (agentMode) app.commandLine.appendSwitch("force-device-scale-factor", "1");

// ---------------------------------------------------------------------------
// VU10 (ADR 0132 D8, D9; owner ruling option A, #2753): the study. A manifest
// is read only from a path the user gave (a drop, argv, a double-click, the
// render spec's `files.variants`), and its rows' paths stay here: the page
// names a variant by row index and label. The compare set has a slot and a
// results host of its own, apart from the window set and its `resultsHost`.
// ---------------------------------------------------------------------------

const variantsFs: VariantsFs = {
  readFile: (path) => readFileSync(path, "utf8"),
  isFile: (path) => {
    try {
      return statSync(path).isFile();
    } catch {
      return false;
    }
  },
  realpath: (path) => realpathSync.native(path),
  fold: FOLD_CASE,
};

const study = new Study({
  read: (manifest) => readVariants(manifest, variantsFs),
  // Only the human's window follows a rewrite (D9.3); an agent run reads the spec's manifest once.
  watch: (manifest, changed) => {
    if (mode !== "view") return { close: () => undefined };
    const listener = (cur: { mtimeMs: number; size: number }, prev: { mtimeMs: number; size: number }) => {
      if (cur.mtimeMs !== prev.mtimeMs || cur.size !== prev.size) changed();
    };
    watchFile(manifest, { interval: 500 }, listener);
    return { close: () => unwatchFile(manifest, listener) };
  },
  note: (message) => process.stderr.write(`apeGmshViewer: ${message}\n`),
  recheck: (manifest, row) => resolveRow(manifest, row.path, variantsFs),
  confined: (manifest, path) => confinedToStudy(manifest, path, variantsFs),
});

/**
 * D9.4: a row's set as main opens it: paired from the row's resolved model,
 * each sibling (results, geometry, user) confined to the manifest's folder
 * again, or why not.
 */
function rowSet(open: string): { ok: true; set: OpenSet } | { ok: false; reason: string } {
  let set: OpenSet;
  try {
    set = pairSet(open, existsSync);
  } catch (err) {
    return { ok: false, reason: err instanceof Error ? err.message : String(err) };
  }
  for (const p of [set.results, set.geometry, set.user]) {
    const why = study.confine(p);
    if (why !== null) return { ok: false, reason: `a file of the variant's set leaves the study: ${why}` };
  }
  return { ok: true, set };
}

/**
 * D9.5: the compare set's results host, apart from the window set's
 * `resultsHost`: `results:open` and `results:close` never touch it, and only
 * `variants:compare`, `compare:readStep` and `compare:close` read or close it.
 */
const compareHost = new ResultsHost(h5wasm as unknown as ResultsH5, h5wasm.ready);

const jobFs: JobFs = {
  readFile: (path) => readFileSync(path, "utf8"),
  exists: existsSync,
  isDir: (path) => statSync(path).isDirectory(),
  list: (path) => readdirSync(path),
  mkdir: (path) => mkdirSync(path, { recursive: true }),
  writeFile: (path, text) => writeFileSync(path, text),
};

/**
 * ADR 0128 D2 startup of an agent mode: its arguments, its `--out` (absent
 * or an empty folder: spec/job.ts), Chromium's profile under `--out`
 * (INV-1), then the render job. Any refusal or unexpected error goes to
 * stderr and exits 2 at once, before any window, dialog or Chromium write:
 * a thrown error at this level would open a modal and hang the run, with the
 * profile still in %APPDATA% (review of #2660).
 */
function agentStartup(): { refused: boolean; job: RenderJobFiles | null } {
  const refuse = (message: string) => {
    process.stderr.write(`apeGmshViewer: ${message}\n`);
    app.exit(2);
    return { refused: true, job: null };
  };
  try {
    if (mode === "describe" && (!file || !out)) return refuse("--mode=describe needs --file and --out");
    if (mode === "render" && (!arg("spec") || !out)) return refuse("--mode=render needs --spec and --out");
    const outDir = resolve(out!);
    const busy = outRefusal(outDir, jobFs);
    if (busy) return refuse(`--out refused: ${busy}`);
    // INV-1: an agent run writes nothing outside `--out`, Chromium's profile and crash dumps included.
    const profile = join(outDir, PROFILE_DIR);
    app.setPath("userData", profile);
    app.setPath("crashDumps", join(profile, "crashdumps"));
    if (mode !== "render") return { refused: false, job: null };
    const specPath = resolve(arg("spec")!);
    // VU10: the spec's manifest is the study, and its rows must include the spec's model (the window variant).
    const p = prepareRender(specPath, outDir, jobFs, (manifest, model) => {
      const r = study.open(manifest);
      if (!r.ok) return r.reason;
      return study.rowOfModel(model) ? null : `${manifest} has no variant row whose model is ${model}; its rows are ${r.rows.map((x) => JSON.stringify(x.label)).join(", ") || "none"}`;
    });
    // D2: the whole spec is refused before any shot: no window, no PNG, the key named.
    if (!p.ok) return refuse(`the spec ${specPath} is refused: ${p.key}: ${p.reason}`);
    return { refused: false, job: p.job };
  } catch (err) {
    return refuse(`${mode} failed to start: ${err instanceof Error ? err.message : String(err)}`);
  }
}

const startup = agentMode ? agentStartup() : { refused: false, job: null };
/** An agent mode refused at startup: no window opens (it has exited with code 2). */
const outBusy = startup.refused;
/** The render job: the validated spec and its set; null when the spec was refused (manifest.json names the key). */
const renderJob = startup.job;

// View mode is single-instance: a second launch hands its argv to the first
// (the "second-instance" handler below) and exits.
if (mode === "view" && !app.requestSingleInstanceLock()) app.exit(0);

/** What `--present` and `--snapshot` hand the running window (the lock's additional data); `ack` is the sender's private pipe. */
type Reach = { kind: "present"; spec: string; ack: string } | { kind: "snapshot"; out: string; ack: string };
/** The window's answer on that pipe. */
type Ack = { ok: true } | { ok: false; code: number; reason: string };

/** A spec file as offers, or as refused rows; never a throw (it runs in the human's window). */
function presentOffers(specPath: string): PresentedDocument {
  return presentDocument(() => readFileSync(specPath, "utf8"), (spec) => resolveSpecFiles(spec, dirname(specPath), existsSync), FOLD_CASE);
}

/** How long the sender waits for the window to acknowledge: a window's main process answers at once, so silence means no window took it. */
const REACH_ACK_MS = 5000;

/** The sender's end of the acknowledgement, listening before the window is asked. */
function ackPipe(): Promise<{ name: string; answer: Promise<Ack | null> }> {
  const id = `apegmsh-viewer-${randomUUID()}`;
  const name = process.platform === "win32" ? `\\\\.\\pipe\\${id}` : join(tmpdir(), `${id}.sock`);
  return new Promise((listening, failed) => {
    let settle: (a: Ack | null) => void = () => {};
    const answer = new Promise<Ack | null>((r) => (settle = r));
    setTimeout(() => settle(null), REACH_ACK_MS).unref();
    const server = createServer((socket) => {
      let text = "";
      socket.setEncoding("utf8");
      socket.on("data", (d) => (text += String(d)));
      socket.on("end", () => settle(JSON.parse(text) as Ack));
      socket.on("error", () => settle(null));
    });
    server.on("error", failed);
    server.listen(name, () => listening({ name, answer }));
  });
}

/** Tell the sender what became of its call. A sender that has gone is no matter for the human's window. */
function sendAck(pipe: string, ack: Ack): void {
  const c = connect(pipe, () => c.end(JSON.stringify(ack)));
  c.on("error", (err) => process.stderr.write(`apeGmshViewer: acknowledging an agent's call: ${err.message}\n`));
}

/** How long `--snapshot` waits for the window's manifest. */
const SNAPSHOT_WAIT_MS = 60000;

/** Wait for the manifest the running window writes last; exit with what it says. */
function awaitSnapshot(dir: string): Promise<never> {
  const path = join(dir, MANIFEST_FILE);
  const t = Date.now();
  return new Promise<never>(() => {
    const timer = setInterval(() => {
      if (!existsSync(path)) {
        if (Date.now() - t < SNAPSHOT_WAIT_MS) return;
        clearInterval(timer);
        process.stderr.write(`apeGmshViewer: snapshot: the window did not answer within ${SNAPSHOT_WAIT_MS / 1000} s\n`);
        app.exit(1);
        return;
      }
      clearInterval(timer);
      const m = JSON.parse(readFileSync(path, "utf8")) as { refused?: { reason: string } };
      if (m.refused) process.stderr.write(`apeGmshViewer: snapshot refused: ${m.refused.reason}\n`);
      else process.stdout.write(`apeGmshViewer: snapshot: ${path}\n`);
      app.exit(m.refused ? 1 : 0);
    }, 50);
  });
}

/**
 * ADR 0128 D5, D6 startup: check what is asked here (exit 2, nothing sent),
 * then reach the running window through the lock. Holding the lock means no
 * window is running: let go of it, say so, exit 1, and open nothing. Denied the
 * lock proves nothing (another agent call may hold it for a moment): only the
 * window's acknowledgement is an exit 0. The module's top level awaits this,
 * and Electron holds `ready` until it ends, so Chromium never starts here and
 * nothing is written beside the human's profile (INV-1).
 */
async function reachWindow(): Promise<void> {
  const refuse = (code: number, message: string) => {
    process.stderr.write(`apeGmshViewer: ${message}\n`);
    app.exit(code);
  };
  try {
    let what: { kind: "present"; spec: string } | { kind: "snapshot"; out: string };
    if (mode === "present") {
      const spec = arg("spec");
      if (!spec) return refuse(2, "--mode=present needs --spec");
      const specPath = resolve(spec);
      const p = presentOffers(specPath);
      if (p.refused) return refuse(2, `the spec ${specPath} is refused: ${p.refused.key}: ${p.refused.reason}`);
      what = { kind: "present", spec: specPath };
    } else {
      if (!out) return refuse(2, "--mode=snapshot needs --out");
      const busy = outRefusal(resolve(out), jobFs);
      if (busy) return refuse(2, `--out refused: ${busy}`);
      what = { kind: "snapshot", out: resolve(out) };
    }
    const ack = await ackPipe();
    if (app.requestSingleInstanceLock({ ...what, ack: ack.name })) {
      app.releaseSingleInstanceLock();
      return refuse(1, "no apeGmshViewer window is running; none was opened. Ask the human to open one.");
    }
    const said = await ack.answer;
    if (said === null) return refuse(1, `no apeGmshViewer window acknowledged within ${REACH_ACK_MS / 1000} s; none was opened. Ask the human to open one.`);
    if (!said.ok) return refuse(said.code, said.reason);
    if (what.kind === "present") {
      process.stdout.write(`apeGmshViewer: present: the window took ${what.spec}\n`);
      app.exit(0);
      return;
    }
    await awaitSnapshot(what.out);
  } catch (err) {
    refuse(2, `${mode} failed to start: ${err instanceof Error ? err.message : String(err)}`);
  }
}
if (reachMode) await reachWindow();

let win: BrowserWindow | null = null;
/** Renderer subscriptions ("open", "fileChanged"); a page load clears them. */
const subscribed = new Set<string>();

/** While an offer opens its set, what would be a dialog is collected for the offer's refusal instead. */
let quietFails: string[] | null = null;

function fail(message: string): void {
  process.stderr.write(`apeGmshViewer: ${message}\n`);
  if (quietFails) {
    quietFails.push(message);
    return;
  }
  if (mode === "view") void app.whenReady().then(() => dialog.showErrorBox("apeGmshViewer", message));
}

/** The renderer hears `channel`; not subscribed, it reloads and re-reads config. */
function deliver(channel: "open" | "fileChanged", payload: unknown): void {
  if (!win) return;
  if (subscribed.has(channel)) win.webContents.send(`viewer:${channel}`, payload);
  else win.webContents.reload();
}

/** The open set and its watcher (view mode only). */
/** The newest watcher: a user file that appears or goes is a `changed`, not a reopen, so only the watcher's set knows the user slot (`viewer:currentSet`). */
let liveWatcher: SetWatcher | null = null;
const session = new OpenSession({
  exists: existsSync,
  watch: (opened, sink) => (liveWatcher = new SetWatcher(opened, sink).start()),
  deliver,
  fail,
  note: (message) => process.stderr.write(`apeGmshViewer: ${message}\n`),
});
/**
 * Open what the user gave. VU10: a `<study>.variants.json` opens as the study,
 * and its first row that passes as the window set (D9.1); any other file
 * opens its set as before, and a file that is no row of the study ends it.
 */
function openPath(path: string, notify: boolean): boolean {
  const abs = resolve(path);
  if (isManifestPath(abs)) return openStudy(abs, notify);
  const ok = session.open(abs, notify);
  if (ok && study.manifest !== null && study.rowOfModel(session.set?.model ?? null) === null) {
    closeCompare();
    study.close();
  }
  sendStudy();
  return ok;
}

function openStudy(manifest: string, notify: boolean): boolean {
  const r = study.open(manifest);
  if (!r.ok) {
    fail(r.reason);
    return false;
  }
  compareHost.close();
  const first = r.rows[0];
  if (!first) {
    sendStudy();
    fail(`${manifest}: no variant can be opened: ${r.refused.map((x) => x.reason).join("; ")}`);
    return false;
  }
  const now = study.row(first.index, first.label);
  const set = now.ok ? rowSet(now.open) : now;
  if (!now.ok || !set.ok) {
    sendStudy();
    fail(`${manifest}: ${"reason" in set ? set.reason : ""}`);
    return false;
  }
  const ok = session.open(now.open, notify);
  sendStudy();
  return ok;
}

if (mode === "view") {
  if (file) openPath(file, false);
  app.on("second-instance", (_e, argv, cwd, data) => {
    // INV-5: an agent's call lists or reads. It opens nothing, restores nothing, and takes no focus.
    const kind = (data as { kind?: unknown } | null)?.kind;
    if (kind !== undefined) {
      heardFromAgent(data as Reach);
      return;
    }
    const next = fileFromArgv(argv, argvSkip, cwd);
    if (next) openPath(next, true);
    if (win) {
      if (win.isMinimized()) win.restore();
      win.focus();
    }
  });
  // macOS delivers a double-clicked file as an event, not in argv.
  app.on("open-file", (e, path) => {
    e.preventDefault();
    openPath(path, app.isReady());
  });
}

// ---------------------------------------------------------------------------
// ADR 0128 D5, D6 in the running window. Offers wait until the page listens;
// they never reload it (INV-5). A snapshot captures the page as it is, then
// asks the page for the spec and the manifest entry of that view; main writes
// every file, into the `--out` the caller named (INV-1), the manifest last.
// ---------------------------------------------------------------------------

let offersReady = false;
type OfferRow = { files: string[]; left: string[][] };
/** The batches the page has not heard yet, each with its rows of `offerRows` (VU14: a page load keeps only these). */
let offersWaiting: { offers: OfferIn[]; rows: OfferRow[] }[] = [];
/**
 * VU12 (#2753): the files `viewer:offerOpen` may open, all chosen by main, one
 * row per offer row main sent in `viewer:offers`: the offer's model and results
 * (none for a refused row), and each set the window left when that offer
 * opened (its Back reopens that set). VU13 (#2760): the rows mirror the page's
 * list (state/reduce.ts `offerReceived`: newest first, at most `MAX_OFFERS`),
 * so an offer that falls off the page's list is gone here too, and its files
 * and the sets it left are no longer offerable. VU14 (#2771): a page load
 * starts an empty list, so it keeps only the rows of batches still waiting to
 * be sent (`pageLoading`).
 */
let offerRows: OfferRow[] = [];
/** state/reduce.ts `MAX_OFFERS`, the page's longest offer list (test/main/setDoors.test.ts holds the two equal). */
const PAGE_OFFERS = 8;
let snapshotSeq = 0;
const snapshotReplies = new Map<number, (reply: SnapshotReply) => void>();
type SnapshotReply = { ok: true; view: unknown; manifest: RunManifest } | { ok: false; reason: string };
interface RunManifest {
  timing: object;
}
/** How long the page has to answer a snapshot. */
const SNAPSHOT_REPLY_MS = 10000;

/** Nothing an agent sends may raise a modal in the human's window: every failure here goes to stderr and to the sender. */
function heardFromAgent(reach: Reach): void {
  try {
    if (reach.kind === "present") {
      const p = presentOffers(reach.spec);
      const rows = p.offers.map((o): OfferRow => {
        if (o.refused !== undefined) return { files: [], left: [] }; // a refused row is listed, never applied, and carries no files
        const { model, results } = (o.payload as OfferPayload).files;
        return { files: [model, ...(results === null ? [] : [results])], left: [] };
      });
      offerRows = [...rows, ...offerRows].slice(0, PAGE_OFFERS);
      if (win && offersReady) win.webContents.send("viewer:offers", p.offers);
      else offersWaiting.push({ offers: p.offers, rows });
      // The sender checked this file a moment ago. Changed since, it is listed as refused, and the sender hears why.
      sendAck(reach.ack, p.refused ? { ok: false, code: 2, reason: `the spec ${reach.spec} is refused: ${p.refused.key}: ${p.refused.reason}` } : { ok: true });
      return;
    }
    if (reach.kind === "snapshot") {
      sendAck(reach.ack, { ok: true });
      void writeSnapshot(reach.out).catch((err: unknown) => {
        const reason = err instanceof Error ? err.message : String(err);
        process.stderr.write(`apeGmshViewer: snapshot: ${reason}\n`);
        try {
          writeManifestLast(reach.out, refusedManifest("--snapshot", "snapshot", reason, { version: app.getVersion(), version_reason: null }, new Date().toISOString()));
        } catch {
          // No folder to write in: the sender's wait ends non-zero.
        }
      });
      return;
    }
    const unknown: never = reach;
    process.stderr.write(`apeGmshViewer: an unknown agent call reached the window: ${shown(unknown)}\n`);
  } catch (err) {
    process.stderr.write(`apeGmshViewer: an agent's call failed: ${err instanceof Error ? (err.stack ?? err.message) : String(err)}\n`);
  }
}

/**
 * VU15 (#2776): the page that may say it listens for offers, as Electron names
 * a sender: the window's main frame as the current load committed it
 * (`did-navigate`). A load (`did-start-loading`, or a new document's
 * `did-start-navigation` in the main frame) clears it until the new page
 * commits, because until then the page on screen is the one being replaced;
 * a page that commits in a new frame has a new process or frame id. So a late
 * `viewer:offersReady` from a page being replaced re-arms nothing.
 */
let pageFrame: { processId: number; routingId: number } | null = null;

function pageCommitted(contents: WebContents): void {
  const f = contents.mainFrame;
  pageFrame = { processId: f.processId, routingId: f.routingId };
}

/** Whether `e` comes from the current load of the window's page (VU15). */
function fromCurrentPage(e: IpcMainEvent): boolean {
  return win !== null && e.sender === win.webContents && pageFrame !== null && e.processId === pageFrame.processId && e.frameId === pageFrame.routingId;
}

ipcMain.on("viewer:offersReady", (e: IpcMainEvent) => {
  if (!fromCurrentPage(e)) {
    process.stderr.write(`apeGmshViewer: viewer:offersReady from a page that is not the window's current load (process ${shown(e.processId)}, frame ${shown(e.frameId)}) is ignored\n`);
    return;
  }
  offersReady = true;
  for (const { offers } of offersWaiting) win?.webContents.send("viewer:offers", offers);
  offersWaiting = [];
});

/**
 * A page load: the old page's subscriptions go, and so does every offer row the
 * new page will not be sent (VU14). VU10: the compare slot goes too, because the
 * new page starts with no compare set, and a slot it never opened is not one it
 * may read (#2817 review 4).
 */
function pageLoading(): void {
  closeCompare();
  subscribed.clear();
  offersReady = false;
  pageFrame = null;
  const waiting = new Set(offersWaiting.flatMap((b) => b.rows));
  offerRows = offerRows.filter((r) => waiting.has(r));
}

/** ADR 0128 D3: the app's version is main's to know; `written_at` lies inside `timing`, the one part that varies between runs (src/spec/manifest.ts). */
function stamped(m: RunManifest) {
  const { timing, ...rest } = m;
  return { ...rest, app: { version: app.getVersion() }, timing: { ...timing, written_at: new Date().toISOString() } };
}

/** The manifest last, and whole: its presence is what the waiting `--snapshot` process reads as done. */
function writeManifestLast(dir: string, text: string): void {
  const path = join(dir, MANIFEST_FILE);
  writeFileSync(`${path}.tmp`, text);
  renameSync(`${path}.tmp`, path);
}

async function writeSnapshot(dir: string): Promise<void> {
  mkdirSync(dir, { recursive: true });
  const refuse = (reason: string) =>
    writeManifestLast(dir, refusedManifest("--snapshot", "snapshot", reason, { version: app.getVersion(), version_reason: null }, new Date().toISOString()));
  if (!win || !offersReady) return refuse("the window's page is not ready");
  const contents = win.webContents;
  const img = await contents.capturePage();
  const size = img.getSize();
  if (img.isEmpty() || size.width === 0) return refuse("the window could not be captured (it may be minimized); it was left as it is");
  const id = ++snapshotSeq;
  const reply = await new Promise<SnapshotReply>((done) => {
    const timer = setTimeout(() => {
      snapshotReplies.delete(id);
      done({ ok: false, reason: `the page did not answer within ${SNAPSHOT_REPLY_MS / 1000} s` });
    }, SNAPSHOT_REPLY_MS);
    snapshotReplies.set(id, (r) => {
      clearTimeout(timer);
      done(r);
    });
    contents.send("viewer:snapshot", { id, image: [size.width, size.height] });
  });
  if (!reply.ok) return refuse(reply.reason);
  writeFileSync(join(dir, SNAPSHOT_PNG), img.toPNG());
  writeFileSync(join(dir, SNAPSHOT_VIEW), JSON.stringify(reply.view, null, 2));
  writeManifestLast(dir, JSON.stringify(stamped(reply.manifest), null, 2));
}

ipcMain.handle("viewer:snapshotReply", (_e, id: unknown, reply: unknown) => {
  const done = typeof id === "number" ? snapshotReplies.get(id) : undefined;
  if (!done) throw new Error(`viewer:snapshotReply: no snapshot ${shown(id)} is waiting`);
  snapshotReplies.delete(id as number);
  done(reply as SnapshotReply);
});

let appReadyAt = 0;
ipcMain.handle("app:config", () => ({
  mode,
  file: mode === "view" ? (session.set?.model ?? null) : file ? resolve(file) : null,
  t0,
  appReadyMs: appReadyAt - t0,
  configMs: Date.now() - t0,
  pick: arg("pick") ?? null,
  // capture: the results sibling of the captured model, when one exists (the
  // V4b stills come from the set the file belongs to, as a double-click opens it);
  // measure (V4c1): the same sibling, for the contour scrub row
  results: (mode === "capture" || mode === "measure") && file ? pairSet(resolve(file), existsSync).results : null,
}));

/** The kind of `path` as the pairing names it, or null for a file the app does not pair. */
function kindOf(path: string): Kind | null {
  try {
    return classify(path).kind;
  } catch {
    return null;
  }
}

/** The open set after the renderer opened `path` as a model: the set of a `<stem>.h5`, else none. */
function followModel(path: string): void {
  if (kindOf(path) === "model") session.rendererOpened(path);
  else session.close();
}

/**
 * VU12 (#2753), owner ruling "option A": the page is no longer trusted to name
 * a path to open. These are the paths a user gesture or main itself produced:
 * a file the user dropped, as Electron names it (preload.ts `gestured`, over
 * `viewer:dropped`), and the paths main sent the page, which are the window's
 * set (`windowSet`: argv / `--file`, `onOpen`, the render job) and `config().file`.
 * VU13 (#2760): only the latest drop counts, and it is not spent by an open (a
 * model the page reads on its own re-reads it on a rewrite); a drop the user
 * has since replaced is no gesture.
 */
let dropped: string | null = null;
ipcMain.on("viewer:dropped", (e, path: unknown) => {
  if (typeof path === "string" && path !== "") dropped = path;
  e.returnValue = null;
});

/** Why the page may not ask to open `path`, or null when a gesture or main produced it. */
function ungestured(door: string, path: string): string | null {
  if (dropped !== null && samePath(dropped, path)) return null;
  const set = windowSet();
  if (!("reason" in set) && [set.model, set.geometry, set.results].some((p) => samePath(p, path))) return null;
  if (mode !== "view" && file && samePath(resolve(file), path)) return null;
  return `${door} opens only a file the user dropped or a path main sent the page; refused ${path}`;
}

ipcMain.handle("model:open", async (_e, path: unknown) => {
  if (typeof path !== "string") return { ok: false, error: `model:open needs a path; got ${shown(path)}` };
  // A free string is refused before any read, so the set stays as it is (VU12).
  const refused = ungestured("model:open", path);
  if (refused !== null) return { ok: false, error: refused };
  let model;
  try {
    model = await openModel(path);
  } catch (err) {
    // The renderer takes the old model off screen (state/reduce.ts `fileFailed`),
    // so the old set ends either way (./session.ts). A `<stem>.h5` that failed
    // (a model still being written) becomes the open set, watched, so its
    // rewrite retries the read. Any other path (a `.user.h5`, a results file)
    // never becomes the set: the user doors would then read beside it (#2752 review F1).
    if (mode === "view" && typeof path === "string") followModel(path);
    return { ok: false, error: err instanceof Error ? err.message : String(err) };
  }
  // A model the renderer opened on its own (a drop on the P0 page): the open
  // set follows what is on screen (./session.ts), by the same rule.
  if (mode === "view" && typeof path === "string") followModel(path);
  // /provenance rides with the model; a refused or broken zone does not stop
  // the model from loading (the renderer shows why go-to-source is off).
  return {
    ok: true,
    model,
    provenance: await openProvenance(path),
    // V4b: the results pairing key, and the element tags of the /sequence element rows.
    lineage: await openLineage(path),
    stageElements: await openStageElements(path, model.sequence),
  };
});

/**
 * VU12 (#2753): the set whose geometry and results files the `results:*` and
 * `geometry:open` doors read, as main chose it and never as the request names
 * it. View mode: the set on screen (`currentSet`), which a double-click, a
 * second instance, a drop (`viewer:requestOpen`) or a model the renderer read
 * (`followModel`) opened. Render mode: the job's set, from the spec. Describe,
 * capture and measure: the set of `--file`, the one `render:job` and
 * `app:config` hand the page. No other mode's page reads these doors. The
 * reason when there is no set.
 */
function windowSet(): { model: string | null; geometry: string | null; results: string | null } | { reason: string } {
  if (mode === "view") return currentSet() ?? { reason: "no set is open" };
  if (mode === "render") return renderJob ?? { reason: "the render job was refused" };
  if ((mode === "describe" || mode === "capture" || mode === "measure") && file) {
    try {
      return pairSet(resolve(file), existsSync);
    } catch (err) {
      return { reason: `the set of ${file} cannot be paired (${err instanceof Error ? err.message : String(err)})` };
    }
  }
  return { reason: `--mode=${mode} opens no set` };
}

/**
 * The window's set's `slot` file when `asked` names it (`samePath`: case-folded
 * where the file system folds), else why not. Only the path this returns is read.
 */
function setFile(door: string, slot: "geometry" | "results", asked: string): { ok: true; path: string } | { ok: false; error: string } {
  const set = windowSet();
  if ("reason" in set) return { ok: false, error: `${door} reads only the open set's ${slot} file, and ${set.reason}; refused ${asked}` };
  const own = set[slot];
  if (own === null || !samePath(asked, own)) return { ok: false, error: `${door} reads only the open set's ${slot} file ${own ?? "(the set has none)"}; refused ${asked}` };
  return { ok: true, path: own };
}

// V4b (#1728): the results file, read lazily here; the renderer holds its
// metadata and asks for one step at a time (./results.ts). Each door reads
// only the window's set's results file (`setFile`, VU12); a step or a scan of
// a file the set has left is refused too, though the host still indexes it.
const resultsHost = new ResultsHost(h5wasm as unknown as ResultsH5, h5wasm.ready);
ipcMain.handle("results:open", (_e, path: unknown) => {
  if (typeof path !== "string") return { ok: false, error: `results:open needs a path; got ${shown(path)}` };
  const f = setFile("results:open", "results", path);
  if (!f.ok) return f;
  return resultsHost.open(f.path);
});
ipcMain.handle("results:readStep", (_e, req: unknown) => {
  const r = req as StepRequest | null;
  if (
    !r ||
    typeof r.path !== "string" ||
    typeof r.stage !== "string" ||
    typeof r.step !== "number" ||
    typeof r.seq !== "number" ||
    (r.element !== undefined && typeof r.element !== "string") ||
    (r.lineDiagram !== undefined && typeof r.lineDiagram !== "string") ||
    (r.tensor !== undefined && !(Array.isArray(r.tensor) && r.tensor.every((c) => typeof c === "string")))
  ) {
    return { ok: false, seq: -1, error: `results:readStep: malformed request ${shown(req)}` };
  }
  const f = setFile("results:readStep", "results", r.path);
  if (!f.ok) return { ok: false, seq: r.seq, error: f.error };
  return resultsHost.readStep({ ...r, path: f.path });
});
ipcMain.handle("results:close", () => resultsHost.close());
// V4c1 (#1756): the history-range scan of a contour's field; progress goes to
// the asking page. In capture mode it reads one step per chunk, 2 s apart,
// so a still can show the legend while the scan is still running (a hidden
// window paints slowly; renderer/app.ts checks the scan was still running).
ipcMain.handle("results:scan", (e, req: unknown) => {
  const r = req as ScanRequest | null;
  if (
    !r ||
    typeof r.path !== "string" ||
    typeof r.stage !== "string" ||
    !Array.isArray(r.components) ||
    typeof r.magnitude !== "boolean" ||
    typeof r.scanId !== "number" ||
    (r.source !== undefined && r.source !== "element" && r.source !== "line") ||
    // V4c2b: a plan only with an element scan, and only one that checks (state/extrapolation.ts `checkPlan`).
    (r.plan !== undefined && (r.source !== "element" || !checkPlan(r.plan)))
  ) {
    return { ok: false, scanId: -1, error: `results:scan: malformed request ${shown(req)}` };
  }
  const f = setFile("results:scan", "results", r.path);
  if (!f.ok) return { ok: false, scanId: r.scanId, error: f.error };
  const sender = e.sender;
  const progress = (p: { scanId: number; done: number; total: number }) => {
    if (!sender.isDestroyed()) sender.send("results:scanProgress", p);
  };
  return resultsHost.scan({ ...r, path: f.path }, progress, mode === "capture" ? { chunkSteps: 1, paceMs: 2000 } : {});
});
ipcMain.handle("results:scanCancel", (_e, scanId: unknown) => {
  if (typeof scanId === "number") resultsHost.cancelScan(scanId);
});

// ---------------------------------------------------------------------------
// VU10 (ADR 0132 D9): the variant doors. Each names a manifest row by index and
// label (`study.row`, which refuses a mismatch); none takes a path. A switch
// opens the row's set as the window set (D9.6). The compare door reads the
// row's set into the compare slot and `compareHost`, and never adopts it: no
// `session.open`, no `followModel`, no gesture record, so `windowSet` and
// every window door are as they were (INV-6).
// ---------------------------------------------------------------------------

/** The study as the page sees it: the row on screen is the window set's model. */
function studyView() {
  const set = windowSet();
  return study.view("reason" in set ? null : set.model);
}

function sendStudy(): void {
  if (win) win.webContents.send("viewer:variants", studyView());
}

function closeCompare(): void {
  compareHost.close();
  study.compare = null;
}

// D9.3: a rewritten manifest is read again; a compare row it no longer holds leaves the slot.
study.onChange = () => {
  if (study.compare === null) compareHost.close();
  sendStudy();
};

ipcMain.handle("variants:current", () => studyView());

ipcMain.handle("variants:switch", (_e, index: unknown, label: unknown) => {
  const r = study.row(index, label);
  if (!r.ok) return { ok: false, reason: `variants:switch: ${r.reason}` };
  if (mode !== "view") return { ok: false, reason: `variants:switch: --mode=${mode} has no window set to switch` };
  const s = rowSet(r.open);
  if (!s.ok) return { ok: false, reason: `variants:switch: ${s.reason}` };
  // The window set cannot also be the compare set.
  if (study.compare?.row.index === r.row.index) closeCompare();
  quietFails = [];
  try {
    const ok = session.open(r.open, true);
    sendStudy();
    return ok ? { ok: true } : { ok: false, reason: quietFails.join("; ") || `cannot open variant ${JSON.stringify(r.row.label)}` };
  } finally {
    quietFails = null;
  }
});

/** The model file's lineage hashes, or why it has none (the compare set's results must pair with its own model). */
async function modelHashes(path: string): Promise<{ femHash: string; modelHash: string } | string> {
  const a = await openLineage(path);
  if (!a.ok) return `its model ${path} cannot be read (${a.error})`;
  if (!a.lineage?.femHash || !a.lineage.modelHash) return `its model ${path} stamps no fem_hash and model_hash in /meta/lineage`;
  return { femHash: a.lineage.femHash, modelHash: a.lineage.modelHash };
}

ipcMain.handle("variants:compare", async (_e, index: unknown, label: unknown) => {
  const r = study.row(index, label);
  if (!r.ok) return { ok: false, reason: `variants:compare: ${r.reason}` };
  const row = r.row;
  const ws = windowSet();
  if (!("reason" in ws) && (samePath(ws.model, r.open) || study.rowOfModel(ws.model)?.index === row.index)) {
    return { ok: false, reason: `variants:compare: variant ${JSON.stringify(row.label)} is the window set; a compare set is another variant` };
  }
  // At most one compare set: a new one replaces it.
  closeCompare();
  const rs = rowSet(r.open);
  if (!rs.ok) return { ok: false, reason: `variants:compare: ${rs.reason}` };
  const set = rs.set;
  if (set.results === null) return { ok: false, reason: `variants:compare: variant ${JSON.stringify(row.label)} has no results file beside ${r.open}` };
  const own = await modelHashes(r.open);
  if (typeof own === "string") return { ok: false, reason: `variants:compare: variant ${JSON.stringify(row.label)}: ${own}` };
  const a = await compareHost.open(set.results);
  if (!a.ok) return { ok: false, reason: `variants:compare: ${a.error}` };
  const lin = a.meta.lineage;
  // The compare set's results must be of its own model, by the hashes the window's pairing uses.
  if (!lin || lin.femHash !== own.femHash || lin.modelHash !== own.modelHash) {
    compareHost.close();
    return { ok: false, reason: `variants:compare: ${set.results} does not pair with ${r.open} (fem_hash and model_hash differ, or are not stamped${a.meta.lineageReason ? `: ${a.meta.lineageReason}` : ""})` };
  }
  study.compare = { row, set };
  sendStudy();
  return {
    ok: true,
    compare: {
      index: row.index,
      label: row.label,
      results: set.results,
      femHash: lin.femHash,
      stages: compareStagesOf(a.meta),
    },
    nodeIds: a.nodeIds,
  };
});

// The compare set's step: the request names the stage, step and request number only; the file is the compare slot's.
ipcMain.handle("compare:readStep", (_e, req: unknown) => {
  const r = req as { stage?: unknown; step?: unknown; seq?: unknown } | null;
  const keys = r !== null && typeof r === "object" ? Object.keys(r).sort().join(",") : "";
  if (!r || keys !== "seq,stage,step" || typeof r.stage !== "string" || typeof r.step !== "number" || typeof r.seq !== "number") {
    return { ok: false, seq: -1, error: `compare:readStep takes { stage, step, seq } and never a path; got ${shown(req)}` };
  }
  const c = study.compare;
  if (c === null || c.set.results === null) return { ok: false, seq: r.seq, error: "compare:readStep: no compare set is open" };
  return compareHost.readStep({ path: c.set.results, stage: r.stage, step: r.step, seq: r.seq });
});

ipcMain.handle("compare:close", () => {
  closeCompare();
  sendStudy();
});

// A /geometry sibling (./zones.ts): only the window's set's (`setFile`, VU12).
ipcMain.handle("geometry:open", (_e, path: unknown) => {
  if (typeof path !== "string") return { ok: false, error: `geometry:open needs a path; got ${shown(path)}` };
  const f = setFile("geometry:open", "geometry", path);
  if (!f.ok) return f;
  return openGeometry(f.path);
});

/** A request as a log line, whatever the IPC carried (a BigInt, a cycle): never throws, unlike `JSON.stringify` (VU11, #2750). */
function shown(req: unknown): string {
  try {
    return inspect(req, { depth: 3, breakLength: Infinity, maxArrayLength: 10, maxStringLength: 300 });
  } catch {
    return Object.prototype.toString.call(req);
  }
}

/** The set on screen in view mode: the open set, with the user slot only the watcher knows (a user file that appears is a `changed`, not a reopen). */
function currentSet(): OpenSet | null {
  if (mode !== "view" || session.set === null) return null;
  return liveWatcher === null ? session.set : { ...session.set, user: liveWatcher.set.user };
}

/**
 * VU11 (#2750): the one `.user.h5` the user doors read, and the model and
 * results its staleness is judged against, all from the window's set and
 * never from the request: in view mode the set on screen (`currentSet`), in
 * render mode the job's set. The user file is the set's user slot, else
 * `<stem>.user.h5` where one would be, as the renderer names it
 * (state/reduce.ts `userPathOf`). The reason when there is none.
 */
function pairedUser(): { ok: true; user: string; model: string | null; results: string | null } | { ok: false; reason: string } {
  const set = mode === "view" ? currentSet() : mode === "render" && renderJob ? { model: renderJob.model, results: renderJob.results, user: null } : null;
  if (set === null) return { ok: false, reason: mode === "view" || mode === "render" ? "no set is open" : `--mode=${mode} reads no user file` };
  const from = set.model ?? set.results;
  let user = set.user;
  try {
    if (user === null && from !== null) user = candidates(from).user;
  } catch (err) {
    return { ok: false, reason: `the open set's ${from} names no user file (${err instanceof Error ? err.message : String(err)})` };
  }
  if (user === null) return { ok: false, reason: "the open set has neither a model nor results" };
  return { ok: true, user, model: set.model, results: set.results };
}

/**
 * The window's set's files when `asked` names them (`samePath`: case-folded
 * where the file system folds), else why not. Only what this returns is read.
 */
function confined(door: string, asked: { path: string; model?: string | null; results?: string | null }): ReturnType<typeof pairedUser> {
  const p = pairedUser();
  if (!p.ok) return { ok: false, reason: `${door} reads only the open set's user file, and ${p.reason}; refused ${asked.path}` };
  if (!samePath(asked.path, p.user)) return { ok: false, reason: `${door} reads only the open set's user file ${p.user}; refused ${asked.path}` };
  for (const k of ["model", "results"] as const) {
    const a = asked[k];
    if (a !== undefined && !samePath(a, p[k])) return { ok: false, reason: `${door} judges staleness only against the open set's ${k} ${p[k] ?? "(none)"}; refused ${a ?? "(none)"}` };
  }
  return p;
}

// ADR 0130 slice VU3a: the set's `<stem>.user.h5`, read only when the renderer
// asks, so the first frame reads none of it. The answer is the file's index
// (reader/userFile.ts `UserIndex`, never a `/results` DATA value) and A6's
// staleness against the set's model and results files (`OpenSet.model` and
// `OpenSet.results`; null when absent). The request names the three files,
// and each must be the window's set's (`pairedUser`); any other path is
// refused (VU11, #2750). A file that cannot be read answers `ok: false` with
// the reason, and raises nothing.
ipcMain.handle("user:read", async (_e, req: unknown) => {
  const r = req as { path?: unknown; model?: unknown; results?: unknown } | null;
  const pathOrNull = (x: unknown) => x === null || typeof x === "string";
  if (!r || typeof r.path !== "string" || !pathOrNull(r.model) || !pathOrNull(r.results)) {
    return { ok: false, error: `user:read needs { path, model, results }; got ${shown(req)}` };
  }
  const p = confined("user:read", { path: r.path, model: r.model as string | null, results: r.results as string | null });
  if (!p.ok) return { ok: false, error: p.reason };
  await h5wasm.ready;
  return readUserSet(h5wasm as unknown as UserH5, p.user, p.model, p.results, listDir);
});

// ADR 0130 slice VU7b-1 (#2740): one result of one `/spectra` case of the
// `.user.h5`, with its per-mode DATA and its reference state, on request
// (reader/userSpectra.ts `CaseRead`). The case is judged by the same read as
// `user:read`'s index, then its values are read; the renderer combines them
// with VU4's evaluator. The path must be the window's set's user file
// (`confined`, VU11). A refused or missing case, a path outside the set, and a
// file that cannot be read answer `ok: false` with the evaluator's reason;
// nothing raises.
ipcMain.handle("user:readCase", async (_e, req: unknown): Promise<CaseRead> => {
  const r = req as { path?: unknown; case?: unknown; target?: unknown; label?: unknown } | null;
  if (!r || typeof r.path !== "string" || typeof r.case !== "string" || typeof r.label !== "string" || typeof r.target !== "string" || !Object.hasOwn(SPECTRA_FAMILIES, r.target)) {
    return { ok: false, reason: "refused", detail: `user:readCase needs { path, case, target: ${Object.keys(SPECTRA_FAMILIES).join(" | ")}, label }; got ${shown(req)}` };
  }
  const p = confined("user:readCase", { path: r.path });
  if (!p.ok) return { ok: false, reason: "refused", detail: p.reason };
  await h5wasm.ready;
  const h5 = h5wasm as unknown as UserH5;
  try {
    return readCaseValues(h5, p.user, readUserFile(h5, p.user, null).spectra, r.case, r.target as SpectrumTarget, r.label);
  } catch (err) {
    return { ok: false, reason: "refused", detail: `${p.user}: ${err instanceof Error ? err.message : String(err)}` };
  }
});

for (const verb of ["subscribe", "unsubscribe"] as const) {
  ipcMain.on(`viewer:${verb}`, (_e, channel: unknown) => {
    if (channel !== "open" && channel !== "fileChanged") {
      fail(`viewer:${verb}: unknown channel ${shown(channel)}`);
      return;
    }
    if (verb === "subscribe") subscribed.add(channel);
    else subscribed.delete(channel);
  });
}

// The set already open, for a new onOpen listener (the preload replays it).
ipcMain.handle("viewer:currentSet", () => currentSet());

ipcMain.handle("viewer:requestOpen", (_e, path: unknown) => {
  if (typeof path !== "string") return { ok: false, reason: `requestOpen needs a path; got ${shown(path)}` };
  const refused = ungestured("requestOpen", path);
  if (refused !== null) return { ok: false, reason: refused };
  return openPath(path, true) ? { ok: true } : { ok: false, reason: `cannot open ${path}` };
});

// ADR 0128 D5: the click on an offer opens its set. Nothing an agent named raises a modal: a file that has gone refuses the offer in the list.
// VU12 (#2753): the path and every one of `needs` (at least one) must be offerable (`offerRows`); any other is refused and the set stays.
// VU13 (#2760): all of them one listed offer's files, or one set a listed offer left, so the set this open leaves goes with its offer.
ipcMain.handle("viewer:offerOpen", (_e, path: unknown, needs: unknown) => {
  if (typeof path !== "string" || !Array.isArray(needs) || needs.length === 0) return { ok: false, reason: `offerOpen needs a path and the files it needs; got ${shown(path)}, ${shown(needs)}` };
  const asked: unknown[] = [path, ...needs];
  const holds = (files: readonly string[]) => asked.every((f) => typeof f === "string" && files.some((o) => samePath(o, f)));
  const offer = offerRows.find((r) => holds(r.files));
  if (!offer && !offerRows.some((r) => r.left.some(holds))) {
    const known = (f: string) => offerRows.some((r) => [r.files, ...r.left].some((s) => s.some((o) => samePath(o, f))));
    const stray = asked.find((f) => typeof f !== "string" || !known(f));
    return { ok: false, reason: `offerOpen opens only the files of a set main offered or the set an offer left; refused ${shown(stray ?? asked)}` };
  }
  for (const f of needs as string[]) {
    if (!existsSync(f)) return { ok: false, reason: `no such file: ${f}` };
    if (!statSync(f).isFile()) return { ok: false, reason: `${f} is not a file` };
  }
  // The set on screen joins the offer's row: its Back (or a refused offer's put-back) reopens it through this door. A Back itself leaves nothing to come back to.
  const left = currentSet();
  const leftFiles = left === null ? [] : [left.model, left.results].filter((f): f is string => f !== null);
  const sameSet = (s: string[]) => s.length === leftFiles.length && s.every((f, i) => samePath(f, leftFiles[i]!));
  if (offer && leftFiles.length > 0 && !offer.left.some(sameSet)) offer.left.push(leftFiles);
  quietFails = [];
  try {
    return openPath(path, true) ? { ok: true } : { ok: false, reason: quietFails.join("; ") || `cannot open ${path}` };
  } catch (err) {
    return { ok: false, reason: err instanceof Error ? err.message : String(err) };
  } finally {
    quietFails = null;
  }
});

// VU13 (#2760): the editor opens only a source file the window's set's model names in its /provenance (./source.ts).
ipcMain.handle("source:goto", async (_e, path: unknown, line: unknown) => goToSource(path, line, await setSources()));

/** The source files the window's set's model names in its /provenance, read now (never the page's copy), or why there are none. */
async function setSources(): Promise<SourceScope> {
  const set = windowSet();
  if ("reason" in set) return { reason: set.reason };
  if (set.model === null) return { reason: "the open set has no model" };
  const p = await openProvenance(set.model);
  if (!p.ok) return { reason: `the /provenance of ${set.model} cannot be read (${p.error})` };
  if (p.zone === null) return { reason: `${set.model} has no /provenance zone` };
  return { files: provenanceSources(p.zone) };
}

ipcMain.handle("measure:metrics", async () => {
  const metrics = app.getAppMetrics();
  const kb = (type: string) =>
    metrics.filter((m) => m.type === type).reduce((s, m) => s + (m.memory?.workingSetSize ?? 0), 0);
  const gpu = (await app.getGPUInfo("basic")) as { gpuDevice?: { vendorId: number; deviceId: number; active?: boolean }[] };
  return {
    mainMB: kb("Browser") / 1024,
    rendererMB: kb("Tab") / 1024,
    gpuProcessMB: kb("GPU") / 1024,
    gpuDevices: gpu.gpuDevice ?? [],
  };
});

ipcMain.handle("measure:done", (_e, result: unknown) => {
  writeFileSync(out!, JSON.stringify(result, null, 2));
  // V4c1: the contour scrub (a model with a results sibling) prints its own ledger row.
  const contour = (result as { contour?: ContourMeasure | null } | null)?.contour;
  if (contour) printContourRow(contour);
  // V4d: the same scrub with the deformed shape on.
  const deform = (result as { deform?: ContourMeasure | null } | null)?.deform;
  if (deform) printContourRow(deform);
  // V4c2a: the scrub with an element contour on, then deformed too.
  // V4c2b: and once per node-located draw mode.
  for (const key of ["elements", "elementsDeform", "elementsAveraged", "elementsNode", "elementsDiscrete"] as const) {
    const row = (result as Partial<Record<typeof key, ContourMeasure | null>> | null)?.[key];
    if (row) printContourRow(row);
  }
  // ADR 0122 slice 4 (#2129): a `<stem>.ladruno` beside the model gets its read row.
  if (mode === "measure" && file) {
    try {
      measureLadrunoRead(resolve(file));
    } catch (err) {
      // headlessEnd, not exitCode + app.quit(): app.quit() exits 0 whatever process.exitCode says.
      headlessEnd(`the .ladruno read row: ${errText(err)}`);
      return;
    }
  }
  app.quit();
});

// ---------------------------------------------------------------------------
// The .ladruno read row (ADR 0122 slice 4, #2129). With a `<stem>.ladruno`
// beside the measured model, main times the .ladruno's index and one nodal
// step, and the same for `<stem>.results.h5` of the same run when one exists,
// then prints one MEASUREMENTS row per file. It runs after the orbit, so it
// shares no clock with the first frame; the renderer's handle on the file is
// closed first, so the index is a fresh open. The `layout` column names the
// displacement dataset's chunk shape and filters: the fork writes chunked,
// deflated datasets and NativeWriter contiguous ones, and that is what the
// two read columns compare.
// ---------------------------------------------------------------------------

/** The translational displacement components, under the .ladruno and the native reader's names. */
const DISPLACEMENT = /^(DISPLACEMENT\/U[xyz]|displacement_[xyz])$/;
const WARM_REPEATS = 21;

interface ReadRow {
  file: string;
  sizeBytes: number;
  stage: string;
  steps: number;
  nodes: number;
  components: string[];
  layout: string;
  indexMs: number;
  stepColdMs: number;
  stepWarmMs: number;
}

function measureLadrunoRead(model: string): void {
  const results = pairSet(model, existsSync).results;
  if (results === null || !/\.ladruno$/i.test(results)) return;
  resultsHost.close();
  const rows = [timeRead(results, openLadruno)];
  const native = results.replace(/\.ladruno$/i, ".results.h5");
  if (existsSync(native)) rows.push(timeRead(native, openResults));
  printReadRows(rows);
}

/** The index (open to a full index) and the middle step of the displacement components, cold then warm. */
function timeRead(path: string, open: (h5: ResultsH5, path: string) => ResultsFile | LadrunoFile): ReadRow {
  const sizeBytes = statSync(path).size;
  const [rf, indexMs] = timed(() => open(h5wasm as unknown as ResultsH5, path));
  let row: Omit<ReadRow, "layout">;
  let dataset: string;
  try {
    const s = rf.index.stages.find((st) => st.kind !== "mode" && st.nodes?.components.some((c) => DISPLACEMENT.test(c)));
    if (!s?.nodes) {
      const seen = rf.index.stages.map((st) => `${st.id}: ${st.nodes ? st.nodes.components.join(", ") : "no nodal results"}`);
      throw new Error(`${path}: the read row needs a stage with nodal displacements; the file has ${seen.join("; ") || "no stages"}`);
    }
    const nodes = s.nodes;
    const comps = nodes.components.filter((c) => DISPLACEMENT.test(c));
    const mid = Math.floor(s.nSteps / 2);
    const [, stepColdMs] = timed(() => rf.readNodalStep(s.id, comps, mid));
    const warm = Array.from({ length: WARM_REPEATS }, () => timed(() => rf.readNodalStep(s.id, comps, mid))[1]);
    // The .ladruno keeps one (T, N, C) DATA per response; the native file one (T, N) dataset per component.
    dataset = rf instanceof LadrunoFile ? `${nodes.path}/DISPLACEMENT/DATA` : `${nodes.path}/${comps[0]!}`;
    row = { file: path, sizeBytes, stage: s.name, steps: s.nSteps, nodes: nodes.ids.length, components: comps, indexMs, stepColdMs, stepWarmMs: median(warm) };
  } finally {
    rf.close();
  }
  return { ...row, layout: layoutOf(path, dataset) };
}

/** `chunks [a, b, c], deflate` or `contiguous`, of one dataset (read after the timed handle is closed). */
function layoutOf(path: string, dataset: string): string {
  const f = openH5(h5wasm as unknown as H5Openable<h5wasm.File>, path);
  try {
    if (Number(f.file_id) < 0) throw new Error(`HDF5 could not open ${path}`);
    const ds = f.get(dataset);
    if (!(ds instanceof h5wasm.Dataset)) throw new Error(`${path}: ${dataset} is not a dataset`);
    const chunks = ds.metadata.chunks;
    const filters = ds.filters.map((x) => x.name || `filter ${x.id}`);
    return [chunks ? `chunks [${chunks.join(", ")}]` : "contiguous", ...filters].join(", ");
  } finally {
    f.close();
  }
}

function printReadRows(rows: ReadRow[]): void {
  process.stdout.write(
    "| file | size (MB) | stage | steps | nodes | components | layout | index (ms) | one step cold / warm (ms) |\n" +
      "|---|---|---|---|---|---|---|---|---|\n" +
      rows
        .map(
          (r) =>
            `| ${r.file.split(/[\\/]/).pop()} | ${(r.sizeBytes / 1048576).toFixed(2)} | ${r.stage} | ${r.steps} | ${r.nodes} | ${r.components.join(", ")} | ${r.layout} | ` +
            `${r.indexMs.toFixed(1)} | ${r.stepColdMs.toFixed(2)} / ${r.stepWarmMs.toFixed(2)} |\n`,
        )
        .join(""),
  );
}

/**
 * Write one still: `--out` itself, or `<stem>.<suffix>.png` beside it. A
 * hidden window's `capturePage` copies the last frame it showed, which can be
 * from before the page's latest changes (#2716, the race `render:capture`
 * settles for #2696). So the page is captured until two consecutive captures
 * are pixel-identical, at most `--capture-tries` times; a page that never
 * settles fails the run (exit 1, the reason naming the still in
 * `<out>.error.txt`), and no PNG is written for it. As for `render:capture`,
 * no still is written once the run is ending (`<out>.error.txt` already says
 * why it failed), and a write that throws part way leaves no partial PNG
 * (#2736).
 */
ipcMain.handle("capture:still", async (e, suffix: string) => {
  const win = BrowserWindow.fromWebContents(e.sender);
  if (!win) throw new Error("capture: no window");
  const safeSuffix = suffix ? suffix.replace(/[\\/]/g, "_") : "";
  const path = safeSuffix ? out!.replace(/\.png$/i, "") + `.${safeSuffix}.png` : out!;
  const img = await settledCapture(win);
  if (!img) {
    headlessEnd(`capture:still ${path}: the page did not settle: no two consecutive of ${CAPTURE_TRIES} captures were pixel-identical (--capture-tries)`);
    return null;
  }
  const size = img.getSize();
  if (img.isEmpty() || size.width === 0) {
    // A capture with no still is a failed run: exit 1 with the reason in <out>.error.txt, never exit 0 (#2708 review).
    headlessEnd(`capturePage returned an empty image for ${path}`);
    return null;
  }
  refuseIfEnding(path);
  writeStill(path, img.toPNG());
  lastCaptured = path;
  return path;
});

ipcMain.handle("capture:done", () => app.quit());

/** Capture only (V4b): resize the page to `width` x `height` CSS pixels, for a still at another window width. */
ipcMain.handle("capture:resize", (e, width: unknown, height: unknown) => {
  if (mode !== "capture") throw new Error("capture:resize: only in --mode=capture");
  if (typeof width !== "number" || typeof height !== "number") throw new Error("capture:resize needs a width and a height");
  BrowserWindow.fromWebContents(e.sender)?.setContentSize(Math.round(width), Math.round(height));
});

// ---------------------------------------------------------------------------
// ADR 0128 D4 agent modes: the renderer (renderer/agentRender.ts) asks for its
// job, hands over each still, and reports the run; main writes every file,
// only into `--out` (INV-1), and only a PNG named by a shot of the spec.
// ---------------------------------------------------------------------------

/** The shot names a PNG may still be written under (each once). */
const unwritten = new Set(renderJob?.spec.shots.map((s) => s.name) ?? []);
/** The shots whose PNG is on disk whole: added only after `writeStill` returns (#2708 review). */
const written = new Set<string>();

function stillPath(name: unknown): string {
  if (mode !== "render" || !renderJob) throw new Error("render: only in --mode=render");
  if (typeof name !== "string" || !unwritten.has(name)) throw new Error(`render: ${shown(name)} is not a shot of the spec still to be written`);
  unwritten.delete(name);
  return join(resolve(out!), `${name}.png`);
}

/**
 * No still once the run is ending: `headlessEnd` has already written the
 * manifest that refuses the run, and a capture that resolves after it (its
 * `await capturePage()` was pending) would leave a PNG with no entry (INV-6,
 * #2708 review round 2). A capture's `headlessEnd` has written
 * `<out>.error.txt`, and a still written after it would stand beside the
 * failure (#2736).
 */
function refuseIfEnding(path: string): void {
  if (ending) throw new Error(`${mode}: the run is ending; ${path} is not written`);
}

/** A still written whole or not at all: a throw part way (a full disk) removes what it wrote, so no PNG stands beside the shot's refusal (#2681 review). */
function writeStill(path: string, png: Uint8Array): void {
  try {
    writeFileSync(path, png);
  } catch (err) {
    try {
      rmSync(path, { force: true });
    } catch (gone) {
      throw new Error(`${errText(err)}; and the partial ${path} could not be removed: ${errText(gone)}`);
    }
    throw err;
  }
}

/**
 * INV-6 when a render cannot reach `render:done`: the renderer held the
 * manifest entries, so the stills main has written have none and are removed,
 * and manifest.json refuses the run, naming them and why. A manifest already
 * written (the run had ended) is left as it is.
 */
function abandonRender(reason: string): void {
  if (!renderJob || !out) return;
  const dir = resolve(out);
  if (existsSync(join(dir, MANIFEST_FILE))) return;
  const drawn = renderJob.spec.shots.map((s) => s.name).filter((n) => written.has(n));
  // Every path handed out is removed, a shot whose capture was in flight too; only the whole ones are named.
  for (const s of renderJob.spec.shots) if (!unwritten.has(s.name)) rmSync(join(dir, `${s.name}.png`), { force: true });
  const shots = renderJob.spec.shots.length;
  const what = drawn.length
    ? `${drawn.length} of ${shots} stills had been written (${drawn.join(", ")}); they are removed, because their manifest entries were the renderer's and a PNG without its entry breaks INV-6`
    : `none of the ${shots} stills had been written`;
  writeManifestLast(dir, refusedManifest(resolve(arg("spec")!), "render", `the run did not finish: ${reason}; ${what}`, { version: app.getVersion(), version_reason: null }, new Date().toISOString()));
}

ipcMain.handle("render:job", () => {
  if (mode === "render" && renderJob) return { mode, ...renderJob };
  if (mode === "describe") {
    const set = pairSet(resolve(file!), existsSync);
    return { mode, model: set.model, geometry: set.geometry, results: set.results };
  }
  throw new Error(`render:job: not an agent mode (--mode=${mode})`);
});

/** A model-mode still the renderer composed (renderer/stillLayers.ts). */
ipcMain.handle("render:png", (_e, name: unknown, png: unknown) => {
  if (!(png instanceof Uint8Array) || png.length < 8) throw new Error("render:png needs the PNG bytes");
  const path = stillPath(name);
  refuseIfEnding(path);
  writeStill(path, png);
  written.add(name as string);
  return { path };
});

/** How many captures an app-mode still may take to settle (`--capture-tries`, 2 or more). */
const CAPTURE_TRIES = Number(arg("capture-tries") ?? 10);
/** The pause between two captures of one still: about a frame. */
const CAPTURE_GAP_MS = 16;

/**
 * The page of `w` as it settled: captured until two consecutive captures are
 * pixel-identical, at most `CAPTURE_TRIES` captures `CAPTURE_GAP_MS` apart;
 * null when no two consecutive ones matched. `render:capture` and
 * `capture:still` both take their stills from it.
 */
async function settledCapture(w: BrowserWindow): Promise<Electron.NativeImage | null> {
  let img = await w.webContents.capturePage();
  for (let n = 1; n < CAPTURE_TRIES; n++) {
    await new Promise((r) => setTimeout(r, CAPTURE_GAP_MS));
    const next = await w.webContents.capturePage();
    const same = next.toBitmap().equals(img.toBitmap());
    img = next;
    if (same) return img;
  }
  return null;
}

/**
 * An app-mode still: the page as captured, which must be exactly the spec's
 * size. A hidden window's `capturePage` copies the last frame the window
 * showed, which can be from before the page's latest changes (#2696: an app
 * still with no panels, or a phase readout a step old, reported `ok`). So the
 * page is captured until two consecutive captures are pixel-identical, at most
 * `--capture-tries` times; a page that never settles refuses the shot, and no
 * PNG is written. Two matching captures prove the frame stopped changing, not
 * that it is the latest frame.
 */
ipcMain.handle("render:capture", async (e, name: unknown) => {
  const w = BrowserWindow.fromWebContents(e.sender);
  if (!w || !renderJob) throw new Error("render:capture: no render window");
  const path = stillPath(name);
  const img = await settledCapture(w);
  if (!img) {
    throw new Error(`render:capture ${String(name)}: the page did not settle: no two consecutive of ${CAPTURE_TRIES} captures were pixel-identical (--capture-tries)`);
  }
  const size = img.getSize();
  const [width, height] = renderJob.spec.size;
  if (img.isEmpty() || size.width !== width || size.height !== height) {
    throw new Error(`render:capture ${String(name)}: the page captured as ${size.width} x ${size.height}, not the spec's ${width} x ${height}`);
  }
  refuseIfEnding(path);
  writeStill(path, img.toPNG());
  written.add(name as string);
  return { path, width: size.width, height: size.height };
});

ipcMain.handle("render:done", (_e, result: unknown) => {
  if (!agentMode) throw new Error("render:done: not an agent mode");
  const dir = resolve(out!);
  const r = result as { manifest?: { shots: { status: string }[]; timing: object }; describe?: unknown } | null;
  if (mode === "render") {
    if (!r?.manifest) throw new Error("render:done: no manifest");
    const path = join(dir, MANIFEST_FILE);
    writeFileSync(path, JSON.stringify(stamped(r.manifest), null, 2));
    const refused = r.manifest.shots.filter((s) => s.status !== "ok").length;
    process.stdout.write(`apeGmshViewer: render: ${r.manifest.shots.length - refused} stills, ${refused} refused; ${path}\n`);
    // D2: a refused shot makes the run's exit code non-zero; the other shots ran. (`app.quit()` exits 0 whatever process.exitCode says.)
    agentExit(refused ? 1 : 0);
    return;
  }
  if (!r?.describe) throw new Error("render:done: no description");
  writeFileSync(join(dir, "describe.json"), JSON.stringify(r, null, 2));
  process.stdout.write(`apeGmshViewer: describe: ${join(dir, "describe.json")}\n`);
  agentExit(0);
});

/** End an agent mode with its exit code, the results file closed first. */
function agentExit(code: number): void {
  ending = true;
  resultsHost.close();
  app.exit(code);
}

/** The renderer's V4c1 contour scrub measurement (renderer/app.ts `measureContour`). */
interface ContourMeasure {
  file: string;
  field: string;
  steps: number;
  scanMs: number;
  scrubStepsPerSec: number;
  stepMs: { median: number; max: number };
  peakMB: { main: number; renderer: number };
  cpu: { before: number; during: number };
}

function printContourRow(c: ContourMeasure): void {
  const name = c.file.split(/[\\/]/).pop();
  const peak = `${(c.peakMB.main + c.peakMB.renderer).toFixed(0)} (${c.peakMB.main.toFixed(0)} + ${c.peakMB.renderer.toFixed(0)})`;
  process.stdout.write(
    "| file | field | steps | CPU before / during | history scan (ms) | scrub (steps/s) | step median / max (ms) | peak MB main+renderer |\n" +
      "|---|---|---|---|---|---|---|---|\n" +
      `| ${name} | ${c.field} | ${c.steps} | ${c.cpu.before.toFixed(0)} / ${c.cpu.during.toFixed(0)} % | ${c.scanMs.toFixed(0)} | ` +
      `${c.scrubStepsPerSec.toFixed(1)} | ${c.stepMs.median.toFixed(2)} / ${c.stepMs.max.toFixed(1)} | ${peak} |\n`,
  );
}

/** Summed os.cpus() ticks: idle and total, for the CPU load of the machine. */
function cpuTicks(): { idle: number; total: number } {
  let idle = 0;
  let total = 0;
  for (const c of cpus()) {
    idle += c.times.idle;
    total += c.times.user + c.times.nice + c.times.sys + c.times.idle + c.times.irq;
  }
  return { idle, total };
}
let cpuMark = cpuTicks();
/** V4c1 measure: the machine's CPU load (%) since the last call (a reviewer re-runs under other loads). */
ipcMain.handle("measure:cpu", () => {
  const now = cpuTicks();
  const d = now.total - cpuMark.total;
  const busy = d > 0 ? (100 * (d - (now.idle - cpuMark.idle))) / d : NaN;
  cpuMark = now;
  return busy;
});
/** V4c1 measure: the main and renderer working sets now (MB). */
ipcMain.handle("measure:memory", () => {
  const m = app.getAppMetrics();
  const mb = (type: string) => m.filter((x) => x.type === type).reduce((s, x) => s + (x.memory?.workingSetSize ?? 0), 0) / 1024;
  return { main: mb("Browser"), renderer: mb("Tab") };
});

ipcMain.handle("app:fail", (_e, message: unknown) => {
  // A mode that fails must say so in its exit code (ADR 0128 D2: never a silent 0). `app.quit()` exits 0
  // whatever process.exitCode says, which let a capture that threw at 38 of 59 stills exit 0 (#2665).
  if (headless) {
    headlessEnd(`the page failed: ${String(message)}`);
    return;
  }
  process.stderr.write(`apeGmshViewer: ${String(message)}\n`);
});

app.whenReady().then(() => {
  appReadyAt = Date.now();
  if (mode === "measure-results") {
    // Not process.exitCode with app.quit(), which exits 0.
    measureResultsMode(file!, out!).catch((err) => headlessEnd(errText(err)));
    return;
  }
  if (badArgs !== null || reachMode || outBusy || (mode === "render" && !renderJob)) return;
  if (agentMode) mkdirSync(resolve(out!), { recursive: true });
  const w = new BrowserWindow({
    // ADR 0128 D9: a render window's page is the spec's size (at device scale 1, above).
    width: renderJob ? renderJob.spec.size[0] : 1600,
    // A tall hidden window lets one still show the whole definition chain.
    height: renderJob ? renderJob.spec.size[1] : mode === "capture" ? 1900 : 1000,
    useContentSize: agentMode,
    show: mode !== "capture" && !agentMode,
    backgroundColor: "#16181d",
    title: "apeGmshViewer",
    webPreferences: {
      preload: join(here, "preload.cjs"),
      contextIsolation: true,
      sandbox: true,
      nodeIntegration: false,
      backgroundThrottling: false,
    },
  });
  win = w;
  w.setMenuBarVisibility(false);
  // Hiding the menu bar grows the page: a render page is put back at the spec's size.
  if (renderJob) w.setContentSize(renderJob.spec.size[0], renderJob.spec.size[1]);
  // A page load drops the old page's subscriptions and the offers it listed.
  w.webContents.on("did-start-loading", pageLoading);
  // VU15 review: `did-start-loading` fires only when the page goes from idle to loading, so a reload while the page
  // is still loading (`deliver` reloads a page that has not subscribed yet) starts no such event. A new document
  // in the main frame starts a navigation every time, so it is a page load too; an in-page or subframe one is not.
  w.webContents.on("did-start-navigation", (d) => {
    if (d.isMainFrame && !d.isSameDocument) pageLoading();
  });
  // VU15: the load that commits is the one whose page may listen for offers.
  w.webContents.on("did-navigate", () => pageCommitted(w.webContents));
  w.on("closed", () => {
    win = null;
  });
  void w.loadFile(join(here, "index.html"));
});

app.on("window-all-closed", () => {
  session.close();
  resultsHost.close();
  app.quit();
});

// ---------------------------------------------------------------------------
// V4a measurement mode (#1707): the results.h5 read path, timed end to end.
// Main reads with reader/results.ts; the renderer gets each step over this
// app's IPC (ipcRenderer.invoke, preload `resultsStep`) and applies it to a
// colour buffer. The renderer page is a blank hidden page: the app's own page
// would start its view/measure flow on the file. Its apply is the WebGL work
// three.js r186 does for a `needsUpdate` colour attribute (one bufferSubData
// of the whole Float32Array, then a draw), written as raw WebGL2 calls
// because three.js is bundled inside the app page and cannot be reached here.
// Nothing here runs outside --mode=measure-results.
// ---------------------------------------------------------------------------

const median = (xs: number[]) => {
  const s = [...xs].sort((a, b) => a - b);
  return s.length ? s[Math.floor((s.length - 1) / 2)]! : NaN;
};
/** Scrub columns are means over the steps: the renderer's clock is coarsened to 0.1 ms. */
const mean = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : NaN);
const timed = <T>(f: () => T): [T, number] => {
  const t = performance.now();
  const v = f();
  return [v, performance.now() - t];
};

/** Peak working sets of the main and renderer processes, sampled every 20 ms. */
function memorySampler() {
  const peak = { mainMB: 0, rendererMB: 0, sumMB: 0 };
  const sample = () => {
    const m = app.getAppMetrics();
    const mb = (type: string) => m.filter((x) => x.type === type).reduce((s, x) => s + (x.memory?.workingSetSize ?? 0), 0) / 1024;
    const a = mb("Browser");
    const b = mb("Tab");
    peak.mainMB = Math.max(peak.mainMB, a);
    peak.rendererMB = Math.max(peak.rendererMB, b);
    peak.sumMB = Math.max(peak.sumMB, a + b);
  };
  const id = setInterval(sample, 20);
  return { peak, sample, stop: () => clearInterval(id) };
}

async function measureMpco(path: string, dataset: string) {
  await h5wasm.ready;
  const size = statSync(path).size;
  let f: h5wasm.File | null = null;
  const t = performance.now();
  try {
    f = openH5(h5wasm as unknown as H5Openable<h5wasm.File>, path);
    // h5wasm returns a negative id instead of throwing when HDF5 refuses the file.
    if (Number(f.file_id) < 0) throw new Error("HDF5 could not open the file (truncated, or not HDF5)");
  } catch (err) {
    return { kind: "mpco", file: path, sizeBytes: size, openMs: performance.now() - t, openError: err instanceof Error ? err.message : String(err) };
  }
  const openMs = performance.now() - t;
  try {
    const ds = f.get(dataset);
    if (!(ds instanceof h5wasm.Dataset)) throw new Error(`${dataset} is not a dataset in ${path}`);
    const [v, readMs] = timed(() => ds.value as ArrayBufferView);
    return { kind: "mpco", file: path, sizeBytes: size, openMs, dataset, shape: ds.shape, readMs, bytes: v.byteLength };
  } finally {
    f.close();
  }
}

async function measureResultsMode(path: string, outPath: string): Promise<void> {
  const mem = memorySampler();
  mem.sample();
  if (/\.mpco$/i.test(path)) {
    const dataset = arg("dataset");
    if (!dataset) throw new Error("an .mpco file needs --dataset=<path>");
    const r = await measureMpco(path, dataset);
    mem.sample();
    mem.stop();
    const result = { ...r, peakMemory: mem.peak };
    writeFileSync(outPath, JSON.stringify(result, null, 2));
    process.stdout.write(`${JSON.stringify(result)}\n`);
    app.quit();
    return;
  }
  await h5wasm.ready;
  const sizeBytes = statSync(path).size;
  const [rf, openMs] = timed(() => openResults(h5wasm as unknown as ResultsH5, path));
  try {
    const result = await measureOpen(rf, path, sizeBytes, openMs, mem);
    writeFileSync(outPath, JSON.stringify(result, null, 2));
    printResultsRow(result);
  } finally {
    rf.close();
    mem.stop();
    app.quit();
  }
}

async function measureOpen(rf: ResultsFile, path: string, sizeBytes: number, openMs: number, mem: ReturnType<typeof memorySampler>) {
  const s = rf.index.stages[0];
  if (!s?.nodes || s.gauss.length === 0) throw new Error(`${path}: the measurement needs a stage with nodal results and a gauss group`);
  const nodes = s.nodes;
  const vec = ["displacement_x", "displacement_y", "displacement_z"].filter((c) => nodes.components.includes(c));
  const g = s.gauss[0]!;
  const comp = g.components[0]!;
  const T = s.nSteps;
  const mid = Math.floor(T / 2);
  const rep = 21;

  // Cold: the first read of this step in this process. Warm: the same read again (median of 21).
  const [, nodalColdMs] = timed(() => rf.readNodalStep(s.id, vec, mid));
  const nodalWarm = Array.from({ length: rep }, () => timed(() => rf.readNodalStep(s.id, vec, mid))[1]);
  const [, elemColdMs] = timed(() => rf.readGaussStep(s.id, g.id, comp, mid));
  const elemWarm = Array.from({ length: rep }, () => timed(() => rf.readGaussStep(s.id, g.id, comp, mid))[1]);
  const node = nodes.ids[Math.floor(nodes.ids.length / 2)]!;
  const [hist, historyColdMs] = timed(() => rf.readNodeHistory(s.id, vec[0]!, node));
  const historyWarm = Array.from({ length: rep }, () => timed(() => rf.readNodeHistory(s.id, vec[0]!, node))[1]);
  mem.sample();

  const nodalEager = nodes.components.length * T * nodes.ids.length * 8;
  const gaussEager = s.gauss.reduce((a, x) => a + x.components.length * T * x.nElements * x.nGauss * 8, 0);

  // Main → renderer → apply, 100 consecutive steps from step 0 (or all, if fewer).
  const SCRUB = Math.min(100, T);
  ipcMain.handle("results:step", (_e, req: unknown) => {
    const r = req as { kind: "nodal" | "gauss"; step: number };
    const [values, readMs] =
      r.kind === "gauss" ? timed(() => [rf.readGaussStep(s.id, g.id, comp, r.step)]) : timed(() => rf.readNodalStep(s.id, vec, r.step));
    return { values, readMs };
  });
  const w = new BrowserWindow({
    width: 1600,
    height: 1000,
    show: false,
    webPreferences: { preload: join(here, "preload.cjs"), contextIsolation: true, sandbox: true, nodeIntegration: false, backgroundThrottling: false },
  });
  await w.loadURL("data:text/html,<!doctype html><title>measure-results</title><body></body>");
  mem.sample();
  const rr = (await w.webContents.executeJavaScript(`(${rendererHarness.toString()})(${nodes.ids.length}, ${SCRUB}, ${mid})`)) as Awaited<
    ReturnType<typeof rendererHarness>
  >;
  mem.sample();
  w.destroy();
  ipcMain.removeHandler("results:step");

  const steps = rr.steps;
  return {
    kind: "results",
    file: path,
    sizeBytes,
    nodes: nodes.ids.length,
    elements: s.gauss.reduce((a, x) => a + x.nElements, 0),
    steps: T,
    fields: { nodal: nodes.components, gauss: s.gauss.map((x) => ({ group: x.id, nGauss: x.nGauss, components: x.components })) },
    sourceType: rf.index.sourceType,
    openMs,
    nodalStep: { components: vec, coldMs: nodalColdMs, warmMs: median(nodalWarm) },
    elementStep: { component: `${g.id}/${comp}`, values: g.nElements * g.nGauss, coldMs: elemColdMs, warmMs: median(elemWarm) },
    nodeHistory: { node, values: hist.length, coldMs: historyColdMs, warmMs: median(historyWarm) },
    transfer: {
      nodalBytes: steps[0]?.bytes ?? 0,
      nodalRoundTripMs: mean(steps.map((x) => x.roundTripMs)),
      nodalReadInMainMs: mean(steps.map((x) => x.readMs)),
      nodalTransferMs: mean(steps.map((x) => x.roundTripMs - x.readMs)),
      gaussBytes: rr.gauss.bytes,
      gaussTransferMs: rr.gauss.transferMs,
    },
    apply: { colourMs: mean(steps.map((x) => x.colourMs)), uploadDrawMs: mean(steps.map((x) => x.uploadDrawMs)) },
    scrub: { steps: SCRUB, totalMs: rr.totalMs, stepsPerSecond: (1000 * SCRUB) / rr.totalMs },
    peakMemory: mem.peak,
    eagerBytes: { nodal: nodalEager, gauss: gaussEager, total: nodalEager + gaussEager },
    gpu: rr.gpu,
  };
}

/**
 * Runs in the renderer (serialised with toString, so it references nothing
 * outside itself). A points draw of `n` nodes; per step: invoke main, map the
 * displacement magnitude through jet into a Float32Array colour buffer,
 * bufferSubData it, draw, gl.finish().
 */
async function rendererHarness(n: number, scrub: number, gaussStep: number) {
  type Bridge = { resultsStep: (req: { kind: "nodal" | "gauss"; step: number }) => Promise<{ values: Float64Array[]; readMs: number }> };
  const viewer = (window as unknown as { viewer: Bridge }).viewer;
  const canvas = document.createElement("canvas");
  canvas.width = 1584;
  canvas.height = 961;
  document.body.appendChild(canvas);
  const gl = canvas.getContext("webgl2");
  if (!gl) throw new Error("no WebGL2 context");
  const sh = (type: number, src: string) => {
    const s = gl.createShader(type)!;
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s) ?? "shader");
    return s;
  };
  const prog = gl.createProgram()!;
  gl.attachShader(prog, sh(gl.VERTEX_SHADER, "#version 300 es\nin vec3 p; in vec3 c; out vec3 vc; void main(){ vc=c; gl_Position=vec4(p*1.8-0.9,1.0); gl_PointSize=2.0; }"));
  gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, "#version 300 es\nprecision mediump float; in vec3 vc; out vec4 o; void main(){ o=vec4(vc,1.0); }"));
  gl.linkProgram(prog);
  gl.useProgram(prog);
  const pos = new Float32Array(3 * n);
  for (let i = 0; i < pos.length; i++) pos[i] = ((i * 7919) % 1000) / 1000;
  const pb = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, pb);
  gl.bufferData(gl.ARRAY_BUFFER, pos, gl.STATIC_DRAW);
  const pl = gl.getAttribLocation(prog, "p");
  gl.enableVertexAttribArray(pl);
  gl.vertexAttribPointer(pl, 3, gl.FLOAT, false, 0, 0);
  const col = new Float32Array(3 * n);
  const cb = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, cb);
  gl.bufferData(gl.ARRAY_BUFFER, col, gl.DYNAMIC_DRAW);
  const cl = gl.getAttribLocation(prog, "c");
  gl.enableVertexAttribArray(cl);
  gl.vertexAttribPointer(cl, 3, gl.FLOAT, false, 0, 0);
  const mag = new Float32Array(n);
  const steps: { roundTripMs: number; readMs: number; bytes: number; colourMs: number; uploadDrawMs: number }[] = [];
  const t0 = performance.now();
  for (let k = 0; k < scrub; k++) {
    const a = performance.now();
    const r = await viewer.resultsStep({ kind: "nodal", step: k });
    const b = performance.now();
    let lo = Infinity;
    let hi = -Infinity;
    for (let i = 0; i < n; i++) {
      let s2 = 0;
      for (const v of r.values) s2 += v[i]! * v[i]!;
      const m = Math.sqrt(s2);
      mag[i] = m;
      if (m < lo) lo = m;
      if (m > hi) hi = m;
    }
    const span = hi > lo ? hi - lo : 1;
    for (let i = 0; i < n; i++) {
      const t = (mag[i]! - lo) / span;
      col[3 * i] = Math.min(1, Math.max(0, 1.5 - Math.abs(4 * t - 3)));
      col[3 * i + 1] = Math.min(1, Math.max(0, 1.5 - Math.abs(4 * t - 2)));
      col[3 * i + 2] = Math.min(1, Math.max(0, 1.5 - Math.abs(4 * t - 1)));
    }
    const c = performance.now();
    gl.bindBuffer(gl.ARRAY_BUFFER, cb);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, col);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.drawArrays(gl.POINTS, 0, n);
    gl.finish();
    const d = performance.now();
    steps.push({ roundTripMs: b - a, readMs: r.readMs, bytes: r.values.reduce((s, v) => s + v.byteLength, 0), colourMs: c - b, uploadDrawMs: d - c });
  }
  const totalMs = performance.now() - t0;
  const gt: number[] = [];
  let gaussBytes = 0;
  for (let i = 0; i < 5; i++) {
    const a = performance.now();
    const r = await viewer.resultsStep({ kind: "gauss", step: gaussStep });
    gt.push(performance.now() - a - r.readMs);
    gaussBytes = r.values[0]!.byteLength;
  }
  gt.sort((x, y) => x - y);
  const dbg = gl.getExtension("WEBGL_debug_renderer_info");
  return {
    steps,
    totalMs,
    gauss: { bytes: gaussBytes, transferMs: gt[2]! },
    gpu: dbg ? String(gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL)) : "unknown",
  };
}

function printResultsRow(r: Awaited<ReturnType<typeof measureOpen>>): void {
  const f0 = (x: number) => x.toFixed(0);
  const f1 = (x: number) => x.toFixed(1);
  const f2 = (x: number) => x.toFixed(2);
  const MB = (b: number) => (b / 1048576).toFixed(1);
  const fields = `${r.fields.nodal.length} nodal + ${r.fields.gauss.map((x) => `${x.components.length} x ${x.nGauss}-pt gauss`).join(", ")}`;
  process.stdout.write(
    "| date | commit | file | size (MB) | nodes | elements | steps | fields | open+index (ms) | nodal step cold / warm (ms) | element step cold / warm (ms) | node history (ms) | transfer main→renderer | apply colour + upload/draw (ms) | scrub (steps/s) | peak memory MB main+renderer | eager whole-history (MB) |\n" +
      "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n" +
      `| ${new Date().toISOString().slice(0, 10)} | ${arg("commit") ?? "?"} | ${r.file.split(/[\\/]/).pop()} | ${MB(r.sizeBytes)} | ${r.nodes} | ${r.elements} | ${r.steps} | ${fields} | ` +
      `${f1(r.openMs)} | ${f2(r.nodalStep.coldMs)} / ${f2(r.nodalStep.warmMs)} | ${f2(r.elementStep.coldMs)} / ${f2(r.elementStep.warmMs)} | ${f1(r.nodeHistory.coldMs)} | ` +
      `${(r.transfer.nodalBytes / 1024).toFixed(0)} KB in ${f2(r.transfer.nodalTransferMs)} ms | ${f2(r.apply.colourMs)} + ${f2(r.apply.uploadDrawMs)} | ${f1(r.scrub.stepsPerSecond)} | ` +
      `${f0(r.peakMemory.sumMB)} (${f0(r.peakMemory.mainMB)} + ${f0(r.peakMemory.rendererMB)}) | ${MB(r.eagerBytes.total)} |\n\n` +
      `detail: element step = ${r.elementStep.values} values; history = ${r.nodeHistory.values} values (warm ${f2(r.nodeHistory.warmMs)} ms); ` +
      `scrub round trip ${f2(r.transfer.nodalRoundTripMs)} ms (read in main ${f2(r.transfer.nodalReadInMainMs)} ms); ` +
      `gauss step transfer ${(r.transfer.gaussBytes / 1024).toFixed(0)} KB in ${f2(r.transfer.gaussTransferMs)} ms; ` +
      `eager nodal ${MB(r.eagerBytes.nodal)} MB + gauss ${MB(r.eagerBytes.gauss)} MB; GPU "${r.gpu}"\n`,
  );
}
