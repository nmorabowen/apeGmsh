// Electron main process. It reads files and writes one (ADR 0112 D4, as ADR
// 0132 D6 narrowed it): no network, no writes except the open set's
// `<stem>.workspace.h5` when the human saves in the window (./workspace.ts,
// HG9 on #2803) and the measurement / capture outputs a script asked for, and
// no child process except the user's editor for go-to-source (D3, ./source.ts). The one socket is a local pipe, private to one agent call:
// the running window's acknowledgement of `--present` or `--snapshot`.
//
// Modes (argv, set by scripts/launch.mjs):
//   --mode=view     open a window on --file, or on the first positional
//                   argument (a double-clicked file), or an empty drop target
//   --mode=measure  open, orbit, pick, write a JSON measurement to --out
//   --mode=capture  open hidden, select a beam, write a PNG still to --out
//                   (and, with a results sibling, one still per results stage)
//   --mode=measure-results  time the results.h5 read path (V4a, #1707;
//                   ./modes/measure.ts), write a JSON to --out and
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

import { app, BrowserWindow, dialog, ipcMain } from "electron";
import { existsSync, mkdirSync, readFileSync, realpathSync, statSync, unwatchFile, watchFile, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { inspect } from "node:util";
import * as h5wasm from "h5wasm/node";
import { openModel } from "../reader/node.ts";
import type { ResultsH5 } from "../reader/results.ts";
import { readUserFile, readUserResultColumn, readUserSet, type UserColumnRead, type UserH5 } from "../reader/userFile.ts";
import { readCaseValues, SPECTRA_FAMILIES, type CaseRead, type SpectrumTarget } from "../reader/userSpectra.ts";
import { candidates, classify, FOLD_CASE, fileFromArgv, listDir, pairSet, samePath, type Kind, type OpenSet } from "./pairing.ts";
import { compareStagesOf, ResultsHost, type ScanRequest, type StepRequest } from "./results.ts";
import { checkPlan } from "../state/extrapolation.ts";
import { OpenSession, Study } from "./session.ts";
import { confinedToStudy, isManifestPath, readVariants, resolveRow, type VariantsFs } from "../reader/variants.ts";
import { goToSource, provenanceSources, type SourceScope } from "./source.ts";
import { SetWatcher } from "./watch.ts";
import { openGeometry, openLineage, openProvenance, openStageElements } from "./zones.ts";
import { nodeDeps, registerWorkspace } from "./workspace.ts";
import type { WorkspaceH5 } from "../reader/workspace.ts";
import { agentMode, arg, argvSkip, badArgs, file, headless, mode, out, reachMode, t0 } from "./args.ts";
import { measureResultsMode, registerMeasure } from "./modes/measure.ts";
import { heardFromAgent, listedOffers, offersPageLoading, pageCommitted, reachWindow, registerReach, type Reach } from "./modes/reach.ts";
import { abandonRender, agentStartup, registerRender, whereStopped } from "./modes/render.ts";

const here = dirname(fileURLToPath(import.meta.url));
/** The preload script of every window, beside dist/main.mjs; a module under ./modes/ takes it from here (VH3, #2868). */
const preload = join(here, "preload.cjs");

// The command line is read in ./args.ts (VH2, #2866); a refused one exits here, before anything starts.
if (badArgs !== null) {
  process.stderr.write(`apeGmshViewer: ${badArgs}\n`);
  app.exit(2);
}

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

/** Set once the run is ending, so a late event (a renderer killed by the exit itself) changes nothing. */
let ending = false;
const errText = (err: unknown) => (err instanceof Error ? (err.stack ?? err.message) : String(err));
/** How long a page may leave main's question unanswered (`--hang-ms`). */
const HANG_MS = Number(arg("hang-ms") ?? 60000);

/** The run ends: the reason on stderr, where it stopped (./modes/render.ts `whereStopped`), a render's `--out` left true to INV-6, exit 1. */
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
  for (const p of [set.results, set.geometry, set.user, set.workspace]) {
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

// ADR 0128 D2: an agent mode's arguments, `--out` and render job are checked before any window (./modes/render.ts).
const startup = agentMode ? agentStartup(study) : { refused: false, job: null };
/** An agent mode refused at startup: no window opens (it has exited with code 2). */
const outBusy = startup.refused;
/** The render job: the validated spec and its set; null when the spec was refused (manifest.json names the key). */
const renderJob = startup.job;

// View mode is single-instance: a second launch hands its argv to the first
// (the "second-instance" handler below) and exits.
if (mode === "view" && !app.requestSingleInstanceLock()) app.exit(0);

// ADR 0128 D5, D6: `--present` and `--snapshot` reach the running window and exit (./modes/reach.ts).
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

// ADR 0128 D5, D6 in the running window: `viewer:offersReady`, `viewer:snapshotReply` and the offer queue (./modes/reach.ts).
registerReach({ window: () => win, shown });

/**
 * A page load: the old page's subscriptions go, and so does every offer row the
 * new page will not be sent (VU14, ./modes/reach.ts `offersPageLoading`). VU10:
 * the compare slot goes too, because the new page starts with no compare set,
 * and a slot it never opened is not one it may read (#2817 review 4).
 */
function pageLoading(): void {
  closeCompare();
  subscribed.clear();
  offersPageLoading();
}

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
function windowSet(): OpenSet | { reason: string } {
  if (mode === "view") return currentSet() ?? { reason: "no set is open" };
  // The job names no user or workspace file; the workspace doors take `<stem>.workspace.h5` beside its model (./workspace.ts `workspaceOf`).
  if (mode === "render") return renderJob ? { model: renderJob.model, geometry: renderJob.geometry, results: renderJob.results, user: null, workspace: null } : { reason: "the render job was refused" };
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

/** The set on screen in view mode: the open set, with the user and workspace slots only the watcher knows (a file that appears there is a `changed`, not a reopen). */
function currentSet(): OpenSet | null {
  if (mode !== "view" || session.set === null) return null;
  return liveWatcher === null ? session.set : { ...session.set, user: liveWatcher.set.user, workspace: liveWatcher.set.workspace };
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

// VU16 (#2800): one component of a state-free `ON_NODES` user result of the
// `.user.h5`, as a contour column (reader/userFile.ts `UserColumnRead`). The
// path must be the window's set's user file (`confined`, VU11), and the entry
// is judged against the set's own model and results (A6) before its DATA is
// read: a stale entry answers `stale` with its reason. A refused, absent,
// per-state or non-nodal entry, a component it does not list, and a file that
// cannot be read answer `ok: false` with the reason; nothing raises.
ipcMain.handle("user:readResult", async (_e, req: unknown): Promise<UserColumnRead> => {
  const r = req as { path?: unknown; name?: unknown; component?: unknown } | null;
  if (!r || typeof r.path !== "string" || typeof r.name !== "string" || typeof r.component !== "string") {
    return { ok: false, reason: "refused", detail: `user:readResult needs { path, name, component }; got ${shown(req)}` };
  }
  const p = confined("user:readResult", { path: r.path });
  if (!p.ok) return { ok: false, reason: "refused", detail: p.reason };
  await h5wasm.ready;
  return readUserResultColumn(h5wasm as unknown as UserH5, p.user, p.model, p.results, listDir, r.name, r.component);
});

// VU9a (ADR 0132 D6, D7; HG9 on #2803): the window set's `<stem>.workspace.h5`,
// read and written only as main names it (`windowSet`): the page never names
// its path (option A, #2753). Only the human's window saves (ADR 0128 INV-1 as
// ADR 0132 narrowed it); an agent mode reads it, and its save door refuses.
registerWorkspace(ipcMain, {
  set: windowSet,
  cannotWrite: () => (mode === "view" ? null : `--mode=${mode} never writes the workspace: only the human's window saves (ADR 0128 INV-1, ADR 0132)`),
  deps: nodeDeps(h5wasm as unknown as WorkspaceH5, h5wasm.ready, `apeGmshViewer ${app.getVersion()}`),
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
// VU12 (#2753): the path and every one of `needs` (at least one) must be offerable (./modes/reach.ts `listedOffers`); any other is refused and the set stays.
// VU13 (#2760): all of them one listed offer's files, or one set a listed offer left, so the set this open leaves goes with its offer.
ipcMain.handle("viewer:offerOpen", (_e, path: unknown, needs: unknown) => {
  if (typeof path !== "string" || !Array.isArray(needs) || needs.length === 0) return { ok: false, reason: `offerOpen needs a path and the files it needs; got ${shown(path)}, ${shown(needs)}` };
  const asked: unknown[] = [path, ...needs];
  const holds = (files: readonly string[]) => asked.every((f) => typeof f === "string" && files.some((o) => samePath(o, f)));
  const offerRows = listedOffers();
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

// The `measure:*` handlers and the .ladruno read row (./modes/measure.ts, VH2 #2866).
registerMeasure({ headlessEnd, errText, resultsHost, preload });

// The `capture:*` and `render:*` handlers (./modes/render.ts, VH3 #2868).
registerRender({
  headlessEnd,
  errText,
  shown,
  resultsHost,
  isEnding: () => ending,
  markEnding: () => {
    ending = true;
  },
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
      preload,
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
