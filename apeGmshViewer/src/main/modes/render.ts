// The ADR 0128 D4 render path of the main process, and the V4b capture
// stills (VH3, #2868: moved out of ../main.ts unchanged). `agentStartup` runs
// once at main's top level, before any window; `registerRender` registers the
// `render:*` and `capture:*` handlers, each once. The modes and their
// arguments are described at the top of ../main.ts.
//
// What it needs of main.ts comes through `registerRender` alone (the study
// through `agentStartup`), so this module never imports main.ts. main.ts's
// `headlessEnd` asks it where a run stopped (`whereStopped`) and leaves a
// render's `--out` true to INV-6 (`abandonRender`).

import { app, BrowserWindow, ipcMain } from "electron";
import { existsSync, rmSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { outRefusal, prepareRender, PROFILE_DIR, type RenderJobFiles } from "../../spec/job.ts";
import { MANIFEST_FILE, refusedManifest } from "../../spec/manifest.ts";
import { agentMode, arg, file, mode, out } from "../args.ts";
import { pairSet, type OpenSet } from "../pairing.ts";
import type { ResultsHost } from "../results.ts";
import type { Study } from "../session.ts";
import { jobFs, stamped, writeManifestLast } from "./out.ts";

/** VU18: the set a render shot draws, as main hands it to the page: the spec's set, or a variant row's. */
export interface DrawnSet {
  model: string;
  geometry: string | null;
  results: string | null;
}

/** What the render path takes from main.ts, handed over once (`registerRender`). */
export interface RenderHost {
  /**
   * VU18: a variant row's set as main opens it (main.ts `rowSet`, D9.4: paired
   * from the row's resolved model, each sibling confined to the study), or why not.
   */
  rowSet: (open: string) => { ok: true; set: OpenSet } | { ok: false; reason: string };
  /** VU18: empty the compare slot (main.ts `closeCompare`): the set a shot draws is never also its compare set. */
  closeCompare: () => void;
  /** End a headless run: the reason on stderr, exit 1 (main.ts). */
  headlessEnd: (reason: string) => void;
  /** main.ts's text of a thrown value. */
  errText: (err: unknown) => string;
  /** main.ts's log line of whatever the IPC carried. */
  shown: (req: unknown) => string;
  /** The window set's results host, closed before an agent mode exits. */
  resultsHost: ResultsHost;
  /** Whether the run is ending (main.ts's `ending`). */
  isEnding: () => boolean;
  /** Mark the run as ending, so a late event changes nothing (main.ts's `ending`). */
  markEnding: () => void;
}

/** The render job, set by `agentStartup`: the validated spec and its set; null when no spec was accepted. */
let renderJob: RenderJobFiles | null = null;
/** The shot names a PNG may still be written under (each once). */
const unwritten = new Set<string>();
/** The shots whose PNG is on disk whole: added only after `writeStill` returns (#2708 review). */
const written = new Set<string>();
/** The last still a capture wrote, to name where a failed capture stopped. */
let lastCaptured: string | null = null;
/** The study `agentStartup` opened from the spec's `files.variants` (main.ts's one `Study`). */
let renderStudy: Study | null = null;
/**
 * VU18: the set the shot in progress draws, and the one main.ts's window doors
 * read in render mode (`drawnSet`): the job's set until a shot names a variant
 * row (`render:variant`).
 */
let drawn: DrawnSet | null = null;

/** VU18: the set the render window's doors read now (main.ts `windowSet`, `pairedUser`); null when no render job was accepted. */
export function drawnSet(): DrawnSet | null {
  return drawn;
}

/**
 * ADR 0128 D2 startup of an agent mode: its arguments, its `--out` (absent
 * or an empty folder: spec/job.ts), Chromium's profile under `--out`
 * (INV-1), then the render job. Any refusal or unexpected error goes to
 * stderr and exits 2 at once, before any window, dialog or Chromium write:
 * a thrown error at this level would open a modal and hang the run, with the
 * profile still in %APPDATA% (review of #2660).
 */
export function agentStartup(study: Study): { refused: boolean; job: RenderJobFiles | null } {
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
    renderJob = p.job;
    renderStudy = study;
    drawn = { model: p.job.model, geometry: p.job.geometry, results: p.job.results };
    for (const s of p.job.spec.shots) unwritten.add(s.name);
    return { refused: false, job: p.job };
  } catch (err) {
    return refuse(`${mode} failed to start: ${err instanceof Error ? err.message : String(err)}`);
  }
}

/** Where the run stopped, for its refusal: the shot in progress (render) or the last still written (capture). */
export function whereStopped(): string {
  if (mode === "render" && renderJob) {
    // A shot taken from `unwritten` but not yet in `written` (its capture in flight) is the one in progress.
    const next = renderJob.spec.shots.find((s) => !written.has(s.name));
    return next ? `it stopped on shot ${next.name}` : "every still had been written";
  }
  if (mode === "capture") return lastCaptured ? `the last still written was ${lastCaptured}` : "no still had been written";
  return "";
}

/**
 * INV-6 when a render cannot reach `render:done`: the renderer held the
 * manifest entries, so the stills main has written have none and are removed,
 * and manifest.json refuses the run, naming them and why. A manifest already
 * written (the run had ended) is left as it is.
 */
export function abandonRender(reason: string): void {
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

/** Register the `capture:*` and `render:*` handlers, each once. */
export function registerRender(host: RenderHost): void {
  const { headlessEnd, errText, shown, resultsHost, isEnding, markEnding, rowSet, closeCompare } = host;

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
    if (isEnding()) throw new Error(`${mode}: the run is ending; ${path} is not written`);
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

  /** End an agent mode with its exit code, the results file closed first. */
  function agentExit(code: number): void {
    markEnding();
    resultsHost.close();
    app.exit(code);
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

  // -------------------------------------------------------------------------
  // ADR 0128 D4 agent modes: the renderer (renderer/agentRender.ts) asks for its
  // job, hands over each still, and reports the run; main writes every file,
  // only into `--out` (INV-1), and only a PNG named by a shot of the spec.
  // -------------------------------------------------------------------------

  ipcMain.handle("render:job", () => {
    // The job names the spec's own set; a shot's variant row is asked for with `render:variant`.
    if (mode === "render" && renderJob) return { mode, ...renderJob };
    if (mode === "describe") {
      const set = pairSet(resolve(file!), existsSync);
      return { mode, model: set.model, geometry: set.geometry, results: set.results };
    }
    throw new Error(`render:job: not an agent mode (--mode=${mode})`);
  });

  /**
   * VU18 (ADR 0132 D9): the set the next shot draws. `null` is the spec's own
   * set. A row `{ index, label }` (never a path, D9.2) is resolved as the switch
   * door resolves one: the manifest read again now (D9.3, `Study.rowNow`), the
   * row matched by index and label and resolved and confined again
   * (`Study.row`), its set paired with each sibling confined (`rowSet`, D9.4).
   * It then becomes the set the window's doors read (`drawnSet`), and the page
   * is told which files to open. A row that does not answer leaves the drawn
   * set as it was and answers why, for the shot's refusal.
   */
  ipcMain.handle("render:variant", (_e, asked: unknown) => {
    if (mode !== "render" || !renderJob || !renderStudy) throw new Error("render:variant: only in --mode=render");
    const study = renderStudy;
    const job = renderJob;
    const refuse = (reason: string) => ({ ok: false as const, reason: `render:variant: ${reason}` });
    if (asked === null) {
      const own = study.rowOfModel(job.model);
      if (own !== null && study.compare?.row.index === own.index) closeCompare();
      drawn = { model: job.model, geometry: job.geometry, results: job.results };
      return { ok: true as const, row: null, set: drawn };
    }
    if (job.variants === null) return refuse("the spec names no files.variants, so a shot names no variant row");
    const keys = typeof asked === "object" && !Array.isArray(asked) ? Object.keys(asked).sort().join(",") : "";
    if (keys !== "index,label") return refuse(`a variant row is named by { index, label } and never a path; got ${shown(asked)}`);
    const { index, label } = asked as { index: unknown; label: unknown };
    const r = study.rowNow(index, label);
    if (!r.ok) return refuse(r.reason);
    const s = rowSet(r.open);
    if (!s.ok) return refuse(`variant row ${r.row.index} (${JSON.stringify(r.row.label)}): ${s.reason}`);
    if (s.set.model === null) return refuse(`variant row ${r.row.index} (${JSON.stringify(r.row.label)}) pairs no model from ${r.open}`);
    if (study.compare?.row.index === r.row.index) closeCompare();
    drawn = { model: s.set.model, geometry: s.set.geometry, results: s.set.results };
    return { ok: true as const, row: { index: r.row.index, label: r.row.label }, set: drawn };
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
}
