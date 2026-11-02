// `npm run profile -- <model.h5>`: drive a real view-mode window over the
// DevTools protocol through named scenarios, and name what costs most.
//
//   node scripts/profile.ts <model.h5> [--out=<dir>] [--scenarios=a,b] [--reps=5]
//                           [--sampling=200] [--hard-cap=1800] [--keep-profiles]
//
// Writes <dir>/result.json and <dir>/summary.md (default out/profile/<stem>-<time>/).
// Exit 0: every scenario that ran proved its change. 1: one failed or proved
// nothing. 2: the command itself was wrong.
//
// It starts only the Electron binary from node_modules, on a scratch profile
// that it deletes at the end, and stops only the process it started. The app's
// source has no hook for this: the store is reached with a one-shot debugger
// breakpoint in Store.dispatch (scripts/profile/probe.ts).

import electronBinary from "electron";
import { execFileSync, spawn, type ChildProcess } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { createServer } from "node:net";
import { cpus, tmpdir } from "node:os";
import { basename, dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { readPng, type Png } from "../test/spec/png.ts";
import {
  cpuBusyPct,
  dispatchBreakLine,
  exitCodeOf,
  frameStats,
  NOISY_LOAD_PCT,
  pixelDelta,
  rankSubscribers,
  renderSummary,
  sourceMapper,
  summarizeProfile,
  verdict,
  type CpuProfile,
  type CpuTimes,
  type Mapper,
  type RunResult,
  type ScenarioResult,
  type Status,
  type SubscriberRaw,
} from "./profile/analyze.ts";
import { Cdp, findTarget, TOOL_URL } from "./profile/cdp.ts";
import { installProbe, type Probe } from "./profile/probe.ts";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");

const SCENARIOS = [
  "load",
  "orbit",
  "store-event-mesh",
  "enter-results",
  "contour-on",
  "scrub",
  "switch-contour",
  "switch-contour-element",
  "toggle-deformed",
  "orbit-contour",
  "hover",
  "store-event-results",
] as const;
type ScenarioName = (typeof SCENARIOS)[number];

function usage(message: string): never {
  console.error(`${message}\nnpm run profile -- <model.h5> [--out=<dir>] [--scenarios=${SCENARIOS.slice(0, 2).join(",")},...] [--reps=5] [--sampling=200] [--hard-cap=1800] [--keep-profiles]\nscenarios: ${SCENARIOS.join(", ")}`);
  process.exit(2);
}

const argv = process.argv.slice(2);
const flag = (name: string) => argv.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3);
const known = ["out", "scenarios", "reps", "sampling", "hard-cap"];
const bad = argv.find((a) => a.startsWith("--") && a !== "--keep-profiles" && !known.some((k) => a.startsWith(`--${k}=`)));
if (bad) usage(`unknown flag ${bad}`);
const positional = argv.filter((a) => !a.startsWith("--"));
if (positional.length !== 1) usage("name exactly one model");
const file = resolve(positional[0]!);
if (!existsSync(file)) usage(`no such file: ${file}`);
const number = (name: string, dflt: number) => {
  const raw = flag(name);
  const v = raw === undefined ? dflt : Number(raw);
  if (!Number.isFinite(v) || v <= 0) usage(`--${name}=${raw} is not a positive number`);
  return v;
};
// Odd, so a toggle scenario ends in the state it changed to, which is what the pixel proof compares.
const reps = Math.floor(number("reps", 5)) | 1;
const samplingUs = number("sampling", 200);
const hardCapMs = number("hard-cap", 1800) * 1000;
const keepProfiles = argv.includes("--keep-profiles");
const wanted = new Set<string>((flag("scenarios") ?? SCENARIOS.join(",")).split(","));
for (const w of wanted) if (!(SCENARIOS as readonly string[]).includes(w)) usage(`unknown scenario ${w}`);
const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
const outDir = resolve(flag("out") ?? join(root, "out", "profile", `${basename(file).replace(/\.[^.]+$/, "")}-${stamp}`));
mkdirSync(outDir, { recursive: true });

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

function freePort(): Promise<number> {
  return new Promise((res, rej) => {
    const s = createServer();
    s.once("error", rej);
    s.listen(0, "127.0.0.1", () => {
      const port = (s.address() as { port: number }).port;
      s.close(() => res(port));
    });
  });
}

/** Whether a process this run started is still running, by its own exit event (a sandboxed shell cannot probe a PID). */
const running = (cp: ChildProcess | null): boolean => cp !== null && cp.pid !== undefined && cp.exitCode === null && cp.signalCode === null;

/** Stop a process this run started, and its children: by PID, never by name. */
function stop(pid: number): void {
  try {
    if (process.platform === "win32") execFileSync("taskkill", ["/PID", String(pid), "/T", "/F"], { stdio: "ignore" });
    else process.kill(pid, "SIGKILL");
  } catch {
    // gone between the check and the kill
  }
}

function cpuTimes(): CpuTimes {
  let idle = 0, total = 0;
  for (const c of cpus()) {
    idle += c.times.idle;
    total += c.times.user + c.times.nice + c.times.sys + c.times.idle + c.times.irq;
  }
  return { idle, total };
}

function commit(): string {
  try {
    return execFileSync("git", ["rev-parse", "--short=8", "HEAD"], { cwd: root, encoding: "utf8" }).trim();
  } catch {
    return "unknown";
  }
}

/** Source maps of the two bundles; a script without one keeps its own name. */
function loadMapper(): Mapper {
  const mappers = new Map<string, ReturnType<typeof sourceMapper>>();
  for (const name of ["renderer.js", "main.mjs"]) {
    const path = join(root, "dist", `${name}.map`);
    if (existsSync(path)) mappers.set(name, sourceMapper(JSON.parse(readFileSync(path, "utf8"))));
  }
  return (url, line, column) => mappers.get(url.split("/").pop() ?? "")?.(line, column) ?? null;
}

// ---- the page ---------------------------------------------------------------

/** Runs a page expression over the state; `s` is the store's state. */
const state = <T>(page: Cdp, body: string) => page.evaluate<T>(`(() => { const s = globalThis.__agvProbe.store.get(); return (${body}); })()`);
const dispatch = (page: Cdp, event: unknown) => page.evaluate<void>(`void globalThis.__agvProbe.store.dispatch(${JSON.stringify(event)})`);
const twoFrames = (page: Cdp) => page.evaluate<void>("new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(() => setTimeout(r, 0))))");

interface Wait {
  finished: boolean;
  ms: number;
  lastSeen: string;
  /** what the page shows instead of the job, when it refused it (see waitFor `refusal`) */
  refused?: string;
}

/** The legend's note when a contour is not drawn (panels/legend.ts): a refusal ends the wait, a "Not drawn" that persists too. */
const CONTOUR_REFUSAL = `([...document.querySelectorAll("#legend .legend-note")].map((n) => n.textContent ?? "").find((t) => /^(Refused|Nothing to colour|Not drawn):/.test(t)) ?? null)`;
/** How long a "Not drawn" note may stand before the wait takes it as the answer. */
const NOT_YET_MS = 30_000;

/**
 * Wait for a state predicate, then for two painted frames. A wait never gives
 * up at a budget: the budget only makes the verdict `slow`. Only the hard cap
 * ends it, and the predicate is read once more first, so a finished job can
 * never be reported as a timeout (#2669 item 4).
 */
async function waitFor(page: Cdp, predicate: string, describe: string, startedAt = performance.now(), refusal: string | null = null): Promise<Wait> {
  const check = () => state<{ ok: boolean; seen: string; refused: string | null }>(page, `{ ok: !!(${predicate}), seen: JSON.stringify(${describe}), refused: ${refusal ?? "null"} }`);
  let last = await check();
  let notYetSince: number | null = null;
  while (!last.ok && performance.now() - startedAt < hardCapMs) {
    if (last.refused && !last.refused.startsWith("Not drawn")) break;
    notYetSince = last.refused ? (notYetSince ?? performance.now()) : null;
    if (notYetSince !== null && performance.now() - notYetSince > NOT_YET_MS) break;
    await sleep(10);
    last = await check();
  }
  const ms = performance.now() - startedAt;
  if (last.ok) await twoFrames(page);
  else if (last.refused) return { finished: true, ms, lastSeen: last.seen, refused: last.refused };
  return { finished: last.ok, ms, lastSeen: last.seen };
}

async function rectOf(page: Cdp, selector: string): Promise<{ x: number; y: number; width: number; height: number } | null> {
  return page.evaluate(`(() => { const e = document.querySelector(${JSON.stringify(selector)}); if (!e || e.disabled) return null; e.scrollIntoView({ block: "center" }); const r = e.getBoundingClientRect(); if (!r.width || !r.height) return null; return { x: r.x, y: r.y, width: r.width, height: r.height }; })()`);
}

async function mouse(page: Cdp, type: "mousePressed" | "mouseReleased" | "mouseMoved", x: number, y: number, button: "left" | "right" | "none" = "none", buttons = 0): Promise<void> {
  await page.send("Input.dispatchMouseEvent", { type, x, y, button, buttons, clickCount: type === "mouseMoved" ? 0 : 1 });
}

/** A real click at an element's centre, through the page's hit test: a panel covering it takes the click, as for a user. */
async function click(page: Cdp, selector: string): Promise<void> {
  const r = await rectOf(page, selector);
  if (!r) throw new Error(`${selector} is not on the page, or is disabled`);
  const x = r.x + r.width / 2, y = r.y + r.height / 2;
  await mouse(page, "mouseMoved", x, y);
  await mouse(page, "mousePressed", x, y, "left", 1);
  await mouse(page, "mouseReleased", x, y, "left", 0);
}

/** A button of the legend by its label. */
async function clickLegendButton(page: Cdp, label: string): Promise<void> {
  const tag = await page.evaluate<boolean>(`(() => { document.querySelector("[data-agv-target]")?.removeAttribute("data-agv-target"); const b = [...document.querySelectorAll("#legend button")].find((x) => x.textContent === ${JSON.stringify(label)}); if (!b) return false; b.setAttribute("data-agv-target", "1"); return true; })()`);
  if (!tag) throw new Error(`the legend has no "${label}" button`);
  await click(page, "[data-agv-target]");
}

async function canvasRect(page: Cdp) {
  const r = await rectOf(page, "#viewport canvas");
  if (!r) throw new Error("no viewport canvas");
  return r;
}

async function screenshot(page: Cdp): Promise<Png> {
  const r = await canvasRect(page);
  const shot = await page.send("Page.captureScreenshot", { format: "png", clip: { ...r, scale: 1 } });
  return readPng(Buffer.from(shot.data, "base64"));
}

/** The point the drag or hover uses: the canvas's centre, which must be the canvas and not a panel. */
async function canvasCentre(page: Cdp) {
  const r = await canvasRect(page);
  const x = r.x + r.width / 2, y = r.y + r.height / 2;
  const top = await page.evaluate<string>(`document.elementFromPoint(${x}, ${y})?.tagName ?? "none"`);
  if (top !== "CANVAS") throw new Error(`the viewport's centre is covered by a ${top}`);
  return { x, y };
}

// ---- one scenario -----------------------------------------------------------

interface Outcome {
  wait: Wait;
  repMs: number[];
  softMs: number;
  /** null when the store changed what it should; else what is wrong */
  stateProblem: string | null;
  proof: string | null;
  /** false: a scenario with no visual claim, whose proof is the store alone */
  visual: boolean;
  skip?: string;
}

/**
 * The app refused what the scenario asked for (a contour the file cannot
 * support): the scenario does not apply to this model, as with no results. It
 * is never `ok`, since a refused contour paints the no-data grey, which
 * changes pixels; the legend's own sentence is the reason.
 */
const refusedBy = (wait: Wait): Outcome => ({ ...skipped(`the app refused it; the legend says "${wait.refused}"`), wait, repMs: [wait.ms] });

const skipped = (why: string): Outcome => ({ wait: { finished: true, ms: 0, lastSeen: "" }, repMs: [], softMs: 0, stateProblem: null, proof: null, visual: false, skip: why });

type Metrics = Record<string, number>;
const readMetrics = async (page: Cdp): Promise<Metrics> => {
  const r = await page.send("Performance.getMetrics");
  return Object.fromEntries((r.metrics as { name: string; value: number }[]).map((m) => [m.name, m.value]));
};

class Run {
  readonly results: ScenarioResult[] = [];
  readonly warnings: string[] = [];
  private rawSubs = new Map<string, SubscriberRaw[]>();
  private page: Cdp;
  private main: Cdp | null;
  private mapper: Mapper;

  constructor(page: Cdp, main: Cdp | null, mapper: Mapper) {
    this.page = page;
    this.main = main;
    this.mapper = mapper;
  }

  /** Whether the probe is up: everything but the first load needs it. */
  private async probed(): Promise<boolean> {
    return this.page.evaluate<boolean>("!!globalThis.__agvProbe");
  }

  /** Set when the app exits mid-run: every later scenario is reported as not run, with this reason. */
  aborted: string | null = null;

  async scenario(name: ScenarioName, what: string, setup: (() => Promise<string | null>) | null, body: () => Promise<Outcome>, opts: { sampling?: number } = {}): Promise<void> {
    if (!wanted.has(name)) return;
    const { page, main } = this;
    if (this.aborted) {
      this.results.push({ ...emptyScenario(name, what), status: "failed", reason: `not run: ${this.aborted}` });
      console.log(`profile: ${name}: not run (${this.aborted})`);
      return;
    }
    console.log(`profile: ${name} ...`);
    const started = performance.now();
    let outcome: Outcome | null = null;
    let error: string | null = null;
    let before: Png | null = null;
    let after: Png | null = null;
    let cpuBefore = cpuTimes();
    let metricsBefore: Metrics = {};
    let profiling = false;
    const sampling = opts.sampling ?? samplingUs;
    let rendererProfile: CpuProfile | null = null;
    let mainProfile: CpuProfile | null = null;
    let live: ReturnType<Probe["read"]> | null = null;
    try {
      const why = setup ? await setup() : null;
      if (why) outcome = skipped(why);
      else {
        before = await screenshot(page);
        if (await this.probed()) await page.evaluate("globalThis.__agvProbe.reset()");
        metricsBefore = await readMetrics(page);
        await page.send("Profiler.enable");
        await page.send("Profiler.setSamplingInterval", { interval: sampling });
        await page.send("Profiler.start");
        if (main) {
          await main.send("Profiler.enable");
          await main.send("Profiler.setSamplingInterval", { interval: sampling });
          await main.send("Profiler.start");
        }
        profiling = true;
        cpuBefore = cpuTimes();
        outcome = await body();
      }
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
    }
    const cpuAfter = cpuTimes();
    const elapsed = performance.now() - started;
    if (profiling) {
      try {
        rendererProfile = (await page.send("Profiler.stop")).profile;
        if (main) mainProfile = (await main.send("Profiler.stop")).profile;
        if (await this.probed()) live = await page.evaluate<ReturnType<Probe["read"]>>("globalThis.__agvProbe.read()");
      } catch (err) {
        error ??= err instanceof Error ? err.message : String(err);
      }
    }
    const metricsAfter = profiling && !page.closed ? await readMetrics(page).catch(() => metricsBefore) : metricsBefore;
    let pixels: number | null = null;
    if (outcome?.visual && before && !outcome.skip && !error) {
      try {
        await sleep(150);
        await twoFrames(page);
        after = await screenshot(page);
        pixels = pixelDelta(before, after);
      } catch (err) {
        error = err instanceof Error ? err.message : String(err);
      }
    }
    let status: Status, reason: string | null;
    if (this.aborted) ({ status, reason } = { status: "failed", reason: `${this.aborted}${error ? ` (${error})` : ""}` });
    else if (error) ({ status, reason } = { status: "failed", reason: error });
    else if (outcome!.skip) ({ status, reason } = { status: "skipped", reason: outcome!.skip });
    else {
      ({ status, reason } = verdict({
        finished: outcome!.wait.finished,
        lastSeen: outcome!.wait.lastSeen,
        elapsedMs: outcome!.wait.ms,
        softMs: outcome!.softMs,
        stateProblem: outcome!.stateProblem,
        pixels: outcome!.visual ? pixels : null,
      }));
    }
    const load = cpuBusyPct(cpuBefore, cpuAfter);
    const longs = live?.longTasks ?? [];
    const frames = frameStats(live?.intervals ?? []);
    if (keepProfiles && rendererProfile) writeFileSync(join(outDir, `${name}.renderer.cpuprofile`), JSON.stringify(rendererProfile));
    if (keepProfiles && mainProfile) writeFileSync(join(outDir, `${name}.main.cpuprofile`), JSON.stringify(mainProfile));
    this.rawSubs.set(name, live?.subscribers ?? []);
    this.results.push({
      name,
      what,
      status,
      reason,
      durationMs: status === "skipped" && !profiling ? 0 : elapsed,
      repMs: outcome?.repMs ?? [],
      proof: { state: outcome?.proof ?? null, pixelsChangedPct: pixels === null ? null : pixels * 100 },
      frames,
      longTasks: { count: longs.length, totalMs: longs.reduce((a, b) => a + b, 0), maxMs: longs.length ? Math.max(...longs) : 0 },
      renderer: rendererProfile ? summarizeProfile(rendererProfile, this.mapper) : null,
      main: mainProfile ? summarizeProfile(mainProfile, this.mapper) : null,
      subscribers: [],
      events: live?.events ?? {},
      machineCpuPct: load,
      noisy: load > NOISY_LOAD_PCT,
      metrics: {
        taskMs: ((metricsAfter["TaskDuration"] ?? 0) - (metricsBefore["TaskDuration"] ?? 0)) * 1000,
        scriptMs: ((metricsAfter["ScriptDuration"] ?? 0) - (metricsBefore["ScriptDuration"] ?? 0)) * 1000,
        layoutMs: ((metricsAfter["LayoutDuration"] ?? 0) - (metricsBefore["LayoutDuration"] ?? 0)) * 1000,
        recalcStyleMs: ((metricsAfter["RecalcStyleDuration"] ?? 0) - (metricsBefore["RecalcStyleDuration"] ?? 0)) * 1000,
        heapUsedMB: (metricsAfter["JSHeapUsedSize"] ?? 0) / 1048576,
      },
    });
    console.log(`profile: ${name}: ${status}${reason ? ` (${reason})` : ""} in ${Math.round(elapsed)} ms, machine ${Math.round(load)}%`);
  }

  /** Names every subscriber by where its function is defined, then ranks each scenario's. */
  async nameSubscribers(): Promise<void> {
    if (!(await this.probed())) return;
    const count = await this.page.evaluate<number>("globalThis.__agvProbe.seen.length");
    const names = new Map<number, string>();
    for (let i = 0; i < count; i++) {
      const fn = await this.page.send("Runtime.evaluate", { expression: `globalThis.__agvProbe.seen[${i}]` });
      const props = await this.page.send("Runtime.getProperties", { objectId: fn.result.objectId, ownProperties: true });
      const loc = (props.internalProperties as { name: string; value?: { value?: { lineNumber: number; columnNumber: number } } }[] | undefined)?.find((p) => p.name === "[[FunctionLocation]]")?.value?.value;
      const fnName = (props.result as { name: string; value?: { value?: string } }[]).find((p) => p.name === "name")?.value?.value || "(anonymous)";
      const at = loc ? this.mapper("renderer.js", loc.lineNumber, loc.columnNumber) : null;
      names.set(i, at ? `${at.file}:${fnName}:${at.line}` : `subscriber ${i} (${fnName})`);
    }
    for (const r of this.results) r.subscribers = rankSubscribers(this.rawSubs.get(r.name) ?? [], (id) => names.get(id) ?? `subscriber ${id}`);
  }
}

function emptyScenario(name: string, what: string): ScenarioResult {
  return {
    name,
    what,
    status: "skipped",
    reason: null,
    durationMs: 0,
    repMs: [],
    proof: { state: null, pixelsChangedPct: null },
    frames: null,
    longTasks: { count: 0, totalMs: 0, maxMs: 0 },
    renderer: null,
    main: null,
    subscribers: [],
    events: {},
    machineCpuPct: NaN,
    noisy: false,
    metrics: {},
  };
}

// ---- the session --------------------------------------------------------------

/** Arm the one-shot breakpoint: the next dispatch hands over the store, then the debugger leaves. */
async function armStoreCapture(page: Cdp): Promise<void> {
  const scripts: { scriptId: string; url: string }[] = [];
  page.on("Debugger.scriptParsed", (p) => scripts.push(p as unknown as { scriptId: string; url: string }));
  await page.send("Debugger.enable");
  // The replay of what already parsed can come after the reply to `enable`.
  const replayed = performance.now();
  while (!scripts.some((s) => s.url.endsWith("renderer.js")) && performance.now() - replayed < 15_000) await sleep(50);
  const script = scripts.find((s) => s.url.endsWith("renderer.js"));
  if (!script) throw new Error(`renderer.js is not loaded; scripts: ${scripts.map((s) => s.url).join(", ")}`);
  const { scriptSource } = await page.send("Debugger.getScriptSource", { scriptId: script.scriptId });
  const bp = await page.send("Debugger.setBreakpoint", { location: { scriptId: script.scriptId, lineNumber: dispatchBreakLine(scriptSource) } });
  page.on("Debugger.paused", (p) => {
    const frame = (p as { callFrames: { callFrameId: string }[] }).callFrames[0]!;
    void (async () => {
      const r = await page.send("Debugger.evaluateOnCallFrame", { callFrameId: frame.callFrameId, expression: `(${installProbe.toString()}).call(this)\n//# sourceURL=${TOOL_URL}` });
      if (r.exceptionDetails) console.error(`profile: the probe failed: ${r.exceptionDetails.exception?.description}`);
      await page.send("Debugger.removeBreakpoint", { breakpointId: bp.breakpointId });
      await page.send("Debugger.resume");
      await page.send("Debugger.disable");
    })();
  });
}

const RESULTS_READY = `s.resultStep !== null && s.resultStep.status !== "loading"`;
const SCAN_SETTLED = `(s.contour.scan === null || s.contour.scan.status !== "running")`;
/** The scan is the shown pick's, not the last field's: its key ends in the pick (state/contour.ts contourKeysOf). */
const SCAN_OF_PICK = `(s.contour.scan !== null && s.contour.scan.key.endsWith("|" + (s.contour.source !== "element" ? s.contour.field : s.contour.gauss === "cell" ? "elem:" + s.contour.field : "elem:" + s.contour.field + "@" + s.contour.gauss) + "|" + s.contour.component))`;

async function runScenarios(page: Cdp, run: Run, openFile: () => void): Promise<void> {
  // Each scenario builds on the last: the model loaded, then the results stage, then a contour.
  await run.scenario(
    "load",
    "open the file in the waiting window, to the first painted frame of the mesh",
    null,
    async () => {
      const t = performance.now();
      openFile();
      // The probe arrives with the first dispatch; until then there is no state to read.
      while (!(await page.evaluate<boolean>("!!globalThis.__agvProbe")) && performance.now() - t < hardCapMs) await sleep(10);
      if (!(await page.evaluate<boolean>("!!globalThis.__agvProbe"))) {
        return { wait: { finished: false, ms: performance.now() - t, lastSeen: "no store event at all since the file was handed over" }, repMs: [], softMs: 30_000, stateProblem: null, proof: null, visual: false };
      }
      const wait = await waitFor(page, `s.mesh !== null || s.artifacts.model?.status === "failed"`, `s.artifacts.model?.status ?? "no model yet"`, t);
      const problem = await state<string | null>(page, `s.mesh !== null ? null : ("the model did not load: " + (s.artifacts.model?.error ?? "no error given"))`);
      return { wait, repMs: [wait.ms], softMs: 30_000, stateProblem: wait.finished ? problem : null, proof: `mesh drawn`, visual: true };
    },
    { sampling: Math.max(samplingUs, 1000) },
  );
  const loaded = await page.evaluate<boolean>("!!globalThis.__agvProbe && globalThis.__agvProbe.store.get().mesh !== null");
  if (!loaded) {
    run.warnings.push("the model did not load, so no later scenario ran");
    return;
  }
  const orbit = async (): Promise<Outcome> => {
    const { x, y } = await canvasCentre(page);
    const t = performance.now();
    await mouse(page, "mouseMoved", x, y);
    await mouse(page, "mousePressed", x, y, "right", 2);
    const MOVES = 120;
    for (let i = 1; i <= MOVES; i++) {
      await mouse(page, "mouseMoved", x + i * 3, y + Math.sin(i / 10) * 20, "right", 2);
      await sleep(8);
    }
    await mouse(page, "mouseReleased", x + MOVES * 3, y, "right", 0);
    await twoFrames(page);
    const ms = performance.now() - t;
    // The drag's wall time is the driver's pacing and the machine's load; its cost is in the frames and the profile.
    return { wait: { finished: true, ms, lastSeen: "" }, repMs: [ms / MOVES], softMs: 120_000, stateProblem: null, proof: "the camera moved", visual: true };
  };
  await run.scenario("orbit", "right-drag orbit over the canvas, 120 real mouse moves", null, orbit);

  /** The cheapest real store change: `hover` alternates between one element and nothing. */
  const storeEvent = async (): Promise<Outcome> => {
    const count = Math.max(reps * 8, 40);
    const rep: number[] = await page.evaluate(`(async () => {
      const store = globalThis.__agvProbe.store;
      const decl = store.get().mesh.elements[0];
      const ms = [];
      for (let i = 0; i < ${count}; i++) {
        const t = performance.now();
        store.dispatch({ type: "hover", at: i % 2 === 0 ? decl : null });
        ms.push(performance.now() - t);
        await new Promise((r) => setTimeout(r, 0));
      }
      return ms;
    })()`);
    const seen = await page.evaluate<{ hover?: number }>("globalThis.__agvProbe.read().events");
    const ran = await page.evaluate<number>("globalThis.__agvProbe.read().subscribers.length");
    const problem = (seen.hover ?? 0) !== count ? `${seen.hover ?? 0} of ${count} hover events dispatched` : ran === 0 ? "no subscriber ran" : null;
    const total = rep.reduce((a, b) => a + b, 0);
    return { wait: { finished: true, ms: total, lastSeen: "" }, repMs: rep, softMs: 16 * count, stateProblem: problem, proof: `${count} hover events, ${ran} subscribers ran`, visual: false };
  };
  await run.scenario("store-event-mesh", "one store change (hover on/off), mesh phase: the cost of every subscriber", null, storeEvent);

  const hasResults = await (async () => {
    // A results sibling opens after the model; give it time, and report if it never settles.
    const wait = await waitFor(page, `s.artifacts.results === null || s.artifacts.results.status !== "opened" || s.phase.axis.some((k) => k.kind === "results")`, `s.artifacts.results?.status ?? null`);
    if (!wait.finished) run.warnings.push(`the results file did not settle: ${wait.lastSeen}`);
    return state<boolean>(page, `s.phase.axis.some((k) => k.kind === "results")`);
  })();
  const noResults = async () => {
    const why = await state<string>(page, `s.artifacts.results ? ("the results did not pair: " + (s.artifacts.results.error ?? s.artifacts.results.status)) : "no results file beside the model"`);
    return why;
  };
  const needResults = (next: (() => Promise<string | null>) | null = null) => async () => (hasResults ? (next ? next() : null) : noResults());

  await run.scenario("enter-results", "click the results phase, to its first step shown", needResults(), async () => {
    const t = performance.now();
    await page.evaluate(`document.querySelector("[data-agv-target]")?.removeAttribute("data-agv-target"); [...document.querySelectorAll("#phase button.phase")].find((b) => b.textContent === "results")?.setAttribute("data-agv-target", "1")`);
    await click(page, "[data-agv-target]");
    const wait = await waitFor(page, `s.phase.at?.kind === "results" && ${RESULTS_READY}`, `{ at: s.phase.at, step: s.resultStep }`, t);
    const problem = await state<string | null>(page, `s.phase.at?.kind !== "results" ? "the phase did not change" : s.resultStep?.status === "failed" ? ("the step failed: " + s.resultStep.error) : null`);
    return { wait, repMs: [wait.ms], softMs: 5000, stateProblem: wait.finished ? problem : null, proof: "on the results phase", visual: false };
  });

  const onResults = () => state<boolean>(page, `s.phase.at?.kind === "results"`);

  await run.scenario(
    "contour-on",
    "click contour in the legend, to the whole-history scan done and the contour painted",
    needResults(async () => ((await onResults()) ? null : "the results phase was not entered")),
    async () => {
      const t = performance.now();
      await clickLegendButton(page, "contour");
      const wait = await waitFor(page, `s.visibility.colourBy === "contour" && (s.contour.field === null || (${SCAN_OF_PICK} && ${SCAN_SETTLED})) && ${RESULTS_READY}`, `{ by: s.visibility.colourBy, field: s.contour.field, scan: s.contour.scan && [s.contour.scan.key, s.contour.scan.status, s.contour.scan.done, s.contour.scan.total] }`, t, CONTOUR_REFUSAL);
      if (wait.refused) return refusedBy(wait);
      const problem = await state<string | null>(page, `s.visibility.colourBy !== "contour" ? "the legend click changed nothing" : s.contour.field === null ? "no field to contour" : s.contour.scan?.status === "failed" ? ("the scan failed: " + s.contour.scan.error) : null`);
      const field = await state<string>(page, `String(s.contour.field) + " " + String(s.contour.component) + ", scan " + (s.contour.scan?.done ?? "?") + "/" + (s.contour.scan?.total ?? "?")`);
      return { wait, repMs: [wait.ms], softMs: 20_000, stateProblem: wait.finished ? problem : null, proof: field, visual: true };
    },
  );

  // The longest stage, so the scrub has steps to walk.
  await run.scenario(
    "scrub",
    "click step +1 and wait for the step painted, per step",
    needResults(async () => {
      if (!(await onResults())) return "the results phase was not entered";
      const best = await state<{ stage: number; steps: number }>(page, `(() => { let stage = 0, steps = 0; s.results.stages.forEach((st, i) => { if (st.time.length > steps) { steps = st.time.length; stage = i; } }); return { stage, steps }; })()`);
      if (best.steps < 2) return `the longest stage has ${best.steps} step`;
      await dispatch(page, { type: "setPhase", at: { kind: "results", stage: best.stage, step: 0 } });
      const w = await waitFor(page, RESULTS_READY, "s.resultStep");
      return w.finished ? null : `the first step never arrived: ${w.lastSeen}`;
    }),
    async () => {
      const steps = await state<number>(page, `s.results.stages[s.phase.at.stage].time.length`);
      const n = Math.min(steps - 1, reps * 4);
      const repMs: number[] = [];
      let wait: Wait = { finished: true, ms: 0, lastSeen: "" };
      let total = 0;
      for (let k = 1; k <= n; k++) {
        const t = performance.now();
        await click(page, "button.step-forward");
        wait = await waitFor(page, `s.resultStep?.step === ${k} && ${RESULTS_READY}`, "s.resultStep", t);
        total += wait.ms;
        if (!wait.finished) break;
        repMs.push(wait.ms);
      }
      const problem = await state<string | null>(page, `s.resultStep?.status === "failed" ? ("step failed: " + s.resultStep.error) : s.resultStep?.step === ${n} ? null : ("the step is " + s.resultStep?.step + ", not ${n}")`);
      return { wait: { ...wait, ms: total }, repMs, softMs: 500 * n, stateProblem: wait.finished ? problem : null, proof: `${repMs.length} steps walked`, visual: true };
    },
  );

  const fieldOptions = () => page.evaluate<{ value: string; selected: boolean }[]>(`[...document.querySelectorAll("#legend select.contour-field option")].filter((o) => !o.disabled).map((o) => ({ value: o.value, selected: o.selected }))`);
  const switchTo = async (pick: (opts: { value: string; selected: boolean }[]) => string | undefined): Promise<Outcome> => {
    const value = pick(await fieldOptions());
    const before = await state<string>(page, `String(s.contour.source) + ":" + String(s.contour.field)`);
    const t = performance.now();
    await page.evaluate(`(() => { const f = document.querySelector("#legend select.contour-field"); f.value = ${JSON.stringify(value)}; f.dispatchEvent(new Event("change", { bubbles: true })); })()`);
    const wait = await waitFor(page, `(String(s.contour.source) + ":" + String(s.contour.field)) === ${JSON.stringify(value)} && ${SCAN_OF_PICK} && ${SCAN_SETTLED} && ${RESULTS_READY}`, `{ field: s.contour.field, scan: s.contour.scan && [s.contour.scan.key, s.contour.scan.status, s.contour.scan.done, s.contour.scan.total] }`, t, CONTOUR_REFUSAL);
    const problem = await state<string | null>(page, `(String(s.contour.source) + ":" + String(s.contour.field)) === ${JSON.stringify(before)} ? "the field did not change" : s.contour.scan?.status === "failed" ? ("the scan failed: " + s.contour.scan.error) : s.resultStep?.status === "failed" ? ("the step failed: " + s.resultStep.error) : null`);
    const shown = await state<string>(page, `String(s.contour.field) + " " + String(s.contour.component) + ", scan " + (s.contour.scan?.done ?? "?") + "/" + (s.contour.scan?.total ?? "?")`);
    if (wait.refused) return refusedBy(wait);
    return { wait, repMs: [wait.ms], softMs: 20_000, stateProblem: wait.finished ? problem : null, proof: shown, visual: true };
  };
  await run.scenario(
    "switch-contour",
    "change the contour field in the legend to another nodal field, to its scan done and painted",
    needResults(async () => {
      if ((await fieldOptions()).filter((o) => o.value.startsWith("nodal:")).length < 2) return "fewer than two nodal contour fields";
      return null;
    }),
    () => switchTo((opts) => opts.find((o) => !o.selected && o.value.startsWith("nodal:"))?.value),
  );
  await run.scenario(
    "switch-contour-element",
    "change the contour field to an element (Gauss) field, to its scan done and painted (#2670)",
    needResults(async () => ((await fieldOptions()).some((o) => o.value.startsWith("element:") && !o.selected) ? null : "no element field in the picker")),
    () => switchTo((opts) => opts.find((o) => !o.selected && o.value.startsWith("element:"))?.value),
  );

  await run.scenario(
    "toggle-deformed",
    "click deformed in the legend, on and off, each to its shape painted",
    needResults(async () => {
      if (!(await onResults())) return "the results phase was not entered";
      const off = await page.evaluate<string | null>(`(() => { const b = document.querySelector("#legend button.deform-toggle"); return !b ? "no deformed button" : b.disabled ? ("the deformed shape is refused: " + b.title) : null; })()`);
      if (off) return off;
      // A real displacement can be below a pixel at true scale: find the scale at which the shape shows.
      const base = await screenshot(page);
      for (const scale of [1, 10, 100, 1000, 10000]) {
        await dispatch(page, { type: "setDeform", on: true });
        await dispatch(page, { type: "setDeformScale", scale });
        await waitFor(page, RESULTS_READY, "s.resultStep");
        await sleep(200);
        const shot = await screenshot(page);
        await dispatch(page, { type: "setDeform", on: false });
        await sleep(100);
        if (pixelDelta(base, shot) > 0.001) break;
      }
      await twoFrames(page);
      return null;
    }),
    async () => {
      const repMs: number[] = [];
      let wait: Wait = { finished: true, ms: 0, lastSeen: "" };
      let total = 0;
      for (let i = 0; i < reps; i++) {
        const on = i % 2 === 0;
        const t = performance.now();
        await click(page, "#legend button.deform-toggle");
        wait = await waitFor(page, `s.deform.on === ${on} && ${RESULTS_READY}`, "{ on: s.deform.on, step: s.resultStep }", t);
        total += wait.ms;
        if (!wait.finished) break;
        repMs.push(wait.ms);
      }
      const problem = await state<string | null>(page, `s.deform.on ? null : "the click did not turn the shape on"`);
      const scale = await state<number>(page, "s.deform.scale");
      return { wait: { ...wait, ms: total }, repMs, softMs: 500 * reps, stateProblem: wait.finished ? problem : null, proof: `on at scale ${scale}`, visual: true };
    },
  );

  await run.scenario("orbit-contour", "the orbit again, with the results, contour and deformed shape on", needResults(), orbit);

  await run.scenario(
    "hover",
    "real mouse moves over the canvas with a line diagram on (hover does nothing without one)",
    needResults(async () => {
      if (!(await onResults())) return "the results phase was not entered";
      const has = await state<boolean>(page, `(s.results.stages[s.phase.at.stage].lineStations?.length ?? 0) > 0`);
      if (!has) return "no line stations on this stage, so no line diagram: Viewport.hoverAt returns at once and hover changes nothing";
      await dispatch(page, { type: "setLineDiagram", on: true });
      const on = await state<boolean>(page, "s.lineDiagram.on");
      return on ? null : "the line diagram did not turn on";
    }),
    async () => {
      const { x, y } = await canvasCentre(page);
      const r = await canvasRect(page);
      const MOVES = 120;
      const t = performance.now();
      for (let i = 0; i < MOVES; i++) {
        // A slow diagonal sweep across the middle of the canvas, away from the panels.
        const u = i / MOVES - 0.5;
        await mouse(page, "mouseMoved", x + u * r.width * 0.4, y + u * r.height * 0.4);
        await sleep(8);
      }
      await twoFrames(page);
      const ms = performance.now() - t;
      const events = await page.evaluate<{ hover?: number }>("globalThis.__agvProbe.read().events");
      await dispatch(page, { type: "hover", at: null });
      await dispatch(page, { type: "setLineDiagram", on: false });
      return {
        wait: { finished: true, ms, lastSeen: "" },
        repMs: [ms / MOVES],
        softMs: 120_000,
        stateProblem: (events.hover ?? 0) === 0 ? `${MOVES} moves over the canvas produced no hover event: the sweep missed every beam` : null,
        proof: `${events.hover} hover events from ${MOVES} moves`,
        visual: false,
      };
    },
  );

  await run.scenario("store-event-results", "one store change (hover on/off), with results and contour on: the cost of every subscriber", needResults(), storeEvent);
}

// ---- main ---------------------------------------------------------------------

async function main(): Promise<number> {
  const scratch = mkdtempSync(join(tmpdir(), "agv-profile-"));
  const [pagePort, mainPort] = [await freePort(), await freePort()];
  const launchArgs = [root, "--mode=view", `--t0=${Date.now()}`, `--user-data-dir=${scratch}`];
  // `view` with no file, so the window and the profilers are up before the file opens: load is then measured whole.
  const child: ChildProcess = spawn(
    electronBinary as unknown as string,
    [...launchArgs, `--remote-debugging-port=${pagePort}`, `--inspect=${mainPort}`, "--disable-features=CalculateNativeWinOcclusion", "--disable-backgrounding-occluded-windows", "--disable-renderer-backgrounding"],
    { stdio: ["ignore", "ignore", "pipe"] },
  );
  let stderr = "";
  child.stderr!.on("data", (d) => {
    stderr = (stderr + String(d)).slice(-4000);
  });
  let exited = false;
  let run: Run | null = null;
  let page: Cdp | null = null;
  let mainCdp: Cdp | null = null;
  child.on("exit", (exitCode, signal) => {
    exited = true;
    // Every call still waiting on the app fails now, and no later scenario starts.
    if (run && !run.aborted) run.aborted = `the app exited mid-run (code ${exitCode ?? signal})`;
    page?.close();
    mainCdp?.close();
  });
  let code = 1;
  let opener: ChildProcess | null = null;
  const machine: RunResult["machine"] = { cpus: cpus()[0]?.model ?? "unknown", cores: cpus().length, gpu: "unknown", viewport: [], pixelRatio: NaN };
  let model: RunResult["model"] = { cells: 0, nodes: 0, results: null, steps: null };
  try {
    page = await Cdp.connect(await findTarget(pagePort, (t) => t.type === "page" && t.url.includes("index.html"), 60_000));
    const warnings: string[] = [];
    try {
      mainCdp = await Cdp.connect(await findTarget(mainPort, () => true, 10_000));
      await mainCdp.send("Runtime.enable");
    } catch (err) {
      warnings.push(`no main-process profile: ${err instanceof Error ? err.message : String(err)}`);
    }
    await page.send("Runtime.enable");
    await page.send("Page.enable");
    await page.send("Performance.enable");
    await page.evaluate("new Promise((r) => document.readyState === 'complete' ? r() : addEventListener('load', r))");
    await page.send("Page.bringToFront");
    await armStoreCapture(page);

    run = new Run(page, mainCdp, loadMapper());
    run.warnings.push(...warnings);
    // A second start on the same profile hands its file to the first window and exits: the app's own path for
    // "open with". `--file=`, because Chromium moves switches ahead of the app path in the argv it hands over, so a
    // positional would be read as the app folder.
    const openFile = () => {
      opener = spawn(electronBinary as unknown as string, [root, "--mode=view", `--user-data-dir=${scratch}`, `--file=${file}`], { stdio: "ignore" });
    };
    await runScenarios(page, run, openFile);
    if (run.aborted) throw new Error(run.aborted);
    await run.nameSubscribers();

    const info = await page.evaluate<{ visibility: string; viewport: number[]; pixelRatio: number; gpu: string }>(`(() => {
      const c = document.querySelector("#viewport canvas");
      const gl = document.createElement("canvas").getContext("webgl2");
      const ext = gl && gl.getExtension("WEBGL_debug_renderer_info");
      return { visibility: document.visibilityState, viewport: c ? [c.width, c.height] : [], pixelRatio: devicePixelRatio, gpu: ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : "unknown" };
    })()`);
    Object.assign(machine, { gpu: info.gpu, viewport: info.viewport, pixelRatio: info.pixelRatio });
    if (info.visibility !== "visible") run.warnings.push(`the page was ${info.visibility} at the end, so fps may be throttled`);
    for (const s of run.results) {
      if (s.status !== "skipped" && s.name !== "load" && s.name !== "store-event-mesh" && s.name !== "store-event-results" && s.frames === null) {
        run.warnings.push(`${s.name}: no animation frame during the scenario; the window may be occluded, so its fps says nothing`);
      }
    }
    model = await page.evaluate<RunResult["model"]>(`(() => { const s = globalThis.__agvProbe?.store.get(); const a = s?.artifacts.model?.counts; return { cells: a?.cells ?? 0, nodes: a?.nodes ?? 0, results: s?.artifacts.results?.path ?? null, steps: s?.results ? Math.max(...s.results.stages.map((t) => t.time.length)) : null }; })()`);
    code = writeResult(run, machine, model, null);
    await page.evaluate("void window.close()").catch(() => {});
  } catch (err) {
    const why = err instanceof Error ? err.message : String(err);
    console.error(`profile failed: ${why}`);
    if (exited) console.error(`electron had exited; its stderr: ${stderr}`);
    // What ran is still evidence: write it, marked as cut short.
    if (run) writeResult(run, machine, model, why);
    code = 1;
  } finally {
    // Only the processes this run started, by PID, with their own children. Give them time to leave on their own first.
    page?.close();
    mainCdp?.close();
    // A large model takes many seconds to tear down after its window closes.
    for (let i = 0; i < 300 && running(child); i++) await sleep(100);
    for (const cp of [child, opener as ChildProcess | null]) {
      if (!running(cp)) continue;
      stop(cp!.pid!);
      for (let i = 0; i < 30 && running(cp); i++) await sleep(100);
      if (running(cp)) console.error(`profile: process ${cp!.pid} (started by this run) is still running; stop it by that PID`);
    }
    await sleep(1000);
    try {
      rmSync(scratch, { recursive: true, force: true, maxRetries: 30, retryDelay: 500 });
    } catch (err) {
      console.error(`profile: the scratch profile ${scratch} was not removed: ${err instanceof Error ? err.message : String(err)}`);
    }
  }
  return code;
}

/** Writes result.json and summary.md; returns the run's exit code (1 when cut short). */
function writeResult(run: Run, machine: RunResult["machine"], model: RunResult["model"], abortedBy: string | null): number {
  const result: RunResult = {
    tool: "apegmsh-viewer-profile",
    version: 1,
    file,
    sizeBytes: statSync(file).size,
    date: new Date().toISOString().slice(0, 10),
    commit: commit(),
    machine,
    model,
    samplingUs,
    warnings: abortedBy ? [`the run was cut short: ${abortedBy}`, ...run.warnings] : run.warnings,
    scenarios: run.results,
  };
  writeFileSync(join(outDir, "result.json"), JSON.stringify(result, null, 1));
  const summary = renderSummary(result);
  writeFileSync(join(outDir, "summary.md"), summary);
  console.log(`\n${summary}`);
  console.log(`profile: wrote ${join(outDir, "result.json")} and summary.md`);
  return abortedBy ? 1 : exitCodeOf(result);
}

// A promise that never settles empties the event loop and Node would exit 0 with nothing written: say so, and fail.
let settled = false;
process.on("beforeExit", () => {
  if (settled) return;
  console.error("profile failed: the run stopped on a call that never settled (the app may have exited); nothing more was measured");
  process.exit(1);
});
const exitCode = await main();
settled = true;
process.exit(exitCode);
