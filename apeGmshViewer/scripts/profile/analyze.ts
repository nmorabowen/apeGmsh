// The pure half of `npm run profile`: everything that turns what the run
// recorded into numbers and a ranking. No process, no socket, no file: the
// driver is scripts/profile.ts, and test/profile/analyze.test.ts holds this.

// ---- source maps ------------------------------------------------------------

export interface RawSourceMap {
  sources: string[];
  mappings: string;
}

export interface SourcePos {
  file: string;
  /** 1-based, as an editor shows it */
  line: number;
}

const B64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
const B64_VALUE = new Map([...B64].map((c, i) => [c, i]));

/** One mapping line's segments as [column, source, line]; a segment with no source is dropped. */
function decodeLine(text: string, state: { source: number; line: number }): [number, number, number][] {
  const out: [number, number, number][] = [];
  let column = 0;
  for (const seg of text.split(",")) {
    if (!seg) continue;
    const fields: number[] = [];
    let value = 0, shift = 0;
    for (const c of seg) {
      const d = B64_VALUE.get(c);
      if (d === undefined) throw new Error(`source map: "${c}" is not base64 VLQ`);
      value |= (d & 31) << shift;
      if (d & 32) shift += 5;
      else {
        fields.push(value & 1 ? -(value >>> 1) : value >>> 1);
        value = 0;
        shift = 0;
      }
    }
    column += fields[0]!;
    if (fields.length < 4) continue;
    state.source += fields[1]!;
    state.line += fields[2]!;
    out.push([column, state.source, state.line]);
  }
  return out;
}

/** The lookup from a generated 0-based (line, column) to the source position, or null where nothing maps. */
export function sourceMapper(map: RawSourceMap): (line: number, column: number) => SourcePos | null {
  const state = { source: 0, line: 0 };
  const lines = map.mappings.split(";").map((l) => decodeLine(l, state));
  return (line, column) => {
    const segs = lines[line];
    if (!segs?.length) return null;
    let lo = 0, hi = segs.length - 1, hit = -1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (segs[mid]![0] <= column) {
        hit = mid;
        lo = mid + 1;
      } else hi = mid - 1;
    }
    const seg = segs[Math.max(hit, 0)]!;
    return { file: cleanSource(map.sources[seg[1]] ?? "?"), line: seg[2] + 1 };
  };
}

/** `../src/panels/loads.ts` is `src/panels/loads.ts`; a package keeps its node_modules path from the package on. */
export function cleanSource(source: string): string {
  const s = source.replace(/\\/g, "/");
  const nm = s.lastIndexOf("node_modules/");
  if (nm >= 0) return s.slice(nm);
  return s.replace(/^(\.\.?\/)+/, "");
}

// ---- the store ---------------------------------------------------------------

/** The Store.dispatch line to stop at: the first statement of the method, in the bundle. */
export function dispatchBreakLine(source: string): number {
  const m = /\bStore = class[\s\S]*?\bdispatch\(\w+\) \{\r?\n/.exec(source);
  if (!m) throw new Error("Store.dispatch was not found in dist/renderer.js: the probe has nothing to attach to");
  return source.slice(0, m.index + m[0].length).split("\n").length - 1;
}

// ---- CPU profiles -----------------------------------------------------------

/** The part of a DevTools `Profiler.Profile` this reads. */
export interface CpuProfile {
  nodes: {
    id: number;
    callFrame: { functionName: string; url: string; lineNumber: number; columnNumber: number };
    children?: number[];
  }[];
  samples?: number[];
  timeDeltas?: number[];
}

export interface FunctionCost {
  name: string;
  /** source file when a map knows it, else the script as the engine names it */
  file: string;
  line: number;
  selfMs: number;
  /** time with this function on the stack, recursion counted once */
  inclusiveMs: number;
}

export interface ProfileSummary {
  sampledMs: number;
  idleMs: number;
  /** native work with no JS frame: layout, paint and GPU submission show here */
  programMs: number;
  gcMs: number;
  /** sampledMs - idleMs: the time the thread was doing something */
  busyMs: number;
  /** gcMs as a percentage of busyMs */
  gcPct: number;
  /** the tool's own page code, taken out of every number here */
  toolMs: number;
  top: FunctionCost[];
}

/** The script name of the tool's own page code: its frames and their native calls are not the viewer's. */
export const TOOL_URL = "agv-probe";

const META = new Set(["(root)", "(idle)", "(program)", "(garbage collector)"]);

export type Mapper = (url: string, line: number, column: number) => SourcePos | null;

/**
 * Self and inclusive time per function. Self comes from the samples and their
 * time deltas; inclusive walks the call tree with the functions on the path,
 * so a recursive function is not counted twice.
 */
export function summarizeProfile(p: CpuProfile, mapper: Mapper = () => null, topN = 15): ProfileSummary {
  const selfByNode = new Map<number, number>();
  const samples = p.samples ?? [];
  const deltas = p.timeDeltas ?? [];
  for (let i = 0; i < samples.length; i++) {
    const id = samples[i]!;
    selfByNode.set(id, (selfByNode.get(id) ?? 0) + Math.max(deltas[i] ?? 0, 0));
  }
  const byId = new Map(p.nodes.map((n) => [n.id, n]));
  // The tool's own time comes out; what it calls in the viewer stays, so a dispatch it made still shows its subscribers.
  let toolUs = 0;
  for (const n of p.nodes) {
    if (n.callFrame.url !== TOOL_URL) continue;
    for (const id of [n.id, ...(n.children ?? [])]) {
      const f = byId.get(id)?.callFrame;
      if (!f || (id !== n.id && f.url)) continue;
      toolUs += selfByNode.get(id) ?? 0;
      selfByNode.delete(id);
    }
  }
  const childIds = new Set(p.nodes.flatMap((n) => n.children ?? []));
  const keyOf = (n: CpuProfile["nodes"][number]) => {
    const f = n.callFrame;
    return `${f.functionName}|${f.url}|${f.lineNumber}|${f.columnNumber}`;
  };
  const self = new Map<string, number>();
  const inclusive = new Map<string, number>();
  const frames = new Map<string, CpuProfile["nodes"][number]["callFrame"]>();
  for (const root of p.nodes.filter((n) => !childIds.has(n.id))) {
    // Iterative: a deep stack must not overflow this one. An exit entry takes the key off the path.
    const onPath = new Set<string>();
    const stack: { id: number; exit: boolean; owner: boolean }[] = [{ id: root.id, exit: false, owner: false }];
    const totals = new Map<number, number>();
    while (stack.length) {
      const top = stack.pop()!;
      const n = byId.get(top.id);
      if (!n) continue;
      const key = keyOf(n);
      if (top.exit) {
        let sum = selfByNode.get(n.id) ?? 0;
        for (const c of n.children ?? []) sum += totals.get(c) ?? 0;
        totals.set(n.id, sum);
        if (top.owner) {
          onPath.delete(key);
          inclusive.set(key, (inclusive.get(key) ?? 0) + sum);
        }
        continue;
      }
      frames.set(key, n.callFrame);
      self.set(key, (self.get(key) ?? 0) + (selfByNode.get(n.id) ?? 0));
      const owner = !onPath.has(key);
      if (owner) onPath.add(key);
      stack.push({ id: n.id, exit: true, owner });
      for (const c of n.children ?? []) stack.push({ id: c, exit: false, owner: false });
    }
  }
  let idle = 0, program = 0, gc = 0, total = 0;
  for (const [key, us] of self) {
    total += us;
    const name = frames.get(key)!.functionName;
    if (name === "(idle)") idle += us;
    else if (name === "(program)") program += us;
    else if (name === "(garbage collector)") gc += us;
  }
  const top: FunctionCost[] = [];
  for (const [key, f] of frames) {
    if (META.has(f.functionName) || f.url === TOOL_URL || (!f.url && !f.functionName)) continue;
    if (!self.get(key) && !inclusive.get(key)) continue;
    const at = mapper(f.url, f.lineNumber, f.columnNumber);
    top.push({
      name: f.functionName || "(anonymous)",
      file: at?.file ?? (f.url ? f.url.split("/").pop()! : "(native)"),
      line: at?.line ?? f.lineNumber + 1,
      selfMs: (self.get(key) ?? 0) / 1000,
      inclusiveMs: (inclusive.get(key) ?? 0) / 1000,
    });
  }
  top.sort((a, b) => b.selfMs - a.selfMs);
  const bySelf = top.slice(0, topN);
  const byInclusive = [...top].sort((a, b) => b.inclusiveMs - a.inclusiveMs).slice(0, topN);
  const merged = [...bySelf, ...byInclusive.filter((f) => !bySelf.includes(f))];
  const busy = total - idle;
  return {
    sampledMs: total / 1000,
    idleMs: idle / 1000,
    programMs: program / 1000,
    gcMs: gc / 1000,
    busyMs: busy / 1000,
    gcPct: busy > 0 ? (100 * gc) / busy : 0,
    toolMs: toolUs / 1000,
    top: merged,
  };
}

// ---- small statistics -------------------------------------------------------

/** The value at quantile q (0..1) by the nearest rank; NaN for no values. */
export function quantile(values: readonly number[], q: number): number {
  if (!values.length) return NaN;
  const s = [...values].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.max(0, Math.ceil(q * s.length) - 1))]!;
}

export interface FrameStats {
  frames: number;
  meanFps: number;
  medianFps: number;
  p95FrameMs: number;
  worstFrameMs: number;
}

/** From the requestAnimationFrame intervals of a scenario (ms). No interval is no frame, never a fps of 0 by accident. */
export function frameStats(intervals: readonly number[]): FrameStats | null {
  if (!intervals.length) return null;
  const total = intervals.reduce((a, b) => a + b, 0);
  const median = quantile(intervals, 0.5);
  return {
    frames: intervals.length,
    meanFps: total > 0 ? (1000 * intervals.length) / total : 0,
    medianFps: median > 0 ? 1000 / median : 0,
    p95FrameMs: quantile(intervals, 0.95),
    worstFrameMs: Math.max(...intervals),
  };
}

export interface CpuTimes {
  idle: number;
  total: number;
}

/** The machine's busy percentage between two samples of `os.cpus()` times summed over the cores. */
export function cpuBusyPct(before: CpuTimes, after: CpuTimes): number {
  const total = after.total - before.total;
  if (total <= 0) return NaN;
  return 100 * (1 - (after.idle - before.idle) / total);
}

/** Above this the machine competes with the run, and the summary marks the numbers as noisy. */
export const NOISY_LOAD_PCT = 70;

/** The fraction (0..1) of pixels that differ by more than `tol` on any channel; different sizes count as all changed. */
export function pixelDelta(a: { width: number; height: number; rgba: Uint8Array }, b: { width: number; height: number; rgba: Uint8Array }, tol = 8): number {
  if (a.width !== b.width || a.height !== b.height) return 1;
  const n = a.width * a.height;
  if (n === 0) return 0;
  let changed = 0;
  for (let i = 0; i < n; i++) {
    const p = i * 4;
    if (
      Math.abs(a.rgba[p]! - b.rgba[p]!) > tol ||
      Math.abs(a.rgba[p + 1]! - b.rgba[p + 1]!) > tol ||
      Math.abs(a.rgba[p + 2]! - b.rgba[p + 2]!) > tol
    ) {
      changed++;
    }
  }
  return changed / n;
}

// ---- verdicts ---------------------------------------------------------------

export type Status = "ok" | "slow" | "no-effect" | "skipped" | "failed";

export interface VerdictInput {
  /** the wait ended with the job done (a late finish is still a finish) */
  finished: boolean;
  /** what the wait saw last, for a job that never finished */
  lastSeen?: string;
  elapsedMs: number;
  softMs: number;
  /** the scenario's own check of the store: null when it changed what it should, else what is wrong */
  stateProblem: string | null;
  /** fraction of the viewport's pixels changed; null when the scenario makes no visual claim */
  pixels: number | null;
  /** below this fraction the view did not change */
  minPixels?: number;
}

/**
 * The one place a scenario's outcome is decided. A job that finished late is
 * `slow` with its true time, never a timeout; a job that really did not finish
 * is `failed`, saying what it last saw; and a scenario that ran but changed
 * nothing is `no-effect`, which fails the run: its timings measure nothing.
 */
export function verdict(v: VerdictInput): { status: Status; reason: string | null } {
  if (!v.finished) return { status: "failed", reason: `not finished after ${Math.round(v.elapsedMs)} ms; last seen: ${v.lastSeen ?? "nothing"}` };
  if (v.stateProblem) return { status: "no-effect", reason: v.stateProblem };
  const min = v.minPixels ?? 0.0005;
  if (v.pixels !== null && v.pixels < min) {
    return { status: "no-effect", reason: `the store changed but only ${(v.pixels * 100).toFixed(3)}% of the viewport's pixels did` };
  }
  if (v.elapsedMs > v.softMs) return { status: "slow", reason: `finished in ${Math.round(v.elapsedMs)} ms, over its ${v.softMs} ms budget` };
  return { status: "ok", reason: null };
}

// ---- subscribers ------------------------------------------------------------

/** What the in-page probe records per subscriber over a scenario (see probe.ts). */
export interface SubscriberRaw {
  id: number;
  calls: number;
  totalMs: number;
  maxMs: number;
  /** every call's time, in order */
  samples: number[];
}

export interface SubscriberCost {
  id: number;
  /** where the function is defined, `src/panels/loads.ts:render:30` */
  where: string;
  calls: number;
  medianMs: number;
  maxMs: number;
  totalMs: number;
}

/** Ranked by median cost of one event: what a panel costs every time anything changes. */
export function rankSubscribers(raw: readonly SubscriberRaw[], nameOf: (id: number) => string): SubscriberCost[] {
  return raw
    .filter((r) => r.calls > 0)
    .map((r) => ({ id: r.id, where: nameOf(r.id), calls: r.calls, medianMs: quantile(r.samples, 0.5), maxMs: r.maxMs, totalMs: r.totalMs }))
    .sort((a, b) => b.medianMs - a.medianMs || b.totalMs - a.totalMs);
}

// ---- the result, ranked -----------------------------------------------------

export interface ScenarioResult {
  name: string;
  what: string;
  status: Status;
  reason: string | null;
  /** wall time of the whole scenario */
  durationMs: number;
  /** one entry per repetition */
  repMs: number[];
  /** what the scenario proved changed */
  proof: { state: string | null; pixelsChangedPct: number | null };
  frames: FrameStats | null;
  longTasks: { count: number; totalMs: number; maxMs: number };
  renderer: ProfileSummary | null;
  main: ProfileSummary | null;
  subscribers: SubscriberCost[];
  /** events dispatched during the scenario, by type */
  events: Record<string, number>;
  /** the machine's busy percentage over the scenario */
  machineCpuPct: number;
  noisy: boolean;
  /** Chromium's own counters over the scenario, ms; heap in MB */
  metrics: Record<string, number>;
}

export interface RunResult {
  tool: "apegmsh-viewer-profile";
  version: 1;
  file: string;
  sizeBytes: number;
  date: string;
  commit: string;
  machine: { cpus: string; cores: number; gpu: string; viewport: number[]; pixelRatio: number };
  model: { cells: number; nodes: number; results: string | null; steps: number | null };
  samplingUs: number;
  warnings: string[];
  scenarios: ScenarioResult[];
}

export interface RankedRow {
  scenario: string;
  /** the median cost of one repetition */
  perOpMs: number;
  name: string;
  where: string;
  selfMs: number;
  inclusiveMs: number;
  process: "renderer" | "main";
}

/** Functions by self time over every scenario that really ran, the app's code before a library's by nothing but cost. */
export function rankFunctions(r: RunResult, topN = 20): RankedRow[] {
  const rows: RankedRow[] = [];
  for (const s of r.scenarios) {
    if (s.status === "skipped") continue;
    const perOp = quantile(s.repMs, 0.5);
    for (const [process, sum] of [["renderer", s.renderer], ["main", s.main]] as const) {
      for (const f of sum?.top ?? []) {
        rows.push({ scenario: s.name, perOpMs: perOp, name: f.name, where: `${f.file}:${f.line}`, selfMs: f.selfMs, inclusiveMs: f.inclusiveMs, process });
      }
    }
  }
  return rows.sort((a, b) => b.selfMs - a.selfMs).slice(0, topN);
}

const fmt = (x: number | null | undefined, digits = 1): string => (x === null || x === undefined || !Number.isFinite(x) ? "n/a" : x.toFixed(digits));

/** The human summary: scenarios by cost, then subscribers, then functions. Honest before pretty. */
export function renderSummary(r: RunResult): string {
  const out: string[] = [];
  const name = r.file.split(/[\\/]/).pop();
  out.push(`# Viewer profile: ${name}`, "");
  out.push(`${r.date}, commit ${r.commit}. ${r.model.cells} cells, ${r.model.nodes} nodes, ${(r.sizeBytes / 1048576).toFixed(1)} MB${r.model.results ? `, results ${r.model.results.split(/[\\/]/).pop()} (${r.model.steps ?? "?"} steps in the longest stage)` : ", no results"}.`);
  out.push(`${r.machine.cores} cores (${r.machine.cpus}); GPU "${r.machine.gpu}"; canvas ${r.machine.viewport.join("x")} @ ${r.machine.pixelRatio}x. Sampling ${r.samplingUs} us.`, "");
  const bad = r.scenarios.filter((s) => s.status === "failed" || s.status === "no-effect");
  out.push(bad.length ? `**${bad.length} scenario(s) did not prove anything: ${bad.map((s) => s.name).join(", ")}.**` : "Every scenario that ran changed what it claims to change.");
  const noisy = r.scenarios.filter((s) => s.noisy && s.status !== "skipped");
  if (noisy.length) out.push(`Noisy (machine above ${NOISY_LOAD_PCT}% busy): ${noisy.map((s) => `${s.name} ${fmt(s.machineCpuPct, 0)}%`).join(", ")}. Treat those times as upper bounds.`);
  for (const w of r.warnings) out.push(`Warning: ${w}`);
  out.push("", "## Scenarios, most expensive first", "");
  out.push("| scenario | status | per op (ms) | ops | total (ms) | fps median | p95 frame (ms) | long tasks | renderer busy (ms) | main busy (ms) | GC % | machine % |");
  out.push("|---|---|---|---|---|---|---|---|---|---|---|---|");
  for (const s of [...r.scenarios].sort((a, b) => (quantile(b.repMs, 0.5) || 0) - (quantile(a.repMs, 0.5) || 0))) {
    out.push(`| ${s.name} | ${s.status} | ${fmt(quantile(s.repMs, 0.5))} | ${s.repMs.length} | ${fmt(s.durationMs, 0)} | ${fmt(s.frames?.medianFps)} | ${fmt(s.frames?.p95FrameMs)} | ${s.longTasks.count} (${fmt(s.longTasks.totalMs, 0)} ms) | ${fmt(s.renderer?.busyMs, 0)} | ${fmt(s.main?.busyMs, 0)} | ${fmt(s.renderer?.gcPct)} | ${fmt(s.machineCpuPct, 0)} |`);
  }
  const notes = r.scenarios.filter((s) => s.reason);
  if (notes.length) {
    out.push("");
    for (const s of notes) out.push(`- **${s.name}** (${s.status}): ${s.reason}`);
  }
  const events = r.scenarios.filter((s) => s.name.startsWith("store-event") && s.status !== "skipped");
  out.push("", "## One store event, per subscriber", "");
  out.push("A subscriber runs on every store change, so its median here is paid on every step, hover and click.", "");
  for (const s of events) {
    const total = s.subscribers.reduce((a, c) => a + c.medianMs, 0);
    out.push(`### ${s.name}: ${fmt(total, 2)} ms over ${s.subscribers.length} subscribers, ${s.repMs.length} events`, "");
    out.push("| subscriber | median (ms) | max (ms) | share |", "|---|---|---|---|");
    for (const c of s.subscribers.slice(0, 10)) out.push(`| ${c.where} | ${fmt(c.medianMs, 3)} | ${fmt(c.maxMs, 2)} | ${total > 0 ? fmt((100 * c.medianMs) / total, 0) : "n/a"}% |`);
    out.push("");
  }
  if (!events.length) out.push("No store-event scenario ran.", "");
  out.push("## Functions by self time, every scenario", "");
  out.push("| function | where | process | scenario | self (ms) | inclusive (ms) |", "|---|---|---|---|---|---|");
  for (const f of rankFunctions(r)) out.push(`| ${f.name} | ${f.where} | ${f.process} | ${f.scenario} | ${fmt(f.selfMs)} | ${fmt(f.inclusiveMs)} |`);
  out.push("");
  return out.join("\n");
}

/** Exit code of a run: a scenario that failed or proved nothing fails the run; slow and skipped are reported, and pass. */
export function exitCodeOf(r: RunResult): number {
  return r.scenarios.some((s) => s.status === "failed" || s.status === "no-effect") ? 1 : 0;
}
