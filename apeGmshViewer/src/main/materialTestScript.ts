// ADR 0135 D3: what the solver is told, and what is believed of its answer. Pure:
// no Electron, no process, so every rule is tested without a solver.
//
// A job comes from the page and its values from a model file, and a model file
// can come from anyone. The script is Tcl, which can start programs, so nothing
// is quoted or escaped: a value is either a finite number, written by this code,
// or a word of letters, digits and underscores. Anything else refuses the run.
//
// The fork reports through `puts`, which reaches stderr, and it exits 0 even when
// the script stopped on an error, so a run counts only when its framed lines are
// all there: the build, one row per step in order, and the closing count.

import type { MaterialTestJob, MaterialTestLine, MaterialTestNd, MaterialTestProtocol } from "../state/types/materialTest.ts";

export const MAX_STEPS = 200_000;
/** Stress control (S3): the most target steps, and the trials each makes (the last commits). */
export const MAX_STRESS_STEPS = 5_000;
export const STRESS_TRIALS = 60;
/** nD (S5): the most steps of a run (each row carries the n-by-n tangent), and the steps that bring it to its confinement. */
export const MAX_ND_STEPS = 20_000;
export const CONFINE_STEPS = 10;
/** The relative stress tolerance of the fork's mixed control for nD runs: a softening material's own return mapping settles at about 1e-10 of its stress, the fork's default. */
export const ND_TOL = 1e-8;
export const ND_MAX_ITER = 50;

/**
 * Each nD loading mode as the fork's `ndTest mixed` sees it (WP-248): one letter per
 * component, e (strain given) or s (stress given), the driven component, and whether
 * the held normal stresses are -p. Components in OpenSees' order: 3D 11 22 33 12 23 31,
 * plane stress 11 22 12, shear as engineering strain.
 */
export const ND_MODES: Readonly<Record<MaterialTestNd["view"], Partial<Record<MaterialTestNd["mode"], { kinds: string; drive: number; confined: boolean }>>>> = {
  ThreeDimensional: {
    uniaxialStress: { kinds: "esssss", drive: 0, confined: false },
    triaxial: { kinds: "esssss", drive: 0, confined: true },
    oedometer: { kinds: "eeeeee", drive: 0, confined: false },
    simpleShear: { kinds: "sssess", drive: 3, confined: true },
  },
  PlaneStress: {
    uniaxialStress: { kinds: "ess", drive: 0, confined: false },
    oedometer: { kinds: "eee", drive: 0, confined: false },
    simpleShear: { kinds: "sse", drive: 2, confined: true },
  },
};
const NORMALS: Readonly<Record<MaterialTestNd["view"], number>> = { ThreeDimensional: 3, PlaneStress: 2 };

/** How an nD job loads its material, checked: the mode's letters, the driven component, the held values and the confinement targets (none when p is 0). */
export function ndLoadOf(raw: unknown): { view: MaterialTestNd["view"]; kinds: string; drive: number; held: number[]; confine: number[][] } {
  if (!isRecord(raw)) return refuse("the nD loading is not an object");
  const view = raw["view"];
  if (view !== "ThreeDimensional" && view !== "PlaneStress") return refuse("the nD view must be ThreeDimensional or PlaneStress");
  const mode = ND_MODES[view][raw["mode"] as MaterialTestNd["mode"]];
  if (!mode) return refuse(`the ${view} view has no ${String(raw["mode"])} loading`);
  const p = raw["confinement"];
  if (!finite(p)) return refuse("the confinement must be a finite number");
  if (!mode.confined && p !== 0) return refuse(`${String(raw["mode"])} holds no confinement: it must be 0`);
  const normal = (i: number): boolean => i < NORMALS[view];
  const held = [...mode.kinds].map((k, i) => (i === mode.drive || k === "e" || !normal(i) ? 0 : 0 - p));
  const goal = [...mode.kinds].map((_, i) => (normal(i) ? 0 - p : 0));
  const confine = p === 0 ? [] : Array.from({ length: CONFINE_STEPS }, (_, j) => goal.map((g) => (g * (j + 1)) / CONFINE_STEPS));
  return { view, kinds: mode.kinds, drive: mode.drive, held, confine };
}

/**
 * The tangent of the driven component along the mode's path: the fork's n-by-n tangent
 * condensed onto it, K_dd - K_ds K_ss^-1 K_sd over the stress-held set s (the strain-held
 * components do not move). Linear algebra on the fork's numbers, no material law (D1).
 * Null when K_ss is singular.
 */
export function condensedTangent(k: readonly number[], kinds: string, drive: number): number | null {
  const n = kinds.length;
  const s = [...kinds].flatMap((c, i) => (c === "s" ? [i] : []));
  const m = s.length;
  // [K_ss | K_sd], reduced by Gauss-Jordan with partial pivoting
  const a = s.map((i) => [...s.map((j) => k[i * n + j]!), k[i * n + drive]!]);
  for (let c = 0; c < m; c++) {
    let piv = c;
    for (let r = c + 1; r < m; r++) if (Math.abs(a[r]![c]!) > Math.abs(a[piv]![c]!)) piv = r;
    if (!(Math.abs(a[piv]![c]!) > 0)) return null;
    [a[c], a[piv]] = [a[piv]!, a[c]!];
    for (let r = 0; r < m; r++) {
      if (r === c) continue;
      const f = a[r]![c]! / a[c]![c]!;
      for (let j = c; j <= m; j++) a[r]![j] = a[r]![j]! - f * a[c]![j]!;
    }
  }
  let t = k[drive * n + drive]!;
  for (let c = 0; c < m; c++) t -= k[drive * n + s[c]!]! * (a[c]![m]! / a[c]![c]!);
  return t;
}

const PREFIX = "@@MT ";
const TYPE = /^[A-Za-z][A-Za-z0-9_]*$/;
const WORD = /^-?[A-Za-z][A-Za-z0-9_]*$/;
const NUMBER = /^[-+]?\d+(\.\d+)?([eE][-+]?\d+)?$/;
/** Removed before anything else runs: behind the token rules, not instead of them. */
const REMOVED = ["exec", "open", "socket", "source", "cd", "file", "glob", "load", "package", "interp"];

export class MaterialTestRefused extends Error {}

const refuse = (why: string): never => {
  throw new MaterialTestRefused(why);
};

const isRecord = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null && !Array.isArray(v);
const finite = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);

function count(v: unknown, what: string): number {
  if (typeof v !== "number" || !Number.isSafeInteger(v) || v < 1 || v > MAX_STEPS) refuse(`${what} must be a whole number from 1 to ${MAX_STEPS}`);
  return v as number;
}

function numbers(v: unknown, what: string): number[] {
  if (!Array.isArray(v) || v.length === 0 || v.length > MAX_STEPS || !v.every(finite)) refuse(`${what} must be a list of finite numbers, 1 to ${MAX_STEPS} long`);
  return v as number[];
}

function ramp(out: number[], to: number, steps: number): void {
  const from = out[out.length - 1]!;
  for (let i = 1; i <= steps; i++) out.push(from + ((to - from) * i) / steps);
  if (out.length > MAX_STEPS + 1) refuse(`the protocol has more than ${MAX_STEPS} steps`);
}

/** The strain of every step, starting from zero. */
export function strainPath(raw: MaterialTestProtocol): number[] {
  const p: unknown = raw;
  if (!isRecord(p)) return refuse("the protocol is not an object");
  const out = [0];
  if (p["kind"] === "monotonic") {
    if (!finite(p["peak"])) refuse("the peak strain must be a finite number");
    ramp(out, p["peak"] as number, count(p["steps"], "steps"));
  } else if (p["kind"] === "cyclic") {
    const n = count(p["stepsPerRamp"], "stepsPerRamp");
    for (const a of numbers(p["amplitudes"], "amplitudes")) {
      ramp(out, a, n);
      ramp(out, -a, 2 * n);
      ramp(out, 0, n);
    }
  } else if (p["kind"] === "values") {
    out.push(...numbers(p["values"], "values"));
  } else {
    refuse("the protocol kind must be monotonic, cyclic or values");
  }
  return out;
}

/** The target stress of every step of a stress protocol, after the starting zero (D6, S3). */
export function stressPath(raw: MaterialTestProtocol): number[] {
  const p: unknown = raw;
  if (!isRecord(p) || p["kind"] !== "stress") return refuse("the protocol is not stress control");
  const n = count(p["stepsPerRamp"], "stepsPerRamp");
  const out = [0];
  for (const t of numbers(p["targets"], "targets")) {
    ramp(out, t, n);
    if (out.length > MAX_STRESS_STEPS + 1) refuse(`a stress protocol has at most ${MAX_STRESS_STEPS} steps`);
  }
  return out.slice(1);
}

/**
 * The stress-control driver (ADR 0135 D6, S3): fixed text, never built from the file.
 * Each target step makes exactly STRESS_TRIALS calls of strainUniaxialTest under
 * `model testUniaxial STRESS_TRIALS`, so the last one, at the chosen strain, commits
 * (TclUniaxialMaterialTester.cpp: the N-th call commits, nothing reverts). The search
 * marches from the committed strain toward the target with bounded growth; when its
 * progress turns back, a peak lies inside the last two steps, and a golden-section
 * search decides whether the peak reaches the target. The first crossing is refined by
 * Illinois false position. A target beyond the peak stops the run, with the most the
 * material reached.
 */
const STRESS_DRIVER = String.raw`proc mtTry {e} {
  global mtUsed mtD mtTarget
  incr mtUsed
  strainUniaxialTest $e
  return [expr {$mtD * ([stressUniaxialTest] - $mtTarget)}]
}
proc mtStep {target} {
  global mtUsed mtD mtTarget mtE mtS mtE0 mtN
  set mtTarget $target
  set mtUsed 0
  set e0 $mtE
  set s0 $mtS
  set mtD [expr {$target >= $s0 ? 1.0 : -1.0}]
  set tol [expr {1e-9 * max(abs($target), abs($s0), 1e-12)}]
  set budget [expr {$mtN - 1}]
  set f0 [expr {$mtD * ($s0 - $target)}]
  set best $e0
  set bestF $f0
  set lo ""
  if {abs($f0) > $tol} {
    set h [expr {0.5 * abs($target - $s0) / abs($mtE0)}]
    set ep $e0 ; set fp $f0
    set eq $e0 ; set fq $f0
    while {$mtUsed < $budget} {
      set e [expr {$ep + $mtD * $h}]
      set f [mtTry $e]
      if {abs($f) < abs($bestF)} { set best $e ; set bestF $f }
      if {$f >= -$tol} { set lo $ep ; set flo $fp ; set hi $e ; set fhi $f ; break }
      if {$f < $fp} {
        set a $eq ; set b $e
        set x1 $ep ; set f1 $fp
        set x2 $x1 ; set f2 $f1
        set gr 0.381966011250105
        while {$mtUsed < $budget - 2 && abs($b - $a) > 1e-12 * max(abs($a), abs($b), 1e-30)} {
          if {abs($b - $x1) > abs($x1 - $a)} { set x2 [expr {$x1 + $gr * ($b - $x1)}] } else { set x2 [expr {$x1 - $gr * ($x1 - $a)}] }
          set f2 [mtTry $x2]
          if {abs($f2) < abs($bestF)} { set best $x2 ; set bestF $f2 }
          if {$f2 >= -$tol} { break }
          if {$f2 > $f1} {
            if {($x2 - $x1) * $mtD > 0} { set a $x1 } else { set b $x1 }
            set x1 $x2 ; set f1 $f2
          } else {
            if {($x2 - $x1) * $mtD > 0} { set b $x2 } else { set a $x2 }
          }
        }
        if {$f2 >= -$tol} { set lo $eq ; set flo $fq ; set hi $x2 ; set fhi $f2 ; break }
        return [list stop "the target $target is beyond the peak: the most the material reaches on this path is [expr {$mtD * $f1 + $target}]"]
      }
      set eq $ep ; set fq $fp
      set ep $e ; set fp $f
      set h [expr {$h * 1.6}]
    }
    if {$lo eq ""} { return [list stop "the target $target was not reached in $budget trials"] }
    set side 0
    while {$mtUsed < $budget && abs($bestF) > $tol} {
      set e [expr {$hi - $fhi * ($hi - $lo) / ($fhi - $flo)}]
      if {!(($e - $lo) * ($e - $hi) < 0)} { set e [expr {0.5 * ($lo + $hi)}] }
      set f [mtTry $e]
      if {abs($f) < abs($bestF)} { set best $e ; set bestF $f }
      if {$f >= 0} {
        set hi $e ; set fhi $f
        if {$side == 1} { set flo [expr {$flo / 2.0}] } ; set side 1
      } else {
        set lo $e ; set flo $f
        if {$side == -1} { set fhi [expr {$fhi / 2.0}] } ; set side -1
      }
    }
    if {abs($bestF) > 1e-6 * max(abs($target), abs($s0), 1e-12)} { return [list stop "the target $target was not reached closely: off by [expr {abs($bestF)}]"] }
  }
  while {$mtUsed < $budget} { strainUniaxialTest $best ; incr mtUsed }
  strainUniaxialTest $best
  set mtE $best
  set mtS [stressUniaxialTest]
  return [list ok $best]
}`;

/** A number as Tcl and OpenSees read it: integers stay integers, so tag slots parse. */
const num = (n: number): string => String(n === 0 ? 0 : n);

function lineOf(raw: unknown): { text: string; tag: number; family: string } {
  if (!isRecord(raw)) return refuse("a material is not an object");
  const m = raw as unknown as MaterialTestLine;
  if (!Number.isSafeInteger(m.tag) || m.tag < 0) refuse("a material tag must be a whole number");
  const family = m.family ?? "uniaxialMaterial";
  if (family !== "uniaxialMaterial" && family !== "nDMaterial") refuse(`material tag ${m.tag}: its family must be uniaxialMaterial or nDMaterial`);
  const who = `${family === "nDMaterial" ? "nD material" : "material"} tag ${m.tag}`;
  if (typeof m.type !== "string" || !TYPE.test(m.type)) refuse(`${who}: its type is not a plain name`);
  if (!Array.isArray(m.args)) refuse(`${who}: its arguments are not a list`);
  const args = m.args.map((a, i) => {
    if (finite(a)) return num(a);
    if (typeof a === "string" && WORD.test(a)) return a;
    return refuse(`${who} (${m.type}): argument ${i} is neither a finite number nor a plain word`);
  });
  return { text: [family, m.type, m.tag, ...args].join(" "), tag: m.tag, family };
}

/**
 * The Tcl script of a job, the strains it imposes (null under stress control, where the
 * fork finds them), and the rows it must print. Throws `MaterialTestRefused`.
 */
export function scriptOf(raw: unknown): { script: string; strain: number[] | null; rows: number; nd: { kinds: string; drive: number } | null } {
  if (!isRecord(raw)) refuse("the job is not an object");
  const job = raw as unknown as MaterialTestJob;
  if (!Array.isArray(job.materials) || job.materials.length === 0) refuse("the job names no material");
  const lines = job.materials.map(lineOf);
  const tags = new Set(lines.map((l) => `${l.family} ${l.tag}`));
  if (tags.size !== lines.length) refuse("two materials of the job share a tag");
  const last = lines[lines.length - 1]!;
  if (last.tag !== job.target) refuse("the material under test must be the job's last line");
  const nd = isRecord(job.protocol) ? (job.protocol as { nd?: unknown }).nd : undefined;
  if ((last.family === "nDMaterial") !== (nd !== undefined)) refuse(last.family === "nDMaterial" ? "an nD material needs a view and a loading mode" : "a uniaxial material takes no nD loading");
  const head = [
    `if {[info commands ladrunoBuild] eq ""} {puts "${PREFIX}nofork"; exit}`,
    "set mtBuild [ladrunoBuild]",
    `foreach mtCommand {${REMOVED.join(" ")}} {rename $mtCommand {}}`,
    `puts "${PREFIX}build $mtBuild"`,
  ];
  if (nd !== undefined) {
    if ((job.protocol as { kind?: unknown }).kind === "stress") refuse("stress control runs uniaxial materials");
    const load = ndLoadOf(nd);
    const strain = strainPath(job.protocol);
    if (strain.length > MAX_ND_STEPS + 1) refuse(`an nD run has at most ${MAX_ND_STEPS} steps`);
    const all = "s".repeat(load.kinds.length);
    const script = [
      ...head,
      // a fork build older than WP-248 has no nD tester: say so, not a Tcl trace
      `if {[catch {model testNDMaterial}]} {puts "${PREFIX}nond"; exit}`,
      ...lines.map((l) => l.text),
      `ndTest material ${job.target} ${load.view}`,
      // to the confinement first, all stresses held, committed and not printed
      ...load.confine.map((c) => `ndTest mixed ${all} ${c.map(num).join(" ")} -tol ${ND_TOL} -maxIter ${ND_MAX_ITER}; ndTest commit`),
      `set mtV {${load.held.map(num).join(" ")}}`,
      `set mtE0 [lindex [ndTest getStrain] ${load.drive}]`,
      "set mtStep 0",
      `foreach mtStrain {${strain.map(num).join(" ")}} {`,
      `  lset mtV ${load.drive} [expr {$mtE0 + $mtStrain}]`,
      `  if {[catch {ndTest mixed ${load.kinds} {*}$mtV -tol ${ND_TOL} -maxIter ${ND_MAX_ITER}} mtErr]} {puts "${PREFIX}stop $mtStep $mtErr"; break}`,
      "  ndTest commit",
      `  puts "${PREFIX}nrow $mtStep [lindex [ndTest getStrain] ${load.drive}] [lindex [ndTest getStress] ${load.drive}] [ndTest getTangent]"`,
      "  incr mtStep",
      "}",
      `puts "${PREFIX}end $mtStep"`,
      "",
    ].join("\n");
    return { script, strain: null, rows: strain.length, nd: { kinds: load.kinds, drive: load.drive } };
  }
  if (isRecord(job.protocol) && (job.protocol as { kind?: unknown }).kind === "stress") {
    const targets = stressPath(job.protocol);
    const script = [
      ...head,
      `model testUniaxial ${STRESS_TRIALS}`,
      ...lines.map((l) => l.text),
      `uniaxialTest ${job.target}`,
      STRESS_DRIVER,
      `set mtN ${STRESS_TRIALS}`,
      "set mtE 0.0",
      "set mtS 0.0",
      "set mtE0 [tangUniaxialTest]",
      // row 0, the starting zero, needs no call: a call would count towards the commit
      `puts "${PREFIX}srow 0 0 [stressUniaxialTest] $mtE0"`,
      "set mtStep 1",
      `foreach mtTarget0 {${targets.map(num).join(" ")}} {`,
      "  set mtR [mtStep $mtTarget0]",
      `  if {[lindex $mtR 0] eq "stop"} {puts "${PREFIX}stop $mtStep [lindex $mtR 1]"; break}`,
      `  puts "${PREFIX}srow $mtStep $mtE $mtS [tangUniaxialTest]"`,
      "  incr mtStep",
      "}",
      `puts "${PREFIX}end $mtStep"`,
      "",
    ].join("\n");
    return { script, strain: null, rows: targets.length + 1, nd: null };
  }
  const strain = strainPath(job.protocol);
  const script = [
    ...head,
    "model testUniaxial",
    ...lines.map((l) => l.text),
    `uniaxialTest ${job.target}`,
    "set mtStep 0",
    `foreach mtStrain {${strain.map(num).join(" ")}} {`,
    "  strainUniaxialTest $mtStrain",
    `  puts "${PREFIX}row $mtStep [stressUniaxialTest] [tangUniaxialTest]"`,
    "  incr mtStep",
    "}",
    `puts "${PREFIX}end $mtStep"`,
    "",
  ].join("\n");
  return { script, strain, rows: strain.length, nd: null };
}

export type ParsedRun =
  | { ok: true; build: string; stress: Float64Array; tangent: Float64Array }
  | { ok: false; error: string; build: string | null };

/** A stress-control run: the strain each step found, and why it stopped early (null when every step was reached). */
export type ParsedStressRun =
  | { ok: true; build: string; strain: Float64Array; stress: Float64Array; tangent: Float64Array; stopped: string | null }
  | { ok: false; error: string; build: string | null };

/** The solver's last words: what it printed after the build line, or else its last lines. Tcl's
 *  `(file "<temp dir>\test.tcl" line N)` becomes `(test script line N)`: the run's temporary
 *  folder is random, so it would make every message, and every still of one, unique. */
function messageOf(lines: string[]): string {
  const after = lines.findIndex((l) => l.startsWith(`${PREFIX}build `));
  const own = lines.slice(after + 1).filter((l) => l.trim() !== "" && !l.startsWith(PREFIX));
  return own.slice(-6).join(" ").replace(/\(file "[^"]*" line (\d+)\)/g, "(test script line $1)").slice(-600).trim();
}

/** What a fork build without `model testNDMaterial` (before WP-248) is told. */
export const NO_ND_TESTER = "this fork build has no nD material tester (model testNDMaterial, WP-248): install ladruno-v20261123 or later";

/** Read stderr against the `steps` rows the script must have produced. */
export function parseRun(stderr: string, steps: number): ParsedRun {
  const lines = stderr.split(/\r?\n/);
  const fail = (what: string, build: string | null): ParsedRun => {
    const said = messageOf(lines);
    return { ok: false, build, error: said ? `${what}. The solver said: ${said}` : `${what}.` };
  };
  const framed = lines.filter((l) => l.startsWith(PREFIX)).map((l) => l.slice(PREFIX.length).trim().split(/\s+/));
  if (framed.some((f) => f[0] === "nofork")) return { ok: false, build: null, error: "this OpenSees executable is not a Ladruno fork build (it has no ladrunoBuild command)" };
  const head = framed[0];
  if (!head || head[0] !== "build" || head.length !== 2) return fail("the solver did not start the script", null);
  const build = head[1]!;
  const stress = new Float64Array(steps);
  const tangent = new Float64Array(steps);
  let row = 0;
  for (const f of framed.slice(1)) {
    if (f[0] !== "row") break;
    const [, i, s, t] = f;
    if (f.length !== 4 || i !== String(row) || row >= steps) return fail(`the solver's row ${row} is malformed`, build);
    if (!NUMBER.test(s!) || !NUMBER.test(t!)) return fail(`at step ${row} the material returned stress ${s} and tangent ${t}`, build);
    stress[row] = Number(s);
    tangent[row] = Number(t);
    row++;
  }
  const last = framed[row + 1];
  if (!last || last[0] !== "end" || last[1] !== String(steps) || row !== steps || framed.length !== steps + 2) {
    return fail(`the run stopped after ${row} of ${steps} steps`, build);
  }
  return { ok: true, build, stress, tangent };
}

/**
 * Read a stress-control run (S3) against the `rows` its script may print: row 0 and
 * one per target step, each with the strain the driver found. A `stop` line after
 * the rows it reached ends the run early with the driver's reason; the rows before
 * it are the curve. Anything else out of frame fails, as for a strain run.
 */
export function parseStressRun(stderr: string, rows: number): ParsedStressRun {
  const lines = stderr.split(/\r?\n/);
  const fail = (what: string, build: string | null): ParsedStressRun => {
    const said = messageOf(lines);
    return { ok: false, build, error: said ? `${what}. The solver said: ${said}` : `${what}.` };
  };
  const framed = lines.filter((l) => l.startsWith(PREFIX)).map((l) => l.slice(PREFIX.length).trim().split(/\s+/));
  if (framed.some((f) => f[0] === "nofork")) return { ok: false, build: null, error: "this OpenSees executable is not a Ladruno fork build (it has no ladrunoBuild command)" };
  const head = framed[0];
  if (!head || head[0] !== "build" || head.length !== 2) return fail("the solver did not start the script", null);
  const build = head[1]!;
  const strain: number[] = [], stress: number[] = [], tangent: number[] = [];
  let k = 1;
  for (; k < framed.length && framed[k]![0] === "srow"; k++) {
    const f = framed[k]!;
    const row = strain.length;
    if (f.length !== 5 || f[1] !== String(row) || row >= rows) return fail(`the solver's row ${row} is malformed`, build);
    const [, , e, s, t] = f;
    if (!NUMBER.test(e!) || !NUMBER.test(s!) || !NUMBER.test(t!)) return fail(`at step ${row} the material returned strain ${e}, stress ${s} and tangent ${t}`, build);
    strain.push(Number(e));
    stress.push(Number(s));
    tangent.push(Number(t));
  }
  let stopped: string | null = null;
  const next = framed[k];
  if (next && next[0] === "stop" && next[1] === String(strain.length) && strain.length > 0) {
    stopped = `step ${next[1]}: ${next.slice(2).join(" ")}`;
    k++;
  }
  const last = framed[k];
  if (!last || last[0] !== "end" || last[1] !== String(strain.length) || k !== framed.length - 1 || strain.length === 0) {
    return fail(`the run stopped after ${Math.max(0, strain.length - 1)} of ${rows - 1} steps`, build);
  }
  if (stopped === null && strain.length !== rows) return fail(`the run stopped after ${strain.length - 1} of ${rows - 1} steps`, build);
  return { ok: true, build, strain: Float64Array.from(strain), stress: Float64Array.from(stress), tangent: Float64Array.from(tangent), stopped };
}

/**
 * Read an nD run (S5) against the `rows` its script may print: one `nrow` per step with
 * the driven component's strain and stress and the fork's n-by-n tangent, condensed here
 * onto the driven component. A `stop` after the rows it reached (the fork's mixed control
 * refused a step) ends the run early with the fork's reason, as stress control does.
 */
export function parseNdRun(stderr: string, rows: number, kinds: string, drive: number): ParsedStressRun {
  const n = kinds.length;
  const lines = stderr.split(/\r?\n/);
  const fail = (what: string, build: string | null): ParsedStressRun => {
    const said = messageOf(lines);
    return { ok: false, build, error: said ? `${what}. The solver said: ${said}` : `${what}.` };
  };
  const framed = lines.filter((l) => l.startsWith(PREFIX)).map((l) => l.slice(PREFIX.length).trim().split(/\s+/));
  if (framed.some((f) => f[0] === "nofork")) return { ok: false, build: null, error: "this OpenSees executable is not a Ladruno fork build (it has no ladrunoBuild command)" };
  const head = framed[0];
  if (!head || head[0] !== "build" || head.length !== 2) return fail("the solver did not start the script", null);
  const build = head[1]!;
  if (framed.some((f) => f[0] === "nond")) return { ok: false, build, error: NO_ND_TESTER };
  const strain: number[] = [], stress: number[] = [], tangent: number[] = [];
  let k = 1;
  for (; k < framed.length && framed[k]![0] === "nrow"; k++) {
    const f = framed[k]!;
    const row = strain.length;
    if (f.length !== 4 + n * n || f[1] !== String(row) || row >= rows) return fail(`the solver's row ${row} is malformed`, build);
    const values = f.slice(2);
    const bad = values.find((v) => !NUMBER.test(v));
    if (bad !== undefined) return fail(`at step ${row} the material returned ${bad}`, build);
    const t = condensedTangent(values.slice(2).map(Number), kinds, drive);
    if (t === null) return fail(`at step ${row} the tangent of the held components is singular`, build);
    strain.push(Number(values[0]));
    stress.push(Number(values[1]));
    tangent.push(t);
  }
  let stopped: string | null = null;
  const next = framed[k];
  if (next && next[0] === "stop" && next[1] === String(strain.length)) {
    if (strain.length === 0) return fail(`the first step was refused: ${next.slice(2).join(" ")}`, build);
    stopped = `step ${next[1]}: ${next.slice(2).join(" ")}`;
    k++;
  }
  const last = framed[k];
  if (!last || last[0] !== "end" || last[1] !== String(strain.length) || k !== framed.length - 1 || strain.length === 0) {
    return fail(`the run stopped after ${Math.max(0, strain.length - 1)} of ${rows - 1} steps`, build);
  }
  if (stopped === null && strain.length !== rows) return fail(`the run stopped after ${strain.length - 1} of ${rows - 1} steps`, build);
  return { ok: true, build, strain: Float64Array.from(strain), stress: Float64Array.from(stress), tangent: Float64Array.from(tangent), stopped };
}
