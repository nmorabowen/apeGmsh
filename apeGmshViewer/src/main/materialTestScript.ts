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

import type { MaterialTestJob, MaterialTestLine, MaterialTestProtocol } from "../state/types/materialTest.ts";

export const MAX_STEPS = 200_000;

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

/** A number as Tcl and OpenSees read it: integers stay integers, so tag slots parse. */
const num = (n: number): string => String(n === 0 ? 0 : n);

function lineOf(raw: unknown): { text: string; tag: number } {
  if (!isRecord(raw)) return refuse("a material is not an object");
  const m = raw as unknown as MaterialTestLine;
  if (!Number.isSafeInteger(m.tag) || m.tag < 0) refuse("a material tag must be a whole number");
  const who = `material tag ${m.tag}`;
  if (typeof m.type !== "string" || !TYPE.test(m.type)) refuse(`${who}: its type is not a plain name`);
  if (!Array.isArray(m.args)) refuse(`${who}: its arguments are not a list`);
  const args = m.args.map((a, i) => {
    if (finite(a)) return num(a);
    if (typeof a === "string" && WORD.test(a)) return a;
    return refuse(`${who} (${m.type}): argument ${i} is neither a finite number nor a plain word`);
  });
  return { text: ["uniaxialMaterial", m.type, m.tag, ...args].join(" "), tag: m.tag };
}

/** The Tcl script of a job, and the strains it imposes. Throws `MaterialTestRefused`. */
export function scriptOf(raw: unknown): { script: string; strain: number[] } {
  if (!isRecord(raw)) refuse("the job is not an object");
  const job = raw as unknown as MaterialTestJob;
  if (!Array.isArray(job.materials) || job.materials.length === 0) refuse("the job names no material");
  const lines = job.materials.map(lineOf);
  const tags = new Set(lines.map((l) => l.tag));
  if (tags.size !== lines.length) refuse("two materials of the job share a tag");
  if (lines[lines.length - 1]!.tag !== job.target) refuse("the material under test must be the job's last line");
  const strain = strainPath(job.protocol);
  const script = [
    `if {[info commands ladrunoBuild] eq ""} {puts "${PREFIX}nofork"; exit}`,
    "set mtBuild [ladrunoBuild]",
    `foreach mtCommand {${REMOVED.join(" ")}} {rename $mtCommand {}}`,
    `puts "${PREFIX}build $mtBuild"`,
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
  return { script, strain };
}

export type ParsedRun = { ok: true; build: string; stress: Float64Array; tangent: Float64Array } | { ok: false; error: string; build: string | null };

/** The solver's last words: what it printed after the build line, or else its last lines. */
function messageOf(lines: string[]): string {
  const after = lines.findIndex((l) => l.startsWith(`${PREFIX}build `));
  const own = lines.slice(after + 1).filter((l) => l.trim() !== "" && !l.startsWith(PREFIX));
  return own.slice(-6).join(" ").slice(-600).trim();
}

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
