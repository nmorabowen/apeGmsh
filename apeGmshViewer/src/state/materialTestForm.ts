// The material tester's form and curve, as the materials window shows them (ADR
// 0135 D6 and Amendment 1). Pure, and a leaf: selectors.ts re-exports it for the
// panel, and the effect thins a finished run's curve with `materialTestPointsOf`.
//
// The form keeps what was typed (strings); `materialTestProtocolOf` turns it into
// a protocol or says which field is wrong. Main checks the protocol again
// (main/materialTestScript.ts), including the step limit, so nothing here guards
// the solver.

import type { DeclPath, MaterialTestForm, MaterialTestOverride, MaterialTestPoints, MaterialTestProtocol, MaterialTestSlot } from "./types.ts";

/** The most points a drawn curve keeps, besides its reversals. */
export const MAX_DRAWN = 4000;

/** A new form for `decl`: a cyclic protocol in strain-like units, to be edited. */
export const materialTestFormFor = (decl: DeclPath): MaterialTestForm => ({
  decl,
  kind: "cyclic",
  peak: "0.01",
  steps: "200",
  amplitudes: "0.002, 0.005, 0.01",
  stepsPerRamp: "50",
  values: "",
  targets: "300, -300, 400",
  second: "history",
  overrides: {},
});

/** ADR 0135 D6: the axes of a material's curve. Values are in the model's units. */
const AXES: Readonly<Record<string, readonly [string, string]>> = {
  LadrunoBondSlip: ["slip", "bond stress"],
  LadrunoCohesiveHinge: ["rotation jump", "moment"],
};

export const materialTestAxesOf = (type: string): readonly [string, string] => AXES[type] ?? ["strain", "stress"];

type Parsed<T> = { ok: T } | { error: string };

function number(text: string, what: string): Parsed<number> {
  const t = text.trim();
  const v = Number(t);
  return t !== "" && Number.isFinite(v) ? { ok: v } : { error: `${what} must be a number` };
}

function count(text: string, what: string): Parsed<number> {
  const t = text.trim();
  const v = Number(t);
  return t !== "" && Number.isSafeInteger(v) && v >= 1 ? { ok: v } : { error: `${what} must be a whole number of at least 1` };
}

function numbers(text: string, what: string): Parsed<number[]> {
  const words = text.split(/[\s,;]+/).filter((w) => w !== "");
  if (words.length === 0) return { error: `${what}: give at least one number` };
  const out: number[] = [];
  for (const w of words) {
    const v = Number(w);
    if (!Number.isFinite(v)) return { error: `${what}: "${w}" is not a number` };
    out.push(v);
  }
  return { ok: out };
}

/** The protocol the form describes, or the first thing wrong with it, in the words of the form. */
export function materialTestProtocolOf(f: MaterialTestForm, axis = "strain"): { protocol: MaterialTestProtocol } | { error: string } {
  if (f.kind === "monotonic") {
    const peak = number(f.peak, `the peak ${axis}`);
    if ("error" in peak) return peak;
    const steps = count(f.steps, "steps");
    if ("error" in steps) return steps;
    return { protocol: { kind: "monotonic", peak: peak.ok, steps: steps.ok } };
  }
  if (f.kind === "cyclic") {
    const amplitudes = numbers(f.amplitudes, "amplitudes");
    if ("error" in amplitudes) return amplitudes;
    const n = count(f.stepsPerRamp, "steps per ramp");
    if ("error" in n) return n;
    return { protocol: { kind: "cyclic", amplitudes: amplitudes.ok, stepsPerRamp: n.ok } };
  }
  if (f.kind === "stress") {
    const targets = numbers(f.targets, "targets");
    if ("error" in targets) return targets;
    const n = count(f.stepsPerRamp, "steps per ramp");
    if ("error" in n) return n;
    return { protocol: { kind: "stress", targets: targets.ok, stepsPerRamp: n.ok } };
  }
  const values = numbers(f.values, `${axis} values`);
  if ("error" in values) return values;
  return { protocol: { kind: "values", values: values.ok } };
}

/** How many steps a protocol imposes after the starting zero (main/materialTestScript.ts `strainPath`). */
export function materialTestStepsOf(p: MaterialTestProtocol): number {
  if (p.kind === "monotonic") return p.steps;
  if (p.kind === "cyclic") return p.amplitudes.length * 4 * p.stepsPerRamp;
  if (p.kind === "stress") return p.targets.length * p.stepsPerRamp;
  return p.values.length;
}

/**
 * The work done on the material: the running trapezoidal integral of stress over
 * strain, from zero (ADR 0135 D7: computed in the app, and labelled so). Over a
 * path that ends where the stress is zero it is the dissipated energy; elsewhere
 * it also holds the stored part.
 */
export function materialTestWorkOf(strain: ArrayLike<number>, stress: ArrayLike<number>): Float64Array {
  const n = Math.min(strain.length, stress.length);
  const w = new Float64Array(n);
  for (let i = 1; i < n; i++) w[i] = w[i - 1]! + 0.5 * (stress[i]! + stress[i - 1]!) * (strain[i]! - strain[i - 1]!);
  return w;
}

/**
 * A run thinned for drawing: every `stride`-th step, the last, and every reversal
 * of strain (so a hysteresis loop keeps its corners). When the reversals alone
 * would pass twice the cap, the stride alone decides. The work is integrated over
 * the full run first, so thinning does not change it.
 */
export function materialTestPointsOf(strain: ArrayLike<number>, stress: ArrayLike<number>, tangent: ArrayLike<number>, max: number = MAX_DRAWN): MaterialTestPoints {
  const n = Math.min(strain.length, stress.length, tangent.length);
  const work = materialTestWorkOf(strain, stress);
  const take = (idx: number[]): MaterialTestPoints => ({
    step: idx,
    strain: idx.map((i) => strain[i]!),
    stress: idx.map((i) => stress[i]!),
    tangent: idx.map((i) => tangent[i]!),
    energy: idx.map((i) => work[i]!),
  });
  if (n <= max) return take(Array.from({ length: n }, (_, i) => i));
  const stride = Math.ceil(n / max);
  const strided: number[] = [];
  const kept: number[] = [];
  for (let i = 0; i < n; i++) {
    const last = i === n - 1;
    const onStride = i % stride === 0;
    const reversal = i > 0 && !last && (strain[i]! - strain[i - 1]!) * (strain[i + 1]! - strain[i]!) < 0;
    if (onStride || last) strided.push(i);
    if (onStride || last || reversal) kept.push(i);
  }
  return take(kept.length > 2 * max ? strided : kept);
}

/** A protocol in one line, for a legend and the CSV's header. */
export function materialTestProtocolText(p: MaterialTestProtocol): string {
  if (p.kind === "monotonic") return `monotonic to ${p.peak} in ${p.steps} steps`;
  if (p.kind === "cyclic") return `cyclic ${p.amplitudes.join(", ")} at ${p.stepsPerRamp} steps per ramp`;
  if (p.kind === "stress") return `stress control to ${p.targets.join(", ")} at ${p.stepsPerRamp} steps per leg`;
  return `${p.values.length} pasted values`;
}

/** What a CSV header says of the run. */
export interface MaterialTestCsvHead {
  decl: DeclPath;
  type: string;
  build: string | null;
  protocol: MaterialTestProtocol;
}

/**
 * The full run as CSV (D7): `#` lines naming the material, the fork build and the
 * protocol, then one row per step. Numbers are written in full (shortest round trip).
 */
export function materialTestCsvOf(head: MaterialTestCsvHead, strain: ArrayLike<number>, stress: ArrayLike<number>, tangent: ArrayLike<number>): string {
  const [x, y] = materialTestAxesOf(head.type);
  const work = materialTestWorkOf(strain, stress);
  const lines = [
    `# ${head.decl} (${head.type})`,
    `# fork build ${head.build ?? "unknown"}; ${materialTestProtocolText(head.protocol)}; values in the model's units`,
    `# work: the trapezoidal integral of ${y} over ${x}, computed by apeGmshViewer`,
    `step,${x},${y},tangent,work`,
  ];
  for (let i = 0; i < work.length; i++) lines.push(`${i},${strain[i]},${stress[i]},${tangent[i]},${work[i]}`);
  return lines.join("\n") + "\n";
}

/** A file name to suggest for a run: the declaration's handle and the run id, in plain characters. */
export const materialTestCsvName = (decl: DeclPath, id: number): string => `${(decl.split("/").pop() || "material").replace(/[^A-Za-z0-9_-]/g, "_")}-run${id}.csv`;

/**
 * The overrides the form describes (D6), against the material's slots: a slot whose
 * field is blank, or reads its declared value, is not overridden. The first field
 * that is not a number, or names a locked slot, is the error.
 */
export function materialTestOverridesOf(f: MaterialTestForm, slots: readonly MaterialTestSlot[]): { overrides: MaterialTestOverride[] } | { error: string } {
  const out: MaterialTestOverride[] = [];
  for (const sl of slots) {
    const typed = (f.overrides[String(sl.slot)] ?? "").trim();
    if (typed === "") continue;
    const v = Number(typed);
    if (!Number.isFinite(v)) return { error: `${sl.label} must be a number` };
    if (v === sl.value) continue;
    if (sl.locked) return { error: `${sl.label} is locked: ${sl.locked}` };
    out.push({ slot: sl.slot, label: sl.label, declared: sl.value, value: v });
  }
  return { overrides: out };
}

/** The changed values as text, to copy back into the script by hand (D8: the script stays the source of truth). */
export function materialTestOverrideText(decl: DeclPath, type: string, overrides: readonly MaterialTestOverride[]): string {
  const handle = decl.split("/").pop() || decl;
  return `${handle} (${type}): ${overrides.map((o) => `${o.label} ${o.declared} -> ${o.value}`).join("; ")}`;
}
