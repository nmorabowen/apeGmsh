// The material tester's form and curve, as the materials window shows them (ADR
// 0135 D6 and Amendment 1). Pure, and a leaf: selectors.ts re-exports it for the
// panel, and the effect thins a finished run's curve with `materialTestPointsOf`.
//
// The form keeps what was typed (strings); `materialTestProtocolOf` turns it into
// a protocol or says which field is wrong. Main checks the protocol again
// (main/materialTestScript.ts), including the step limit, so nothing here guards
// the solver.

import type { DeclPath, MaterialTestForm, MaterialTestPoints, MaterialTestProtocol } from "./types.ts";

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
  const values = numbers(f.values, `${axis} values`);
  if ("error" in values) return values;
  return { protocol: { kind: "values", values: values.ok } };
}

/** How many steps a protocol imposes after the starting zero (main/materialTestScript.ts `strainPath`). */
export function materialTestStepsOf(p: MaterialTestProtocol): number {
  if (p.kind === "monotonic") return p.steps;
  if (p.kind === "cyclic") return p.amplitudes.length * 4 * p.stepsPerRamp;
  return p.values.length;
}

/**
 * The curve thinned for drawing: every `stride`-th point, the last, and every
 * reversal of strain (so a hysteresis loop keeps its corners). When the reversals
 * alone would pass twice the cap, the stride alone decides.
 */
export function materialTestPointsOf(strain: ArrayLike<number>, stress: ArrayLike<number>, max: number = MAX_DRAWN): MaterialTestPoints {
  const n = Math.min(strain.length, stress.length);
  const take = (idx: number[]): MaterialTestPoints => ({ strain: idx.map((i) => strain[i]!), stress: idx.map((i) => stress[i]!) });
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
