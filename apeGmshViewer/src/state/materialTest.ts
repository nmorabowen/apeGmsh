// ADR 0135 D4: the job that tests one declared uniaxial or (D9, S5) nD material,
// from the state alone. A dependency may be of either family, and tags are per family.
// The material's command comes from `/opensees/materials` (`Decl.params`)
// and its dependencies from the `$decl` values of `/opensees/decl_params`, with
// the file's tags unchanged so that the slots holding a tag stay valid.
//
// A file without decl_params has no way to name dependencies: the material's own
// line is sent, and if it does reference another, the solver says so.
//
// Overrides (D6, D8) replace numeric slots of the material under test only, in
// this job only; the model is never changed. `materialTestSlotsOf` lists the
// slots the form offers, labelled by `params_names` when the file has them, else
// by the preceding flag and the position after it, and locks a slot holding a
// dependency's tag.

import type { DeclRecord, DeclValue, Definitions } from "../model/types.ts";
import { declOfTag } from "./declTags.ts";
import type { Decl, DeclPath, MaterialTestJob, MaterialTestLine, MaterialTestOverride, MaterialTestProtocol, MaterialTestSlot, State } from "./types.ts";

export type MaterialTestJobAnswer = { job: MaterialTestJob } | { error: string };

type Family = "uniaxialMaterial" | "nDMaterial";
const FAMILIES: readonly string[] = ["uniaxialMaterial", "nDMaterial"];
const isFamily = (k: string): k is Family => FAMILIES.includes(k);
const named = (k: Family): string => (k === "nDMaterial" ? "nD material" : "uniaxial material");

function declKeysOf(v: DeclValue, out: string[]): void {
  if (v === null || typeof v !== "object") return;
  if (Array.isArray(v)) {
    for (const x of v) declKeysOf(x, out);
    return;
  }
  if (typeof v["$decl"] === "string") {
    out.push(v["$decl"]);
    return;
  }
  for (const x of Object.values(v)) declKeysOf(x, out);
}

/** The one tag `rec` owns as a material, with its family, or why there is none. */
function tagOf(defs: Definitions, rec: DeclRecord): { family: Family; tag: number } | string {
  const family = rec.family;
  if (!isFamily(family)) return `it references ${rec.key}, which is not a material`;
  const runs = (defs.decls?.tags ?? []).filter((t) => t.decl === rec.row && t.kind === family);
  const only = runs[0];
  if (runs.length !== 1 || !only || only.count !== 1) return `${rec.key} does not own exactly one ${named(family)} tag`;
  return { family, tag: only.first };
}

export function materialTestJobOf(s: State, path: DeclPath, protocol: MaterialTestProtocol, overrides: readonly MaterialTestOverride[] = []): MaterialTestJobAnswer {
  const target = s.decls[path];
  if (!target) return { error: `${path} is not a declaration of this model` };
  if (!isFamily(target.kind)) return { error: `${path} is a ${target.kind}; the tester runs uniaxial and nD materials` };
  if (target.tag === null) return { error: `${path} has no tag in this file` };
  const nd = (protocol as { nd?: unknown }).nd !== undefined;
  if (target.kind === "nDMaterial" && protocol.kind === "stress") return { error: `${path}: stress control runs uniaxial materials` };
  if ((target.kind === "nDMaterial") !== nd) return { error: target.kind === "nDMaterial" ? `${path} needs a view and a loading mode` : `${path} is uniaxial and takes no nD loading` };

  const byTag = new Map<string, Decl[]>();
  for (const d of Object.values(s.decls)) {
    if (!isFamily(d.kind) || d.tag === null) continue;
    const at = `${d.kind} ${d.tag}`;
    (byTag.get(at) ?? byTag.set(at, []).get(at)!).push(d);
  }
  const defs = s.definitions;
  const byKey = new Map((defs?.decls?.records ?? []).map((r) => [r.key, r]));
  const materials: MaterialTestLine[] = [];
  const done = new Set<string>();
  const open = new Set<string>();

  /** Depth first, so a dependency's line comes before the line naming it. Returns why it cannot, or null. */
  const visit = (d: Decl): string | null => {
    const tag = d.tag!;
    const family = d.kind as Family;
    const at = `${family} ${tag}`;
    if (done.has(at)) return null;
    if (open.has(at)) return `${named(family)} tag ${tag} references itself through its dependencies`;
    open.add(at);
    const rec = defs ? declOfTag(defs, family, tag) : null;
    const keys: string[] = [];
    for (const owner of rec?.params ?? []) for (const v of Object.values(owner.params)) declKeysOf(v, keys);
    for (const key of keys) {
      const dep = byKey.get(key);
      if (!dep) return `${key} is not a declaration of this file`;
      const depTag = tagOf(defs!, dep);
      if (typeof depTag === "string") return depTag;
      const hits = byTag.get(`${depTag.family} ${depTag.tag}`) ?? [];
      const hit = hits[0];
      if (hits.length !== 1 || !hit) return `${named(depTag.family)} tag ${depTag.tag} (${key}) is not one material of this file`;
      const why = visit(hit);
      if (why) return why;
    }
    open.delete(at);
    done.add(at);
    materials.push({ ...(family === "nDMaterial" ? { family } : {}), type: d.type, tag, args: [...d.params] });
    return null;
  };
  const why = visit(target);
  if (why) return { error: `cannot test ${path}: ${why}` };
  if (overrides.length === 0) return { job: { target: target.tag, materials, protocol } };
  const slots = slotsOf(s, target, materials.slice(0, -1).map((m) => m.tag));
  const line = materials[materials.length - 1]!;
  const args = [...line.args];
  for (const o of overrides) {
    const slot = slots.find((x) => x.slot === o.slot);
    if (!slot) return { error: `cannot test ${path}: argument ${o.slot} is not a numeric slot of its command` };
    if (slot.locked) return { error: `cannot test ${path}: ${slot.label} is locked (${slot.locked})` };
    if (!Number.isFinite(o.value)) return { error: `cannot test ${path}: ${slot.label} must be a finite number` };
    args[o.slot] = o.value;
  }
  materials[materials.length - 1] = { ...line, args };
  return { job: { target: target.tag, materials, protocol } };
}

/** The numeric slots of `d`'s command, labelled, with a dependency's tag slot locked. */
function slotsOf(s: State, d: Decl, depTags: readonly number[]): MaterialTestSlot[] {
  const rec = s.definitions ? declOfTag(s.definitions, d.kind, d.tag) : null;
  const names = rec?.params[0]?.paramsNames ?? null;
  const named = names !== null && names.length === d.params.length;
  const out: MaterialTestSlot[] = [];
  let flag = "";
  let after = 0;
  let inFlag = false;
  d.params.forEach((v, i) => {
    if (typeof v === "string") {
      // a flag ("-iso") and the words right after it ("voce") name the slots that follow
      if (v.startsWith("-") || !inFlag) {
        flag = v;
        after = 0;
      } else flag += ` ${v}`;
      inFlag = true;
      return;
    }
    inFlag = false;
    after++;
    const label = named ? names[i]! : flag ? `${flag} · ${after}` : `slot ${i + 1}`;
    out.push({ slot: i, label, value: v, locked: null });
  });
  // A wrapper names its dependency by tag: lock the first slot holding each one.
  for (const tag of depTags) {
    const hit = out.find((x) => x.value === tag && x.locked === null);
    if (hit) hit.locked = `it holds the tag of material ${tag}, which this one depends on`;
  }
  return out;
}

/** The slots the form offers for `path`: every numeric argument of its command. */
export function materialTestSlotsOf(s: State, path: DeclPath): MaterialTestSlot[] {
  const d = s.decls[path];
  if (!d || !isFamily(d.kind)) return [];
  const probe = d.kind === "nDMaterial" ? { view: "ThreeDimensional" as const, mode: "uniaxialStress" as const, confinement: 0 } : undefined;
  const job = materialTestJobOf(s, path, { kind: "values", values: [0], ...(probe ? { nd: probe } : {}) });
  const depTags = "job" in job ? job.job.materials.slice(0, -1).map((m) => m.tag) : [];
  return slotsOf(s, d, depTags);
}
