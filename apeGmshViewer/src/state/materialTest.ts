// ADR 0135 D4: the job that tests one declared uniaxial material, from the state
// alone. The material's command comes from `/opensees/materials` (`Decl.params`)
// and its dependencies from the `$decl` values of `/opensees/decl_params`, with
// the file's tags unchanged so that the slots holding a tag stay valid.
//
// A file without decl_params has no way to name dependencies: the material's own
// line is sent, and if it does reference another, the solver says so.

import type { DeclRecord, DeclValue, Definitions } from "../model/types.ts";
import { declOfTag } from "./selectors.ts";
import type { Decl, DeclPath, MaterialTestJob, MaterialTestLine, MaterialTestProtocol, State } from "./types.ts";

export type MaterialTestJobAnswer = { job: MaterialTestJob } | { error: string };

const KIND = "uniaxialMaterial";

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

/** The one tag `rec` owns as a uniaxial material, or why there is none. */
function tagOf(defs: Definitions, rec: DeclRecord): number | string {
  if (rec.family !== KIND) return `it references ${rec.key}, which is not a uniaxial material`;
  const runs = (defs.decls?.tags ?? []).filter((t) => t.decl === rec.row && t.kind === KIND);
  const only = runs[0];
  if (runs.length !== 1 || !only || only.count !== 1) return `${rec.key} does not own exactly one uniaxial material tag`;
  return only.first;
}

export function materialTestJobOf(s: State, path: DeclPath, protocol: MaterialTestProtocol): MaterialTestJobAnswer {
  const target = s.decls[path];
  if (!target) return { error: `${path} is not a declaration of this model` };
  if (target.kind !== KIND) return { error: `${path} is a ${target.kind}; the tester runs uniaxial materials` };
  if (target.tag === null) return { error: `${path} has no tag in this file` };

  const byTag = new Map<number, Decl[]>();
  for (const d of Object.values(s.decls)) {
    if (d.kind === KIND && d.tag !== null) (byTag.get(d.tag) ?? byTag.set(d.tag, []).get(d.tag)!).push(d);
  }
  const defs = s.definitions;
  const byKey = new Map((defs?.decls?.records ?? []).map((r) => [r.key, r]));
  const materials: MaterialTestLine[] = [];
  const done = new Set<number>();
  const open = new Set<number>();

  /** Depth first, so a dependency's line comes before the line naming it. Returns why it cannot, or null. */
  const visit = (d: Decl): string | null => {
    const tag = d.tag!;
    if (done.has(tag)) return null;
    if (open.has(tag)) return `material tag ${tag} references itself through its dependencies`;
    open.add(tag);
    const rec = defs ? declOfTag(defs, KIND, tag) : null;
    const keys: string[] = [];
    for (const owner of rec?.params ?? []) for (const v of Object.values(owner.params)) declKeysOf(v, keys);
    for (const key of keys) {
      const dep = byKey.get(key);
      if (!dep) return `${key} is not a declaration of this file`;
      const depTag = tagOf(defs!, dep);
      if (typeof depTag === "string") return depTag;
      const hits = byTag.get(depTag) ?? [];
      const hit = hits[0];
      if (hits.length !== 1 || !hit) return `uniaxial material tag ${depTag} (${key}) is not one material of this file`;
      const why = visit(hit);
      if (why) return why;
    }
    open.delete(tag);
    done.add(tag);
    materials.push({ type: d.type, tag, args: [...d.params] });
    return null;
  };
  const why = visit(target);
  if (why) return { error: `cannot test ${path}: ${why}` };
  return { job: { target: target.tag, materials, protocol } };
}
