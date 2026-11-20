// The section thickness of every drawn face triangle that belongs to a shell
// or a membrane, read from what the model file archives. Pure: no three.js.
//
// Where the thickness lives (no schema field of its own; all of it is the
// archived OpenSees command):
// - a shell element (`ShellMITC4`, `ASDShellQ4`, ...) names its section by
//   `secTag`, the first arg after its nodes; the section's own params carry
//   the thickness: `ElasticMembranePlateSection E nu h rho`, `ElasticPlateSection
//   E nu h`, `PlateFiber matTag h`, `LayeredShell n (mat t)...` (the sum of its
//   layers) and `LadrunoShellModifier innerTag ...` (its inner section's);
// - a `LadrunoMembrane` row carries `-thick t -mat m` or `-stiff E nu t`
//   (`LadrunoMembrane._emit`); one declared with `pg=` is drawn by its mesh
//   cells, so its decl_params fields (`thickness`, `thick`, `stiff`, or an
//   `ElasticMembranePlateSection` `section`) give those cells theirs.
//
// No shell the bridge writes carries a mid-surface offset, so the slab is
// centred on the drawn surface. A thickness that cannot be read is 0: that
// triangle stays a zero-thickness face, never a guessed slab.

import type { ElementRef } from "../chain/resolve.ts";
import type { DeclValue, ElementMeta, ModelFile, Param } from "../model/types.ts";
import { MEMBRANE_TYPE } from "../mesh/membrane.ts";
import type { Decl, DeclPath } from "./types.ts";

/** The shell elements whose first arg after the nodes is a section tag (`OPS_ShellMITC4`, `OPS_ASDShellQ4`, ...). */
export const SHELL_TYPES: ReadonlySet<string> = new Set([
  "ShellMITC4",
  "ShellMITC9",
  "ShellDKGQ",
  "ShellDKGT",
  "ShellNLDKGQ",
  "ShellNLDKGT",
  "ASDShellQ4",
  "ASDShellT3",
]);

export interface ShellThickness {
  /** (nTri) float32 thickness per drawn face triangle, in `triRefs` order; 0 where none is known */
  readonly triThickness: Float32Array;
  /** how many triangles have a thickness > 0 */
  readonly count: number;
}

const positive = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) && v > 0 ? v : null);

/** The thickness a section declaration carries, following `LadrunoShellModifier` to its inner section. */
export function sectionThickness(tag: number, secByTag: ReadonlyMap<number, Decl>): number | null {
  let curr: number | null = tag;
  for (let depth = 0; depth < 5 && curr !== null; depth++) {
    const sec = secByTag.get(curr);
    if (!sec) return null;
    const p = sec.params;
    switch (sec.type) {
      case "ElasticMembranePlateSection":
      case "ElasticPlateSection":
        return positive(p[2]);
      case "PlateFiber":
        return positive(p[1]);
      case "LayeredShell":
      case "LayeredShellFiberSection": {
        const n = p[0];
        if (typeof n !== "number" || !Number.isInteger(n) || n < 1 || p.length < 1 + 2 * n) return null;
        let t = 0;
        for (let i = 0; i < n; i++) {
          const ti = positive(p[2 + 2 * i]);
          if (ti === null) return null;
          t += ti;
        }
        return t;
      }
      case "LadrunoShellModifier": {
        const inner = p[0];
        curr = typeof inner === "number" && Number.isInteger(inner) ? inner : null;
        break;
      }
      default:
        return null;
    }
  }
  return null;
}

/** The thickness of a `LadrunoMembrane` row: `-thick t` or the third value of `-stiff E nu t`. */
export function membraneThickness(args: readonly Param[]): number | null {
  const i = args.indexOf("-thick");
  if (i >= 0) return positive(args[i + 1]);
  const j = args.indexOf("-stiff");
  if (j >= 0) return positive(args[j + 3]);
  return null;
}

/** The thickness of a shell row: its section's, by the `secTag` that is its first arg. */
function shellRowThickness(meta: ElementMeta, row: number, secByTag: ReadonlyMap<number, Decl>): number | null {
  if (meta.type === MEMBRANE_TYPE) return membraneThickness(meta.args[row] ?? []);
  if (!SHELL_TYPES.has(meta.type)) return null;
  const secTag = meta.args[row]?.[0];
  return typeof secTag === "number" && Number.isInteger(secTag) ? sectionThickness(secTag, secByTag) : null;
}

/** A `pg=` membrane's thickness from its decl_params fields (`LadrunoMembrane`'s dataclass). */
function membraneFieldThickness(fields: Readonly<Record<string, DeclValue>>, decls: Readonly<Record<DeclPath, Decl>>): number | null {
  const direct = positive(fields["thickness"]) ?? positive(fields["thick"]);
  if (direct !== null) return direct;
  const stiff = fields["stiff"];
  if (Array.isArray(stiff)) return positive(stiff[2]);
  const sec = fields["section"];
  if (sec && typeof sec === "object" && !Array.isArray(sec) && typeof sec["$decl"] === "string") {
    const d = decls[sec["$decl"]];
    if (d?.type === "ElasticMembranePlateSection") return positive(d.params[2]);
  }
  return null;
}

/** fem element id -> thickness of the `pg=` membranes declared on its surface group. */
function meshedMembraneThickness(model: ModelFile, decls: Readonly<Record<DeclPath, Decl>>): Map<number, number> {
  const byGroup = new Map<string, number>();
  for (const rec of model.opensees?.definitions.decls?.records ?? []) {
    for (const p of rec.params) {
      const pg = p.params["pg"];
      if (p.type !== MEMBRANE_TYPE || typeof pg !== "string") continue;
      const t = membraneFieldThickness(p.params, decls);
      if (t !== null) byGroup.set(pg, t);
    }
  }
  const out = new Map<number, number>();
  if (!byGroup.size) return out;
  for (const g of [...model.physicalGroups, ...model.labels]) {
    const t = g.dim === 2 ? byGroup.get(g.name) : undefined;
    if (t === undefined) continue;
    for (const id of g.elementIds) out.set(id, t);
  }
  return out;
}

/** The thickness of every drawn face triangle (`MeshBuffers.triRefs`), 0 where it is not a shell or a membrane. */
export function shellThicknessOf(model: ModelFile, triRefs: readonly ElementRef[], decls: Readonly<Record<DeclPath, Decl>>): ShellThickness {
  const triThickness = new Float32Array(triRefs.length);
  const metas = model.opensees?.elementMeta ?? [];
  if (!metas.some((m) => SHELL_TYPES.has(m.type) || m.type === MEMBRANE_TYPE)) return { triThickness, count: 0 };

  const secByTag = new Map<number, Decl>();
  for (const d of Object.values(decls)) if (d.kind === "section" && d.tag !== null) secByTag.set(d.tag, d);

  // fem element id -> the thickness of the shell emitted on it (the first shell row that has one).
  const byFem = new Map<number, number>();
  for (const meta of metas) {
    if (!SHELL_TYPES.has(meta.type)) continue;
    meta.femEids.forEach((fe, row) => {
      if (fe < 0 || byFem.has(fe)) return;
      const t = shellRowThickness(meta, row, secByTag);
      if (t !== null) byFem.set(fe, t);
    });
  }
  const meshed = meshedMembraneThickness(model, decls);

  // An ops row's thickness, computed once per row (a membrane panel is many triangles).
  const opsCache = new Map<string, number>();
  let count = 0;
  triRefs.forEach((r, i) => {
    let t: number | null | undefined;
    if (r.kind === "fem") {
      const id = model.blocks[r.blockIndex]!.ids[r.row]!;
      t = byFem.get(id) ?? meshed.get(id);
    } else {
      const key = `${r.metaIndex}:${r.row}`;
      t = opsCache.get(key);
      if (t === undefined) {
        t = shellRowThickness(metas[r.metaIndex]!, r.row, secByTag) ?? 0;
        opsCache.set(key, t);
      }
    }
    if (t && t > 0) {
      triThickness[i] = t;
      count++;
    }
  });
  return { triThickness, count };
}
