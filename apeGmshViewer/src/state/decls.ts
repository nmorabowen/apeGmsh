// Declarations: how a ModelFile's objects and elements become `Decl`
// records keyed by declaration path (types.ts), and how an element's links
// are followed. The link semantics are the V1 chain resolver's
// (src/chain/resolve.ts): the same syntax table, the same `via` fields and
// the same problem texts. test/store.test.ts holds the two engines against
// each other on the fixture and on the synthetic fail-closed models.

import { formatTable, type Field } from "../chain/resolve.ts";
import {
  ELEMENT_SYNTAX,
  INTEGRATION_SYNTAX,
  SECTIONS_WITH_MATERIAL_REFS,
  SECTIONS_WITHOUT_MATERIALS,
  type Decoded,
} from "../chain/signatures.ts";
import type { ModelFile, OpsFamily, OpsObject, Param } from "../model/types.ts";
import type { Decl, DeclPath, ElementFacts } from "./types.ts";

export const FAMILY_DIR: Record<OpsFamily, string> = {
  uniaxialMaterial: "/opensees/materials/uniaxial",
  nDMaterial: "/opensees/materials/nd",
  section: "/opensees/sections",
  geomTransf: "/opensees/transforms",
  beamIntegration: "/opensees/beam_integration",
};

export function fmt(p: Param | unknown): string {
  if (typeof p === "number") return Number.isNaN(p) ? "NaN" : String(p);
  if (typeof p === "string") return p;
  return JSON.stringify(p);
}

/** What following links needs: the objects by family and tag, and by HDF5 path. */
export interface Lookup {
  byTag(family: OpsFamily, tag: number): DeclPath[];
  byH5(path: string): DeclPath | null;
}

export interface Links {
  refs: Record<string, DeclPath>;
  links: Record<string, Field>;
  problems: string[];
}

function uniqueKey(refs: Record<string, unknown>, label: string): string {
  if (!(label in refs)) return label;
  for (let i = 1; ; i++) if (!(`${label}#${i}` in refs)) return `${label}#${i}`;
}

/**
 * Follow the decoded tag slots of `args` (read from `source`) to their
 * declarations. With `refsOnly`, the link fields and problem texts are not
 * built (the loader keeps refs alone for every element; the selector builds
 * the texts for the one shown).
 */
function followSlots(out: Links, lookup: Lookup, owner: string, decoded: Decoded, args: readonly Param[], source: string, ownerType: string, refsOnly = false): void {
  if ("reason" in decoded) {
    if (!refsOnly) out.problems.push(`${owner}: ${ownerType} args not decoded: ${decoded.reason}`);
    return;
  }
  for (const s of decoded.slots) {
    const tag = args[s.slot];
    if (typeof tag !== "number" || Number.isNaN(tag)) {
      if (!refsOnly) out.problems.push(`${owner}: ${s.label} (slot ${s.slot}) is missing`);
      continue;
    }
    const hits = lookup.byTag(s.family, tag);
    if (hits.length !== 1) {
      if (!refsOnly) out.problems.push(`${owner}: ${s.label} ${tag} matches ${hits.length} objects in ${FAMILY_DIR[s.family]}`);
      continue;
    }
    const key = uniqueKey(out.refs, s.label);
    out.refs[key] = hits[0]!;
    if (refsOnly) continue;
    out.links[key] = {
      label: s.label,
      value: `${tag} in ${FAMILY_DIR[s.family]}`,
      source: `${source}[${s.slot}]`,
      interpreted: true,
      note: `OpenSees ${ownerType} syntax "${decoded.syntax}"; matched on @tag`,
    };
  }
}

/** A Fiber section's materials: every patch / fiber / layer row names one by HDF5 path. */
function sectionMaterials(out: Links, lookup: Lookup, sec: OpsObject): void {
  if (SECTIONS_WITHOUT_MATERIALS.has(sec.type)) return;
  if (!SECTIONS_WITH_MATERIAL_REFS.has(sec.type)) {
    out.problems.push(`${sec.path}: section type ${sec.type}: material links are not decoded`);
    return;
  }
  const uses = new Map<string, string[]>();
  for (const t of Object.values(sec.tables)) {
    const col = t.columns.indexOf("material_ref");
    if (col < 0) continue;
    t.rows.forEach((r, i) => {
      const ref = String(r[col]);
      const where = `${t.path}[${i}].material_ref`;
      const list = uses.get(ref);
      if (list) list.push(where);
      else uses.set(ref, [where]);
    });
  }
  if (uses.size === 0) out.problems.push(`${sec.path}: Fiber section lists no material_ref`);
  let i = 0;
  for (const [ref, wheres] of uses) {
    const to = lookup.byH5(ref);
    if (!to) {
      out.problems.push(`${sec.path}: material_ref ${ref} does not exist in the file`);
      continue;
    }
    const key = `material_ref#${i++}`;
    out.refs[key] = to;
    out.links[key] = {
      label: "material_ref",
      value: `${ref} (${wheres.length} row${wheres.length === 1 ? "" : "s"})`,
      source: wheres[0]! + (wheres.length > 1 ? ` (+${wheres.length - 1})` : ""),
      interpreted: false,
    };
  }
}

/** The links of an object: a beamIntegration's section (interpreted), a section's materials (read). */
export function objectLinks(obj: OpsObject, lookup: Lookup): Links {
  const out: Links = { refs: {}, links: {}, problems: [] };
  if (obj.family === "beamIntegration") {
    const dec: Decoded = INTEGRATION_SYNTAX[obj.type]?.(obj.params) ?? { reason: `beamIntegration type ${obj.type} is not in the syntax table` };
    followSlots(out, lookup, obj.path, dec, obj.params, `${obj.path}@params`, `beamIntegration ${obj.type}`);
  } else if (obj.family === "section") {
    sectionMaterials(out, lookup, obj);
  }
  return out;
}

/**
 * The links of an element: one decode per joined element_meta row. `hasOpensees`
 * is whether the file has an /opensees zone at all.
 */
export function elementLinks(facts: ElementFacts, hasOpensees: boolean, lookup: Lookup, refsOnly = false): Links {
  const out: Links = { refs: {}, links: {}, problems: [] };
  if (facts.cell && !refsOnly) {
    if (!hasOpensees) out.problems.push("the file has no /opensees zone: there is no definition chain");
    else if (facts.metas.length === 0) out.problems.push(`FEM element ${facts.femId} has no row in /opensees/element_meta/*/fem_eids`);
  }
  for (const m of facts.metas) {
    const dec: Decoded = ELEMENT_SYNTAX[m.type]?.(m.args as Param[]) ?? { reason: `element type ${m.type} is not in the syntax table` };
    followSlots(out, lookup, refsOnly ? "" : `${m.h5}[${m.row}]`, dec, m.args, refsOnly ? "" : `${m.h5}/args[${m.row}]`, m.type, refsOnly);
  }
  return out;
}

/** The read fields of an object: type, tag, positional params, other attributes, tables. */
export function objectFields(obj: OpsObject): Field[] {
  const fields: Field[] = [
    { label: "type", value: obj.type, source: `${obj.path}@type`, interpreted: false },
    { label: "tag", value: String(obj.tag), source: `${obj.path}@tag`, interpreted: false },
  ];
  obj.params.forEach((p, i) => {
    fields.push({
      label: `params[${i}]`,
      value: fmt(p),
      source: typeof p === "string" ? `${obj.path}@params_str[${i}]` : `${obj.path}@params[${i}]`,
      interpreted: false,
      note: "positional; the file carries no parameter name",
    });
  });
  for (const [k, v] of Object.entries(obj.attrs)) {
    fields.push({ label: k, value: Array.isArray(v) ? `[${v.map(fmt).join(", ")}]` : fmt(v), source: `${obj.path}@${k}`, interpreted: false });
  }
  for (const [k, t] of Object.entries(obj.tables)) {
    fields.push({ label: k, value: formatTable(t), source: t.path, interpreted: false });
  }
  return fields;
}

/**
 * The read fields of an element, derived from its facts (the chain selector
 * calls this on demand). `nodes` is the cell's node tags as the pick read
 * them from the connectivity blob; without them (a pinned element with no
 * pick) the row names the source and the count only.
 */
export function elementFields(facts: ElementFacts, groupNames: string[], nodes: readonly number[] | null, npe: number | null): Field[] {
  const fields: Field[] = [];
  if (facts.cell) {
    const path = `/elements/${facts.cell.alias}`;
    fields.push(
      { label: "FEM id", value: String(facts.femId), source: `${path}/ids[${facts.cell.row}]`, interpreted: false },
      { label: "cell type", value: facts.cell.alias, source: `${path} (group name)`, interpreted: false },
      {
        label: "nodes",
        value: nodes ? nodes.join(", ") : `${npe ?? "?"} node tags (select the element to list them)`,
        source: `${path}/connectivity[${facts.cell.row}]`,
        interpreted: false,
      },
      {
        label: "physical groups",
        value: groupNames.join(", ") || "(none)",
        source: "/physical_groups/element_side/*/element_ids",
        interpreted: false,
      },
    );
  } else {
    const m = facts.metas[0];
    fields.push({
      label: "nodes",
      value: facts.inlineNodes.length ? facts.inlineNodes.join(", ") : "(none)",
      source: `${m?.h5 ?? "?"}/inline_connectivity[${m?.row ?? "?"}]`,
      interpreted: false,
      note: "OpenSees-only element: no neutral-zone cell",
    });
  }
  for (const m of facts.metas) {
    fields.push(
      { label: "OpenSees type", value: m.type, source: `${m.h5} (group name)`, interpreted: false },
      { label: "OpenSees tag", value: String(m.tag), source: `${m.h5}/ids[${m.row}]`, interpreted: false },
      { label: "args", value: m.args.map(fmt).join(" "), source: `${m.h5}/args[${m.row}]`, interpreted: false },
    );
  }
  return fields;
}

/** The decl path of a neutral-zone cell. */
export const cellPath = (femId: number): DeclPath => `mesh/element/${femId}`;
/** The decl path of an OpenSees-only element row. */
export const opsRowPath = (type: string, row: number): DeclPath => `opensees/element/${type}#${row}`;

/**
 * Assign every OpenSees object its declaration path: `opensees/<family>/<name>`
 * when /opensees/names carries one for its family and tag, else
 * `opensees/<family>/#<k>`, k being the object's position in the reader's
 * listing of that family (the declaration order arrives with /provenance in
 * V2c/V2d). Two objects of one family with the same registered name is a
 * writer fault: the second gets `#k` appended and a warning names both.
 */
export function objectPaths(model: ModelFile, warnings: string[]): Map<OpsObject, { path: DeclPath; name: string; nameSource: string }> {
  const out = new Map<OpsObject, { path: DeclPath; name: string; nameSource: string }>();
  const taken = new Set<DeclPath>();
  const perFamily = new Map<OpsFamily, number>();
  for (const obj of model.opensees?.objects ?? []) {
    const k = perFamily.get(obj.family) ?? 0;
    perFamily.set(obj.family, k + 1);
    const alias = model.opensees!.names.find((n) => n.kind === obj.family && n.tag === obj.tag);
    let path = alias ? `opensees/${obj.family}/${alias.name}` : `opensees/${obj.family}/#${k}`;
    if (taken.has(path)) {
      const clash = path;
      path = `${path}#${k}`;
      warnings.push(`/opensees/names: ${obj.family} "${alias?.name}" names more than one object; ${obj.path} is listed as ${path}, not ${clash}`);
    }
    taken.add(path);
    out.set(obj, {
      path,
      name: alias ? alias.name : obj.groupName,
      nameSource: alias ? "/opensees/names" : `${obj.path} (group name)`,
    });
  }
  return out;
}

export function objectDecl(obj: OpsObject, id: { path: DeclPath; name: string; nameSource: string }, lookup: Lookup): Decl {
  const links = objectLinks(obj, lookup);
  return {
    path: id.path,
    kind: obj.family,
    type: obj.type,
    name: id.name,
    nameSource: id.nameSource,
    h5: obj.path,
    tag: obj.tag,
    params: obj.params,
    refs: links.refs,
    links: links.links,
    problems: links.problems,
    fields: objectFields(obj),
  };
}
