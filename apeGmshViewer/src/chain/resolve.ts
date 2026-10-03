// Resolve an element's definition chain from a ModelFile:
// element -> geomTransf -> beamIntegration -> section -> materials.
//
// Each field carries the HDF5 path it came from and whether the app had
// to interpret the file to get it (see signatures.ts). A link that cannot
// be resolved is reported in `problems`, by name; the chain never drops
// a link silently.

import type { ElementMeta, ModelFile, OpsFamily, OpsObject, Param, Table } from "../model/types.ts";
import {
  ELEMENT_SYNTAX,
  INTEGRATION_SYNTAX,
  SECTIONS_WITH_MATERIAL_REFS,
  SECTIONS_WITHOUT_MATERIALS,
  type Decoded,
} from "./signatures.ts";

export interface Field {
  label: string;
  value: string;
  /** HDF5 path (and `@attr` or `[index]`) the value came from. */
  source: string;
  /** true when the value needed OpenSees syntax knowledge, not just a read. */
  interpreted: boolean;
  note?: string;
}

export type Role = "element" | OpsFamily;

export interface ChainNode {
  role: Role;
  /** user alias from /opensees/names, else the HDF5 group name */
  name: string;
  nameSource: string;
  type: string;
  path: string;
  /** how the parent reached this node */
  via: Field | null;
  fields: Field[];
  children: ChainNode[];
}

export interface Chain {
  root: ChainNode;
  /** unresolved links, each naming what is missing */
  problems: string[];
}

/** What a pick refers to: a neutral-zone cell, or an OpenSees-only row. */
export type ElementRef =
  | { kind: "fem"; blockIndex: number; row: number }
  | { kind: "ops"; metaIndex: number; row: number };

interface MetaHit {
  meta: ElementMeta;
  row: number;
}

const femIndexCache = new WeakMap<ModelFile, Map<number, MetaHit[]>>();
const groupIndexCache = new WeakMap<ModelFile, Map<number, string[]>>();

function femIndex(model: ModelFile): Map<number, MetaHit[]> {
  let idx = femIndexCache.get(model);
  if (idx) return idx;
  idx = new Map();
  for (const meta of model.opensees?.elementMeta ?? []) {
    meta.femEids.forEach((fe, row) => {
      if (fe < 0) return;
      const list = idx!.get(fe);
      if (list) list.push({ meta, row });
      else idx!.set(fe, [{ meta, row }]);
    });
  }
  femIndexCache.set(model, idx);
  return idx;
}

function groupsOf(model: ModelFile, femId: number): string[] {
  let idx = groupIndexCache.get(model);
  if (!idx) {
    idx = new Map();
    for (const g of model.physicalGroups) {
      for (const e of g.elementIds) {
        const list = idx.get(e);
        if (list) list.push(g.name);
        else idx.set(e, [g.name]);
      }
    }
    groupIndexCache.set(model, idx);
  }
  return idx.get(femId) ?? [];
}

const NAME_KIND: Record<OpsFamily, string> = {
  uniaxialMaterial: "uniaxialMaterial",
  nDMaterial: "nDMaterial",
  section: "section",
  geomTransf: "geomTransf",
  beamIntegration: "beamIntegration",
};

function fmt(p: Param | unknown): string {
  if (typeof p === "number") return Number.isNaN(p) ? "NaN" : String(p);
  if (typeof p === "string") return p;
  return JSON.stringify(p);
}

/** Rows of a plain array the inspector prints in full before it summarises. */
const TABLE_ROWS_SHOWN = 8;

/**
 * An object's dataset as the inspector shows it. A plain numeric array (the
 * reader names its columns `value` or `[j]`) prints its values: one bracket
 * per row (`[1, 0, 0]`), or one list for a 1-D array (`[3, 4]`); a compound table prints its row count and column
 * names. A dataset with no values (a 2-D transform's `per_element_vecxz`,
 * shape (1, 0) by design) is "empty". Rows are listed in file order and
 * joined to nothing: elements reach a transform through its tag (#1295).
 */
export function formatTable(t: Table): string {
  if (t.rows.length === 0 || t.columns.length === 0) return "empty (no values)";
  const positional = t.columns.every((c, j) => (t.columns.length === 1 ? c === "value" : c === `[${j}]`));
  if (!positional) return `${t.rows.length} rows (${t.columns.join(", ")})`;
  const row = (r: unknown[]) => (r.length === 1 ? fmt(r[0]) : `[${r.map(fmt).join(", ")}]`);
  const shown = t.rows.slice(0, TABLE_ROWS_SHOWN).map(row);
  const flat = t.columns.length === 1 ? `[${shown.join(", ")}]` : shown.join("; ");
  const more = t.rows.length - shown.length;
  return more > 0 ? `${flat} ... (+${more} rows)` : flat;
}

function objectNode(model: ModelFile, obj: OpsObject, via: Field | null): ChainNode {
  const alias = model.opensees?.names.find((n) => n.kind === NAME_KIND[obj.family] && n.tag === obj.tag);
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
    fields.push({
      label: k,
      value: formatTable(t),
      source: t.path,
      interpreted: false,
    });
  }
  return {
    role: obj.family,
    name: alias ? alias.name : obj.groupName,
    nameSource: alias ? "/opensees/names" : `${obj.path} (group name)`,
    type: obj.type,
    path: obj.path,
    via,
    fields,
    children: [],
  };
}

function findByTag(model: ModelFile, family: OpsFamily, tag: number): OpsObject[] {
  return (model.opensees?.objects ?? []).filter((o) => o.family === family && o.tag === tag);
}

const FAMILY_DIR: Record<OpsFamily, string> = {
  uniaxialMaterial: "/opensees/materials/uniaxial",
  nDMaterial: "/opensees/materials/nd",
  section: "/opensees/sections",
  geomTransf: "/opensees/transforms",
  beamIntegration: "/opensees/beam_integration",
};

class Resolver {
  readonly problems: string[] = [];
  private readonly model: ModelFile;
  constructor(model: ModelFile) {
    this.model = model;
  }

  /** Follow the decoded tag slots of `args` (from `source`) to their objects. */
  followSlots(owner: string, decoded: Decoded, args: Param[], source: string, ownerType: string): ChainNode[] {
    if ("reason" in decoded) {
      this.problems.push(`${owner}: ${ownerType} args not decoded: ${decoded.reason}`);
      return [];
    }
    const out: ChainNode[] = [];
    for (const s of decoded.slots) {
      const tag = args[s.slot];
      if (typeof tag !== "number" || Number.isNaN(tag)) {
        this.problems.push(`${owner}: ${s.label} (slot ${s.slot}) is missing`);
        continue;
      }
      const via: Field = {
        label: s.label,
        value: `${tag} in ${FAMILY_DIR[s.family]}`,
        source: `${source}[${s.slot}]`,
        interpreted: true,
        note: `OpenSees ${ownerType} syntax "${decoded.syntax}"; matched on @tag`,
      };
      const hits = findByTag(this.model, s.family, tag);
      if (hits.length !== 1) {
        this.problems.push(
          `${owner}: ${s.label} ${tag} matches ${hits.length} objects in ${FAMILY_DIR[s.family]}`,
        );
        continue;
      }
      out.push(this.expand(objectNode(this.model, hits[0]!, via), hits[0]!));
    }
    return out;
  }

  expand(node: ChainNode, obj: OpsObject): ChainNode {
    if (obj.family === "beamIntegration") {
      const dec: Decoded = INTEGRATION_SYNTAX[obj.type]?.(obj.params) ?? { reason: `beamIntegration type ${obj.type} is not in the syntax table` };
      node.children = this.followSlots(obj.path, dec, obj.params, `${obj.path}@params`, `beamIntegration ${obj.type}`);
    } else if (obj.family === "section") {
      node.children = this.sectionMaterials(obj);
    }
    return node;
  }

  sectionMaterials(sec: OpsObject): ChainNode[] {
    if (SECTIONS_WITHOUT_MATERIALS.has(sec.type)) return [];
    if (!SECTIONS_WITH_MATERIAL_REFS.has(sec.type)) {
      this.problems.push(`${sec.path}: section type ${sec.type}: material links are not decoded`);
      return [];
    }
    // Fiber: every patch / fiber / layer row names its material by HDF5 path.
    const uses = new Map<string, string[]>();
    for (const t of Object.values(sec.tables)) {
      const col = t.columns.indexOf("material_ref");
      if (col < 0) continue;
      t.rows.forEach((r, i) => {
        const ref = String(r[col]);
        const list = uses.get(ref);
        const where = `${t.path}[${i}].material_ref`;
        if (list) list.push(where);
        else uses.set(ref, [where]);
      });
    }
    if (uses.size === 0) this.problems.push(`${sec.path}: Fiber section lists no material_ref`);
    const out: ChainNode[] = [];
    for (const [ref, wheres] of uses) {
      const obj = this.model.opensees?.objects.find((o) => o.path === ref);
      if (!obj) {
        this.problems.push(`${sec.path}: material_ref ${ref} does not exist in the file`);
        continue;
      }
      const via: Field = {
        label: "material_ref",
        value: `${ref} (${wheres.length} row${wheres.length === 1 ? "" : "s"})`,
        source: wheres[0]! + (wheres.length > 1 ? ` (+${wheres.length - 1})` : ""),
        interpreted: false,
      };
      out.push(objectNode(this.model, obj, via));
    }
    return out;
  }
}

/** Build the chain for one picked element. */
export function resolveChain(model: ModelFile, ref: ElementRef): Chain {
  const r = new Resolver(model);
  const fields: Field[] = [];
  let hits: MetaHit[] = [];
  let title: string;
  let path: string;

  if (ref.kind === "fem") {
    const b = model.blocks[ref.blockIndex];
    if (!b) throw new RangeError(`no element block ${ref.blockIndex}`);
    const femId = b.ids[ref.row]!;
    path = `/elements/${b.alias}`;
    title = `element ${femId}`;
    const conn = Array.from(b.connectivity.subarray(ref.row * b.npe, (ref.row + 1) * b.npe));
    fields.push(
      { label: "FEM id", value: String(femId), source: `${path}/ids[${ref.row}]`, interpreted: false },
      { label: "cell type", value: b.alias, source: `${path} (group name)`, interpreted: false },
      { label: "nodes", value: conn.join(", "), source: `${path}/connectivity[${ref.row}]`, interpreted: false },
      {
        label: "physical groups",
        value: groupsOf(model, femId).join(", ") || "(none)",
        source: "/physical_groups/element_side/*/element_ids",
        interpreted: false,
      },
    );
    hits = femIndex(model).get(femId) ?? [];
    if (!model.opensees) {
      r.problems.push("the file has no /opensees zone: there is no definition chain");
    } else if (hits.length === 0) {
      r.problems.push(`FEM element ${femId} has no row in /opensees/element_meta/*/fem_eids`);
    }
  } else {
    const meta = model.opensees?.elementMeta[ref.metaIndex];
    if (!meta) throw new RangeError(`no element_meta ${ref.metaIndex}`);
    hits = [{ meta, row: ref.row }];
    path = meta.path;
    title = `element ${meta.ids[ref.row]}`;
    const conn = meta.inlineConnectivity?.[ref.row];
    fields.push({
      label: "nodes",
      value: conn ? conn.join(", ") : "(none)",
      source: `${meta.path}/inline_connectivity[${ref.row}]`,
      interpreted: false,
      note: "OpenSees-only element: no neutral-zone cell",
    });
  }

  const children: ChainNode[] = [];
  for (const { meta, row } of hits) {
    const args = meta.args[row] ?? [];
    fields.push(
      { label: "OpenSees type", value: meta.type, source: `${meta.path} (group name)`, interpreted: false },
      { label: "OpenSees tag", value: String(meta.ids[row]), source: `${meta.path}/ids[${row}]`, interpreted: false },
      { label: "args", value: args.map(fmt).join(" "), source: `${meta.path}/args[${row}]`, interpreted: false },
    );
    const dec: Decoded = ELEMENT_SYNTAX[meta.type]?.(args) ?? { reason: `element type ${meta.type} is not in the syntax table` };
    children.push(...r.followSlots(`${meta.path}[${row}]`, dec, args, `${meta.path}/args[${row}]`, meta.type));
  }

  const root: ChainNode = {
    role: "element",
    name: title,
    nameSource: path,
    type: hits[0]?.meta.type ?? "(no OpenSees type)",
    path,
    via: null,
    fields,
    children,
  };
  return { root, problems: r.problems };
}

/** Flatten a chain depth-first (for tests and the measurement ledger). */
export function walk(node: ChainNode, depth = 0, out: { node: ChainNode; depth: number }[] = []) {
  out.push({ node, depth });
  for (const c of node.children) walk(c, depth + 1, out);
  return out;
}
