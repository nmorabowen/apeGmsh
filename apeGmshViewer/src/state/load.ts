// From a ModelFile to the `fileLoaded` payload: the declarations keyed by
// path, the name index, the render blobs and the artifact record. This is
// the one place that turns the reader's typed arrays into BlobRefs; after it
// runs, the state holds no array of the file.

import type { ElementRef } from "../chain/resolve.ts";
import { buildMesh, colourGroups, type MeshBuffers } from "../mesh/build.ts";
import type { Group, ModelFile, OpsFamily, Param } from "../model/types.ts";
import { sourceOf, type ProvenanceZone } from "../reader/provenance.ts";
import type { BlobStore } from "./blobs.ts";
import { cellPath, elementLinks, objectDecl, objectPaths, opsRowPath, type Lookup } from "./decls.ts";
import { groupAdjacency } from "./adjacency.ts";
import { assignSlots, NO_GROUP, OPS_ONLY, slotColour } from "./palette.ts";
import { emptyRecord } from "./reduce.ts";
import type { ArtifactInfo, BlockInfo, Decl, DeclPath, DeclSource, ElementFacts, LegendEntry, MeshInfo, ModelLoad, ProvenanceEntry, ZoneStatus } from "./types.ts";

/** The legend rows that are not a physical group (build.ts names them). */
const NO_GROUP_ROW = "(no physical group)";
const OPS_ONLY_ROW = "(OpenSees-only, no cell)";

export const groupPath = (g: Group, kind: "physical_group" | "label"): DeclPath => `mesh/${kind}/${g.name}`;

function addName(names: Record<string, DeclPath[]>, name: string, path: DeclPath): void {
  (names[name] ??= []).push(path);
}

/** Build the declarations and the name index of a model. Exported for the tests; `loadModel` calls it. */
export function declarationsOf(model: ModelFile, blobs: BlobStore, warnings: string[]): { decls: Record<DeclPath, Decl>; names: Record<string, DeclPath[]> } {
  // Null-prototype records: a group or alias named `constructor` is a key, not Object.prototype's.
  const decls = emptyRecord<Decl>();
  const names = emptyRecord<DeclPath[]>();

  // Groups and labels: one decl each, their element ids as a blob.
  for (const [kind, list] of [["physical_group", model.physicalGroups], ["label", model.labels]] as const) {
    for (const g of list) {
      const path = groupPath(g, kind);
      if (path in decls) throw new Error(`${g.path}: a second ${kind} named "${g.name}"`);
      decls[path] = {
        path,
        kind,
        type: kind,
        name: g.name,
        nameSource: `${g.path} (group name)`,
        h5: g.path,
        tag: null,
        params: [g.dim, g.tag],
        refs: {},
        links: {},
        problems: [],
        fields: [
          { label: "dim", value: String(g.dim), source: `${g.path}@dim`, interpreted: false },
          { label: "tag", value: String(g.tag), source: `${g.path}@tag`, interpreted: false },
          { label: "elements", value: String(g.elementIds.length), source: `${g.path}/element_ids`, interpreted: false },
        ],
        group: { dim: g.dim, tag: g.tag, count: g.elementIds.length, ids: blobs.put(`model${g.path}/element_ids`, g.elementIds) },
      };
      addName(names, g.name, path);
    }
  }

  // OpenSees objects.
  const ids = objectPaths(model, warnings);
  const byTag = new Map<string, DeclPath[]>();
  const byH5 = new Map<string, DeclPath>();
  for (const [obj, id] of ids) {
    const key = `${obj.family}:${obj.tag}`;
    (byTag.get(key) ?? byTag.set(key, []).get(key)!).push(id.path);
    byH5.set(obj.path, id.path);
  }
  const lookup: Lookup = {
    byTag: (family: OpsFamily, tag: number) => byTag.get(`${family}:${tag}`) ?? [],
    byH5: (path: string) => byH5.get(path) ?? null,
  };
  for (const [obj, id] of ids) {
    decls[id.path] = objectDecl(obj, id, lookup);
    if (id.nameSource === "/opensees/names") addName(names, id.name, id.path);
  }

  // Elements: the element_meta rows joined to each cell by fem_eids, and the
  // physical groups that contain it.
  const hasOpensees = model.opensees !== null;
  const metaByFem = new Map<number, { type: string; tag: number; h5: string; row: number; args: readonly Param[] }[]>();
  for (const meta of model.opensees?.elementMeta ?? []) {
    meta.femEids.forEach((fe, row) => {
      if (fe < 0) return;
      const hit = { type: meta.type, tag: meta.ids[row]!, h5: meta.path, row, args: meta.args[row] ?? [] };
      const list = metaByFem.get(fe);
      if (list) list.push(hit);
      else metaByFem.set(fe, [hit]);
    });
  }
  const groupsByFem = new Map<number, DeclPath[]>();
  for (const g of model.physicalGroups) {
    const path = groupPath(g, "physical_group");
    for (const e of g.elementIds) {
      const list = groupsByFem.get(e);
      if (list) list.push(path);
      else groupsByFem.set(e, [path]);
    }
  }
  model.blocks.forEach((b, block) => {
    const h5 = `/elements/${b.alias}`;
    for (let row = 0; row < b.ids.length; row++) {
      const femId = b.ids[row]!;
      const path = cellPath(femId);
      if (path in decls) throw new Error(`${h5}/ids[${row}]: FEM id ${femId} is declared twice`);
      // No node copy: the cell's tags are a range of State.blocks[block].connectivity (decision 15).
      const facts: ElementFacts = {
        femId,
        cell: { alias: b.alias, block, row },
        inlineNodes: NONE,
        groups: groupsByFem.get(femId) ?? NONE,
        metas: metaByFem.get(femId) ?? NONE,
      };
      decls[path] = elementDecl(path, h5, `element ${femId}`, facts, hasOpensees, lookup);
    }
  });
  model.opensees?.elementMeta.forEach((meta) => {
    meta.femEids.forEach((fe, row) => {
      if (fe >= 0) return;
      const path = opsRowPath(meta.type, row);
      const facts: ElementFacts = {
        femId: null,
        cell: null,
        inlineNodes: meta.inlineConnectivity?.[row] ?? NONE,
        groups: NONE,
        metas: [{ type: meta.type, tag: meta.ids[row]!, h5: meta.path, row, args: meta.args[row] ?? NONE }],
      };
      decls[path] = elementDecl(path, meta.path, `element ${meta.ids[row]}`, facts, hasOpensees, lookup);
    });
  });
  return { decls, names };
}

/** Shared frozen empties: an element decl carries no per-element copy of them. */
const NONE: readonly never[] = Object.freeze([]);
const NO_REFS: Readonly<Record<string, never>> = Object.freeze(Object.create(null) as Record<string, never>);

function elementDecl(path: DeclPath, h5: string, name: string, facts: ElementFacts, hasOpensees: boolean, lookup: Lookup): Decl {
  // Only the refs are kept per element; the chain selector re-derives the
  // link fields and the problems from the facts when the element is shown.
  const { refs } = elementLinks(facts, hasOpensees, lookup, true);
  const first = facts.metas[0];
  return {
    path,
    kind: "element",
    type: first?.type ?? "(no OpenSees type)",
    name,
    nameSource: h5,
    h5,
    tag: null,
    params: first?.args ?? NONE,
    refs: Object.keys(refs).length ? refs : NO_REFS,
    links: NO_REFS,
    problems: NONE,
    fields: NONE,
    element: facts,
  };
}

/** The neutral-zone blocks with their connectivity as blobs. */
export function blocksOf(model: ModelFile, blobs: BlobStore): BlockInfo[] {
  return model.blocks.map((b) => ({
    alias: b.alias,
    npe: b.npe,
    connectivity: blobs.put(`model/elements/${b.alias}/connectivity`, b.connectivity, [b.ids.length, b.npe]),
  }));
}

/** The decl path a mesh ref (a drawn primitive's element) points at. */
function pathOfRef(model: ModelFile, r: ElementRef): DeclPath {
  if (r.kind === "fem") return cellPath(model.blocks[r.blockIndex]!.ids[r.row]!);
  return opsRowPath(model.opensees!.elementMeta[r.metaIndex]!.type, r.row);
}

/** The render blobs, the per-primitive element and legend indices, and the coloured legend. */
export function meshInfoOf(model: ModelFile, mesh: MeshBuffers, blobs: BlobStore): MeshInfo {
  // Legend: build.ts's rows (most elements first). Groups that are neighbours
  // in the view (an element of each shares a node) get contrasting slots of
  // the office palette (R2, the #1330 ruling); the adjacency is kept in the
  // state so the assignment can be checked and never lives in a view.
  const { byElement, order } = colourGroups(model);
  const groups = mesh.legend.filter((e) => e.name !== NO_GROUP_ROW && e.name !== OPS_ONLY_ROW);
  const rowOf = new Map(groups.map((e, i) => [e.name, i]));
  const adjacency = groupAdjacency(model, byElement, order.length)
    .map(([a, b]): [number, number] | null => {
      const ra = rowOf.get(order[a]!), rb = rowOf.get(order[b]!);
      return ra === undefined || rb === undefined ? null : ra < rb ? [ra, rb] : [rb, ra];
    })
    .filter((p): p is [number, number] => p !== null)
    .sort((p, q) => p[0] - q[0] || p[1] - q[1]);
  const slots = assignSlots(groups.length, adjacency);
  const legend: LegendEntry[] = mesh.legend.map((e) => {
    const gi = rowOf.get(e.name);
    if (gi !== undefined) {
      const slot = slots[gi]!;
      const { color, ring } = slotColour(slot);
      return { decl: `mesh/physical_group/${e.name}`, name: e.name, color, elements: e.elements, cue: ring === 0 ? null : "stripe", slot };
    }
    return { decl: null, name: e.name, color: e.name === NO_GROUP_ROW ? NO_GROUP : OPS_ONLY, elements: e.elements, cue: null, slot: null };
  });
  const legendIndex = new Map(legend.map((e, i) => [e.name, i]));
  const noGroupRow = legendIndex.get(NO_GROUP_ROW) ?? -1;
  const opsOnlyRow = legendIndex.get(OPS_ONLY_ROW) ?? -1;

  const elements: DeclPath[] = [];
  const elementIndex = new Map<string, number>();
  const index = (refs: ElementRef[], what: string): { element: Int32Array; group: Int32Array } => {
    const element = new Int32Array(refs.length);
    const group = new Int32Array(refs.length);
    refs.forEach((r, i) => {
      const key = r.kind === "fem" ? `f${r.blockIndex}:${r.row}` : `o${r.metaIndex}:${r.row}`;
      let ei = elementIndex.get(key);
      if (ei === undefined) {
        ei = elements.length;
        elementIndex.set(key, ei);
        elements.push(pathOfRef(model, r));
      }
      element[i] = ei;
      let gi: number;
      if (r.kind === "ops") gi = opsOnlyRow;
      else {
        const og = byElement.get(model.blocks[r.blockIndex]!.ids[r.row]!);
        gi = og === undefined ? noGroupRow : (legendIndex.get(order[og]!) ?? -1);
      }
      if (gi < 0) throw new Error(`${what}[${i}]: no legend row for its colour group`);
      group[i] = gi;
    });
    return { element, group };
  };
  const lines = index(mesh.lineRefs, "lineRefs");
  const tris = index(mesh.triRefs, "triRefs");
  return {
    linePositions: blobs.put("model/mesh/linePositions", mesh.linePositions, [mesh.lineRefs.length, 6]),
    triPositions: blobs.put("model/mesh/triPositions", mesh.triPositions, [mesh.triRefs.length, 9]),
    edgePositions: blobs.put("model/mesh/edgePositions", mesh.edgePositions, [mesh.edgePositions.length / 6, 6]),
    lineElement: blobs.put("model/mesh/lineElement", lines.element),
    triElement: blobs.put("model/mesh/triElement", tris.element),
    lineGroup: blobs.put("model/mesh/lineGroup", lines.group),
    triGroup: blobs.put("model/mesh/triGroup", tris.group),
    elements,
    legend,
    adjacency,
    center: mesh.center,
    radius: mesh.radius,
    counts: mesh.counts,
    warnings: mesh.warnings,
  };
}

/**
 * Join /provenance records onto the declarations, by decl path. Unnamed
 * objects (`#k`) are never joined: the app's `#k` is the reader's listing
 * order and /provenance's is the declaration order, so the same key can name
 * two objects (selectors.sourceFor says so to the user). A record for a
 * family the app does not model (a fix, a pattern) joins nothing; that is
 * expected, not an error.
 */
export function joinProvenance(decls: Record<DeclPath, Decl>, zone: ProvenanceZone): number {
  let joined = 0;
  for (const path of zone.records.path) {
    if (path.includes("/#")) continue;
    const d = decls[path];
    if (!d) continue;
    const source = declSourceOf(zone, path);
    if (!source) continue;
    decls[path] = { ...d, provenance: source };
    joined++;
  }
  return joined;
}

/** Where go-to-source jumps for the record at `path`: its site, else its script line; null when it has neither. */
function declSourceOf(zone: ProvenanceZone, path: string): DeclSource | null {
  const r = sourceOf(zone, path);
  if (!r.ok) return null;
  const at = r.site ?? r.script!;
  return {
    file: at.file,
    line: at.line,
    function: at.function,
    sha256: at.sha256,
    script: r.script && r.site ? { file: r.script.file, line: r.script.line } : null,
  };
}

/**
 * Every /provenance record as a listing entry, in capture order. Synthesised
 * records are kept (the listing shows them by default); a synthesised
 * object's source is the user's call of the verb that made it.
 */
export function provenanceEntries(zone: ProvenanceZone): ProvenanceEntry[] {
  const rows = zone.records.path.map((key, i): ProvenanceEntry => ({
    key,
    origin: zone.records.origin[i]!,
    seq: zone.records.seq[i]!,
    source: declSourceOf(zone, key),
  }));
  return rows.sort((a, b) => a.seq - b.seq);
}

/**
 * Everything `fileLoaded` carries for a model artifact. `provenance` is the
 * file's /provenance zone when main read it (null: the file has none);
 * `provenanceRefused` is the zone's status when main could not read it (the
 * model loads all the same, without sources).
 */
export function loadModel(
  model: ModelFile,
  blobs: BlobStore,
  provenance: ProvenanceZone | null = null,
  provenanceRefused: ZoneStatus | null = null,
): ModelLoad {
  const warnings: string[] = [...model.warnings];
  const mesh = buildMesh(model);
  const { decls, names } = declarationsOf(model, blobs, warnings);
  if (provenance) {
    warnings.push(...provenance.warnings);
    joinProvenance(decls, provenance);
  }
  // A malformed zone is loud (the banner), not an "older apeGmsh" refusal.
  if (provenanceRefused?.status === "malformed") {
    warnings.push(`/provenance is malformed and was not read (go-to-source is off): ${provenanceRefused.reason}`);
  }
  const sid = model.meta["session_id"];
  const info: ArtifactInfo = {
    path: model.path,
    status: "ready",
    error: null,
    stale: false,
    sizeBytes: model.sizeBytes,
    readMs: model.readMs,
    zones: {
      neutral: { status: "ready", version: model.neutralVersion },
      opensees: model.opensees ? { status: "ready", version: model.opensees.version } : { status: "absent" },
      provenance: provenanceRefused ?? (provenance ? { status: "ready", version: provenance.version } : { status: "absent" }),
    },
    sessionId: typeof sid === "string" ? sid : null,
    warnings: [...warnings, ...mesh.warnings],
    counts: {
      nodes: model.nodeIds.length,
      cells: model.blocks.reduce((a, b) => a + b.ids.length, 0),
      opsOnly: mesh.counts.opsOnly,
    },
  };
  return {
    info,
    decls,
    names,
    blocks: blocksOf(model, blobs),
    mesh: meshInfoOf(model, mesh, blobs),
    provenance: provenance ? provenanceEntries(provenance) : [],
  };
}
