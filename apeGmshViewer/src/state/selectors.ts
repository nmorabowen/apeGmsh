// Selectors: pure reads of the state that panels and the viewport render
// from. A panel imports this module and store.ts, nothing else of the app.

import type { Chain, ChainNode, Field } from "../chain/resolve.ts";
import type { OpsFamily } from "../model/types.ts";
import { elementFields, elementLinks, type Lookup } from "./decls.ts";
import type { ArtifactInfo, Decl, DeclPath, LegendEntry, Pick, State, ZoneStatus } from "./types.ts";

function lookupOf(s: State): Lookup {
  // Tags are not keys of the state; an element's links are followed by the
  // same table the loader used, over the objects the state holds.
  let byTag: Map<string, DeclPath[]> | null = null;
  let byH5: Map<string, DeclPath> | null = null;
  const build = () => {
    byTag = new Map();
    byH5 = new Map();
    for (const d of Object.values(s.decls)) {
      if (d.tag === null) continue;
      const key = `${d.kind}:${d.tag}`;
      (byTag.get(key) ?? byTag.set(key, []).get(key)!).push(d.path);
      byH5.set(d.h5, d.path);
    }
  };
  return {
    byTag: (family: OpsFamily, tag: number) => {
      if (!byTag) build();
      return byTag!.get(`${family}:${tag}`) ?? [];
    },
    byH5: (path: string) => {
      if (!byH5) build();
      return byH5!.get(path) ?? null;
    },
  };
}

/** A declaration as a chain node, with the links that could be followed as children. */
function nodeOf(s: State, d: Decl, via: Field | null, problems: string[], seen: Set<DeclPath>, nodes: readonly number[] | null): ChainNode {
  let fields: readonly Field[] = d.fields, links = d.links, refs = d.refs;
  if (d.element) {
    const hasOpensees = s.artifacts.model?.zones["opensees"]?.status === "ready";
    const l = elementLinks(d.element, hasOpensees, lookupOf(s));
    const npe = d.element.cell ? (s.blocks[d.element.cell.block]?.npe ?? null) : null;
    fields = elementFields(d.element, d.element.groups.map((g) => s.decls[g]?.name ?? g), nodes, npe);
    links = l.links;
    refs = l.refs;
    problems.push(...l.problems);
  } else {
    problems.push(...d.problems);
  }
  const children: ChainNode[] = [];
  for (const [key, to] of Object.entries(refs)) {
    const child = s.decls[to];
    if (!child) throw new Error(`${d.path}: ref ${key} points at ${to}, which is not a declaration`);
    const link = links[key];
    if (!link) throw new Error(`${d.path}: ref ${key} has no link field`);
    if (seen.has(to)) {
      problems.push(`${d.path}: ${key} -> ${to} closes a cycle`);
      continue;
    }
    children.push(nodeOf(s, child, link, problems, new Set([...seen, to]), null));
  }
  if (d.kind === "physical_group" || d.kind === "label") throw new Error(`${d.path}: a ${d.kind} has no definition chain`);
  return {
    role: d.kind,
    name: d.name,
    nameSource: d.nameSource,
    type: d.type,
    path: d.h5,
    via,
    fields: [...fields],
    children,
  };
}

/**
 * The definition chain of an element or an OpenSees object, or null when the
 * path is not declared or is a group (a group has members, not a chain).
 */
export function chainOf(s: State, path: DeclPath, nodes: readonly number[] | null = null): Chain | null {
  const d = s.decls[path];
  if (!d || d.kind === "physical_group" || d.kind === "label") return null;
  const problems: string[] = [];
  const root = nodeOf(s, d, null, problems, new Set([path]), nodes);
  return { root, problems };
}

/** The pick of a selected declaration, if it is selected (its node tags come from it). */
const pickOf = (s: State, decl: DeclPath): Pick | undefined => s.selection.picks.find((p) => p.decl === decl);

/** What the inspector shows: the pinned declarations, then the selection. */
export function inspected(s: State): { decl: DeclPath; pinned: boolean; chain: Chain }[] {
  const out: { decl: DeclPath; pinned: boolean; chain: Chain }[] = [];
  for (const p of s.inspector.pinned) {
    const chain = chainOf(s, p, pickOf(s, p)?.nodes ?? null);
    if (chain) out.push({ decl: p, pinned: true, chain });
  }
  for (const p of s.selection.decls) {
    if (s.inspector.pinned.includes(p)) continue;
    const chain = chainOf(s, p, pickOf(s, p)?.nodes ?? null);
    if (chain) out.push({ decl: p, pinned: false, chain });
  }
  return out;
}

export const legendOf = (s: State): LegendEntry[] => s.mesh?.legend ?? [];

export const isHidden = (s: State, decl: DeclPath | null): boolean => decl !== null && s.visibility.hidden.includes(decl);

/** The header's summary of the model artifact, or null when none is loaded. */
export function modelSummary(s: State): { name: string; stats: string } | null {
  const a = s.artifacts.model;
  if (!a || a.status !== "ready") return null;
  const v = (z: string) => {
    const st = a.zones[z];
    return st?.status === "ready" ? st.version : null;
  };
  const ops = v("opensees");
  return {
    name: a.path.split(/[\\/]/).pop() ?? a.path,
    stats:
      `${a.counts.nodes.toLocaleString()} nodes · ${a.counts.cells.toLocaleString()} cells` +
      (a.counts.opsOnly ? ` (+${a.counts.opsOnly.toLocaleString()} OpenSees-only)` : "") +
      ` · neutral ${v("neutral") ?? "?"}` +
      (ops ? ` · opensees ${ops}` : " · no /opensees"),
  };
}

/** Loud reader and mesh warnings of every artifact. */
export function warningsOf(s: State): string[] {
  const out: string[] = [];
  for (const a of Object.values(s.artifacts)) if (a) out.push(...a.warnings);
  if (s.mesh) out.push(...s.mesh.warnings);
  return out;
}

/** The zones a reader refused, as one sentence each (never a blank view). */
export function refusalsOf(s: State): string[] {
  const out: string[] = [];
  for (const [kind, a] of Object.entries(s.artifacts) as [string, ArtifactInfo | null][]) {
    if (!a) continue;
    for (const [zone, st] of Object.entries(a.zones) as [string, ZoneStatus][]) {
      if (st.status !== "refused") continue;
      out.push(
        `${kind} file written by ${st.newer ? "a newer" : "an older"} apeGmsh: its ${zone} zone is version ${st.version}, ` +
          `this app reads ${st.accepted} (${st.reason})`,
      );
    }
  }
  return out;
}

/** The load failure to show instead of the model, or null. */
export function failureOf(s: State): string | null {
  const a = s.artifacts.model;
  return a && a.status === "failed" ? `${a.path}: ${a.error ?? "failed"}` : null;
}
