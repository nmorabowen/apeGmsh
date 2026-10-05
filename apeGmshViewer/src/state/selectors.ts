// Selectors: pure reads of the state that panels and the viewport render
// from. A panel imports this module and store.ts, nothing else of the app.

import type { Chain, ChainNode, Field } from "../chain/resolve.ts";
import type { OpsFamily } from "../model/types.ts";
import { elementFields, elementLinks, type Lookup } from "./decls.ts";
import { pairSessions } from "../reader/geometry.ts";
import { PROVENANCE_ORIGIN_FROM } from "../reader/provenance.ts";
import { roleColouring } from "./roles.ts";
import type { ArtifactInfo, Decl, DeclPath, DeclSource, LegendEntry, Origin, Pick, State, ZoneStatus } from "./types.ts";

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

/** The colour-by-role legend: the roles the file records, the unassigned row, and whether it records any. */
export const roleLegendOf = (s: State): { legend: LegendEntry[]; fileHasRoles: boolean } | null => {
  const r = roleColouring(s);
  return r ? { legend: r.legend, fileHasRoles: r.fileHasRoles } : null;
};

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

const baseName = (p: string): string => p.split(/[\\/]/).pop() ?? p;

/**
 * Whether the geometry sibling may be drawn, and the notice when it may not
 * (h5-schema.md, the pairing rule): it pairs with the model only when both
 * carry an equal `/meta/session_id`. With no model open it is drawn alone;
 * while the model is still loading it waits, with no notice.
 */
export function geometryPairing(s: State): { draw: boolean; notice: string | null } {
  const g = s.geometry;
  if (!g) return { draw: false, notice: null };
  const m = s.artifacts.model;
  if (!m) return { draw: true, notice: null };
  if (m.status !== "ready") return { draw: false, notice: null };
  const p = pairSessions(m.sessionId, g.sessionId, baseName(g.path));
  return p.paired ? { draw: true, notice: null } : { draw: false, notice: `Geometry not drawn: ${p.reason}.` };
}

/** Notices that are not warnings: the stale-geometry notice. */
export function noticesOf(s: State): string[] {
  const n = geometryPairing(s).notice;
  return n ? [n] : [];
}

/**
 * Where go-to-source would jump for `decl`, or why it cannot. An unnamed
 * object's path (`#k`) is never joined: the app numbers unnamed objects in
 * the reader's listing order and /provenance in declaration order, so one
 * `#k` can name two different objects.
 */
export function sourceFor(s: State, decl: DeclPath): { ok: true; source: DeclSource; label: string } | { ok: false; reason: string } {
  const d = s.decls[decl];
  if (!d) {
    // A /provenance record with no declaration in the app (a series, a
    // pattern, a synthesised object): the listing's go-to-source.
    const r = s.provenance.find((p) => p.key === decl);
    if (!r) return { ok: false, reason: `${decl} is not a declaration of the loaded model` };
    if (!r.source) return { ok: false, reason: `${decl} has no source frame in /provenance` };
    return { ok: true, source: r.source, label: `${baseName(r.source.file)}:${r.source.line}` };
  }
  if (d.provenance) return { ok: true, source: d.provenance, label: `${baseName(d.provenance.file)}:${d.provenance.line}` };
  const zone = s.artifacts.model?.zones["provenance"];
  if (!zone || zone.status === "absent") return { ok: false, reason: "the model file has no /provenance zone (written before apeGmsh recorded sources)" };
  if (zone.status === "refused") return { ok: false, reason: `the /provenance zone was refused: ${zone.reason}` };
  if (zone.status === "malformed") return { ok: false, reason: `the /provenance zone is malformed: ${zone.reason}` };
  if (d.kind === "element") {
    return { ok: false, reason: "an element has no provenance key yet: /provenance records the declaration, and the file does not join it to its elements" };
  }
  if (decl.includes("/#")) {
    return { ok: false, reason: "an unnamed declaration is not joined to /provenance (the app's #k is the reader's listing order, not the declaration order); give it a name= to reach its source" };
  }
  return { ok: false, reason: `no /provenance record for ${decl}` };
}

/**
 * The declaration an OpenSees object of the chain was read from, by its HDF5
 * group (a chain node names its group, not its decl path). Objects have one
 * group each; elements share their block's and are never looked up here.
 */
export function declOfH5(s: State, h5: string): DeclPath | null {
  for (const d of Object.values(s.decls)) if (d.tag !== null && d.h5 === h5) return d.path;
  return null;
}

/** One row of the sources listing (panels/sources.ts). */
export interface SourceRow {
  key: DeclPath;
  origin: Origin;
  /** `<verb>` of a synthesised key (`support` in `support:<stage>/hold`); null for the user's own declarations */
  verb: string | null;
  /**
   * The key cut after each `/`, the only places it may wrap: joined they are
   * the key; the first is the zone prefix (`opensees/`).
   */
  segments: string[];
  /** `file:line` of the go-to-source target, or null when the record has no frame */
  label: string | null;
  /** the button's tooltip: the absolute file, the line and the function; or the reason it is off */
  title: string;
  /** why go-to-source is off for this row (shown on a second line), or null when it can jump */
  off: string | null;
}

/** `a/b:c/d` -> [`a/`, `b:c/`, `d`]: the pieces between which a path may wrap. */
export function pathSegments(key: string): string[] {
  return key.match(/[^/]*\/|[^/]+$/g) ?? [];
}

/**
 * The sources listing: every /provenance record of the model, in capture
 * order. Synthesised objects are listed by default, marked by `origin`
 * (maintainer ruling on #1378); there is no filter that hides them.
 */
export function sourcesOf(s: State): SourceRow[] {
  return s.provenance.map((p) => {
    const where = sourceFor(s, p.key);
    const name = p.key.split("/").slice(2).join("/");
    const colon = name.indexOf(":");
    return {
      key: p.key,
      origin: p.origin,
      verb: p.origin === "synthesised" && colon > 0 ? name.slice(0, colon) : null,
      segments: pathSegments(p.key),
      label: where.ok ? where.label : null,
      title: where.ok ? `open ${where.source.file}:${where.source.line} (${where.source.function}) in the editor` : where.reason,
      off: where.ok ? null : where.reason,
    };
  });
}

/**
 * The sources panel's header: the record count, how many are synthesised,
 * and a notice when the records carry no `origin` column (a file below
 * 1.1.0), so none is marked: the app does not guess one from the key. The
 * notice follows the column, not the version: a 1.0.x file that carries the
 * column is read as written and needs none. `null` when there is nothing to
 * list: no model, no /provenance, or no record (the panel hides).
 */
export function sourcesHeaderOf(s: State): { count: number; synthesised: number; notice: string | null } | null {
  if (s.provenance.length === 0) return null;
  const zone = s.artifacts.model?.zones["provenance"];
  const version = zone?.status === "ready" ? zone.version.split(".").slice(0, 2).join(".") : "?";
  const { major, minor } = PROVENANCE_ORIGIN_FROM;
  return {
    count: s.provenance.length,
    synthesised: s.provenance.filter((p) => p.origin === "synthesised").length,
    notice: s.provenanceOrigin ? null : `provenance ${version}: synthesised records are not marked (origin is recorded from ${major}.${minor})`,
  };
}

/** The latest go-to-source answer for `decl`, when it failed. */
export function sourceFailure(s: State, decl: DeclPath): string | null {
  const l = s.source.last;
  return l && l.decl === decl && !l.ok ? (l.reason ?? "go-to-source failed") : null;
}

/** The load failure to show instead of the model, or null. */
export function failureOf(s: State): string | null {
  const a = s.artifacts.model;
  return a && a.status === "failed" ? `${a.path}: ${a.error ?? "failed"}` : null;
}
