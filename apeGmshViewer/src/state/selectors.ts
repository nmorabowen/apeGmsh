// Selectors: pure reads of the state that panels and the viewport render
// from. A panel imports this module and store.ts, nothing else of the app.

import type { Chain, ChainNode, Field } from "../chain/resolve.ts";
import type { OpsFamily, Param } from "../model/types.ts";
import { elementFields, elementLinks, type Lookup } from "./decls.ts";
import { pairSessions } from "../reader/geometry.ts";
import { PROVENANCE_ORIGIN_FROM } from "../reader/provenance.ts";
import { roleColouring } from "./roles.ts";
import { activationNotice, activationOf, type Activation } from "./activation.ts";
import { pairingNotice, pairResults, type ResultsPairing } from "./results.ts";

export type { Activation } from "./activation.ts";
export type { ResultsPairing } from "./results.ts";
import type {
  AnalysisDef,
  ArtifactInfo,
  Decl,
  DeclPath,
  DeclRecord,
  DeclSource,
  DeclValue,
  Definitions,
  DefObject,
  LegendEntry,
  Origin,
  Pick,
  SequenceZone,
  StageDef,
  State,
  WindowContent,
  WindowKind,
  WindowRow,
  ZoneStatus,
} from "./types.ts";

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

/** The rows of `/sequence/stages`: what `State.stage` indexes (0 when the zone is not ready). */
export const stageCount = (seq: SequenceZone): number => (seq.status === "ready" ? seq.stages.rows.length : 0);

/**
 * What a `/sequence/stages` row is, to tell whether a re-read still has the
 * same stage at that index: its `order`, `name` and `source` cells (the
 * whole row when a column is missing). Null when the index is not a row.
 */
export function stageIdentity(seq: SequenceZone, index: number): string | null {
  if (seq.status !== "ready" || !Number.isInteger(index) || index < 0 || index >= seq.stages.rows.length) return null;
  const row = seq.stages.rows[index]!;
  const at = ["order", "name", "source"].map((c) => seq.stages.columns.indexOf(c));
  return JSON.stringify(at.includes(-1) ? row : at.map((i) => row[i]));
}

/** Notices that are not warnings: the stale-geometry notice, and (V4b) why results or their stage activation are not drawn. */
export function noticesOf(s: State): string[] {
  const out: string[] = [];
  const n = geometryPairing(s).notice;
  if (n) out.push(n);
  if (s.results) {
    const p = pairingNotice(resultsPairing(s), baseName(s.results.path));
    if (p) out.push(p);
  }
  const a = activationNow(s);
  if (a) {
    const t = activationNotice(a.activation, a.stageName);
    if (t) out.push(t);
  }
  return out;
}

// -- V4b: results pairing, the time axis and stage activation ---------------------

/** Whether the read results file pairs with the loaded model by content (state/results.ts). */
export function resultsPairing(s: State): ResultsPairing {
  const m = s.artifacts.model;
  return pairResults(m?.status === "ready" ? { lineage: s.lineage } : null, s.results);
}

/** What the time scrubber shows: the results stage on screen, its step and time, or null off the results phases. */
export interface Timeline {
  /** the results stages, in reader order, with the index `setPhase` names */
  stages: { index: number; label: string }[];
  stage: number;
  step: number;
  steps: number;
  time: number;
  kind: "static" | "transient" | "mode";
  /** the current step's fetch: `loading`, `ready`, `failed: <why>` */
  fetch: string;
}

export function timelineOf(s: State): Timeline | null {
  const at = s.phase.at;
  if (at?.kind !== "results" || !s.results) return null;
  const st = s.results.stages[at.stage];
  if (!st) return null;
  const r = s.resultStep;
  const fetch = !r ? "loading" : r.status === "failed" ? `failed: ${r.error ?? "unknown error"}` : r.status;
  return {
    stages: s.results.stages.map((x, i) => ({ index: i, label: x.name === "" ? `${x.id} (unnamed)` : x.name })),
    stage: at.stage,
    step: at.step,
    steps: st.time.length,
    time: st.time[at.step]!,
    kind: st.kind,
    fetch,
  };
}

let activationMemo: { key: readonly unknown[]; value: { activation: Activation; stageName: string } | null } | null = null;

/** The stage activation of the results stage on screen (state/activation.ts), or null off the results phases. */
export function activationNow(s: State): { activation: Activation; stageName: string } | null {
  const at = s.phase.at;
  if (at?.kind !== "results" || !s.results) return null;
  const st = s.results.stages[at.stage];
  if (!st) return null;
  const key = [s.sequence, s.stageElements, s.definitions, st.name];
  if (activationMemo && activationMemo.key.every((k, i) => k === key[i])) return activationMemo.value;
  const value = { activation: activationOf(s.sequence, s.stageElements, s.definitions, st.name), stageName: st.name };
  activationMemo = { key, value };
  return value;
}

let inactiveMemo: { key: readonly unknown[]; value: { mask: Uint8Array; hidden: number; unmapped: number } | null } | null = null;

/**
 * The elements a `filtered` activation hides, as a mask over `mesh.elements`
 * (1 = not drawn), with the count hidden and the inactive tags that name no
 * drawn element; null when nothing is hidden by activation. The viewport
 * draws through renderer/visible.ts with it, the one visibility path.
 */
export function inactiveElements(s: State): { mask: Uint8Array; hidden: number; unmapped: number } | null {
  const a = activationNow(s);
  if (!a || a.activation.state !== "filtered" || !s.mesh) return null;
  const key = [a.activation, s.mesh, s.decls];
  if (inactiveMemo && inactiveMemo.key.every((k, i) => k === key[i])) return inactiveMemo.value;
  const tags = a.activation.inactiveTags;
  const mask = new Uint8Array(s.mesh.elements.length);
  const found = new Set<number>();
  s.mesh.elements.forEach((p, i) => {
    for (const m of s.decls[p]?.element?.metas ?? []) {
      if (tags.has(m.tag)) {
        mask[i] = 1;
        found.add(m.tag);
      }
    }
  });
  const hidden = mask.reduce((n, v) => n + v, 0);
  const value = { mask, hidden, unmapped: tags.size - found.size };
  inactiveMemo = { key, value };
  return value;
}

/**
 * The jump to a recorded source. A source with no digest (a pseudo-file, or a
 * file the writer could not read) opens only when main found its path on
 * disk; otherwise it is "source not recorded" and the button is off. There is
 * no edit check: with no digest there is nothing to compare.
 */
function jumpOf(src: DeclSource): { ok: true; source: DeclSource; label: string } | { ok: false; reason: string } {
  if (!src.recorded) return { ok: false, reason: `source not recorded (${src.file})` };
  return { ok: true, source: src, label: `${baseName(src.file)}:${src.line}` };
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
    return jumpOf(r.source);
  }
  if (d.provenance) return jumpOf(d.provenance);
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

// ---- model-definition windows (V3c0) ----------------------------------------

/** One value, short: a `$decl` ref as `-> key`, a `$struct` as `Class{...}`, an `$opaque` as its class. */
export function shortValue(v: DeclValue | Param[] | undefined): string {
  if (v === undefined) return "";
  if (v === null) return "null";
  if (Array.isArray(v)) {
    const head = (v as DeclValue[]).slice(0, 6).map(shortValue).join(", ");
    return v.length > 6 ? `[${head}, ... ${v.length}]` : `[${head}]`;
  }
  if (typeof v === "object") {
    if (typeof v["$decl"] === "string") return `-> ${v["$decl"]}`;
    if (typeof v["$struct"] === "string") return `${v["$struct"]}{...}`;
    if (typeof v["$opaque"] === "string") return v["$opaque"];
    return `{${Object.keys(v).join(", ")}}`;
  }
  return String(v);
}

/** A declaration's parameters by field name (its first owner), `field=value` each. */
export function namedParams(d: DeclRecord | null): string {
  const p = d?.params[0];
  if (!p) return "";
  return Object.entries(p.params).map(([k, v]) => `${k}=${shortValue(v)}`).join(" · ");
}

/** The `/opensees/decls` row that owns tag `tag` of allocator kind `kind`, or null. */
export function declOfTag(defs: Definitions, kind: string, tag: number | null): DeclRecord | null {
  if (!defs.decls || tag === null) return null;
  const run = defs.decls.tags.find((t) => t.kind === kind && tag >= t.first && tag < t.first + t.count);
  return run ? (defs.decls.records[run.decl] ?? null) : null;
}

/**
 * The declaration of each recorder group, from `rows/<store>`: "one per
 * group, in index order" (h5-schema.md), the index being the trailing `_<i>`
 * of the group name. No join (an empty map) when the counts differ.
 */
export function recorderDecls(defs: Definitions, recorders: readonly DefObject[], store: string): Map<string, DeclRecord> {
  const out = new Map<string, DeclRecord>();
  const rows = defs.decls?.rows[store];
  if (!defs.decls || !rows || rows.length !== recorders.length) return out;
  const idx = recorders.map((r) => Number(/_(\d+)$/.exec(r.groupName)?.[1] ?? Number.NaN));
  if (idx.some((i) => !Number.isInteger(i))) return out;
  const order = recorders.map((r, i) => ({ r, i: idx[i]! })).sort((a, b) => a.i - b.i);
  order.forEach(({ r }, k) => {
    const d = defs.decls!.records[rows[k]!];
    if (d) out.set(r.path, d);
  });
  return out;
}

const stageSection = (st: StageDef): string => `stage ${st.index} · ${st.name}`;
const countOf = (o: DefObject, table: string): number => o.tables[table]?.rows.length ?? 0;
const CHAIN = ["analysis", "integrator", "algorithm", "test", "system", "numberer", "handler"] as const;

function analysisRow(a: AnalysisDef, section: string): WindowRow {
  const t = a.attrs;
  const chain = CHAIN.filter((k) => t[k] !== undefined).map((k) => {
    const args = t[`${k}_args`];
    return `${k}=${shortValue(t[k])}` + (args === undefined ? "" : `(${(Array.isArray(args) ? args : [args]).map((x) => shortValue(x)).join(", ")})`);
  });
  const title = (["analysis", "integrator", "algorithm"] as const).filter((k) => t[k] !== undefined).map((k) => shortValue(t[k])).join(" · ");
  return { key: a.path, section, title: title || "(no chain tokens)", detail: chain.join(" · "), h5: a.path, declKey: null, select: null };
}

function patternRow(defs: Definitions, p: DefObject, section: string): WindowRow {
  const d = declOfTag(defs, "pattern", p.tag);
  const ref = p.attrs["series_ref"];
  const parts = [
    `${p.type} tag ${p.tag ?? "?"}`,
    typeof ref === "string" ? `series ${ref.slice(ref.lastIndexOf("/") + 1)}` : "",
    countOf(p, "loads") ? `${countOf(p, "loads")} loads` : "",
    countOf(p, "sps") ? `${countOf(p, "sps")} sps` : "",
    countOf(p, "element_loads") ? `${countOf(p, "element_loads")} element loads` : "",
    countOf(p, "sp_holds") ? `${countOf(p, "sp_holds")} holds` : "",
  ];
  return { key: p.path, section, title: d?.name || p.groupName, detail: parts.filter(Boolean).join(" · "), h5: p.path, declKey: d?.key ?? null, select: null };
}

function recorderRow(r: DefObject, section: string, d: DeclRecord | undefined): WindowRow {
  const file = r.attrs["file"], kind = r.attrs["kind"];
  const parts = [r.type, typeof file === "string" ? file : "", typeof kind === "string" ? kind : ""];
  return { key: r.path, section, title: d?.name || r.groupName, detail: parts.filter(Boolean).join(" · "), h5: r.path, declKey: d?.key ?? null, select: null };
}

const MATERIAL_KINDS: ReadonlySet<string> = new Set(["uniaxialMaterial", "nDMaterial", "section"]);

/**
 * What each model-definition window lists, from the store alone (ADR 0112
 * D6): the five V3c0 placeholders show these rows, and the V3 slice that owns
 * a window renders them in full. A row that is a declaration of the store (a
 * material or a section) carries it in `select`, which a click dispatches.
 */
export function windowContentOf(s: State, kind: WindowKind): WindowContent {
  const model = s.artifacts.model;
  if (!model || model.status !== "ready") return { rows: [], note: "no model is open" };
  const defs = s.definitions;
  if (kind === "materials") {
    const rows: WindowRow[] = Object.values(s.decls)
      .filter((d) => MATERIAL_KINDS.has(d.kind))
      .map((d) => {
        const rec = defs ? declOfTag(defs, d.kind, d.tag) : null;
        const named = namedParams(rec);
        const positional = d.params.map((x) => shortValue(x)).join(" ");
        return {
          key: d.h5,
          section: d.kind,
          title: d.name,
          detail: `${d.type} tag ${d.tag ?? "?"}` + (named ? ` · ${named}` : positional ? ` · ${positional}` : ""),
          h5: d.h5,
          declKey: rec?.key ?? null,
          select: d.path,
        };
      });
    return { rows, note: rows.length ? null : "the model declares no material or section" };
  }
  if (!defs) return { rows: [], note: "the model has no /opensees zone" };
  const rows: WindowRow[] = [];
  switch (kind) {
    case "stages": {
      for (const st of defs.stages) {
        const steps = st.attrs["analyze_steps"], dt = st.attrs["analyze_dt"];
        const parts = [
          steps !== undefined ? `${shortValue(steps)} steps` : "",
          dt !== undefined ? `dt ${shortValue(dt)}` : "static",
          `${st.patterns.length} patterns`,
          `${st.recorders.length} recorders`,
          st.bcs.fix ? `${st.bcs.fix.rows.length} fix` : "",
          ...st.other,
        ];
        rows.push({ key: st.path, section: "stages", title: `${st.index} · ${st.name}`, detail: parts.filter(Boolean).join(" · "), h5: st.path, declKey: null, select: null });
      }
      const seq = s.sequence;
      const seqNote =
        seq.status === "ready" ? `/sequence ${seq.version}: ${seq.stages.rows.length} stages, ${seq.changes.rows.length} changes` :
        seq.status === "refused" ? `/sequence ${seq.version} was not read (${seq.reason})` :
        "this file has no /sequence zone; the stages are read from /opensees/stages";
      const none = defs.analysis ? "unstaged model: one implicit stage, from /opensees/analysis" : "the model declares no stage";
      return { rows, note: rows.length ? seqNote : `${none}; ${seqNote}` };
    }
    case "analysis":
      if (defs.analysis) rows.push(analysisRow(defs.analysis, "global"));
      for (const st of defs.stages) if (st.analysis) rows.push(analysisRow(st.analysis, stageSection(st)));
      return { rows, note: rows.length ? null : "the model declares no analysis" };
    case "patterns": {
      for (const p of defs.patterns) rows.push(patternRow(defs, p, "global"));
      for (const st of defs.stages) for (const p of st.patterns) rows.push(patternRow(defs, p, stageSection(st)));
      for (const t of defs.timeSeries) {
        const d = declOfTag(defs, "timeSeries", t.tag);
        const named = namedParams(d);
        rows.push({ key: t.path, section: "time series", title: d?.name || t.groupName, detail: `${t.type} tag ${t.tag ?? "?"}` + (named ? ` · ${named}` : ""), h5: t.path, declKey: d?.key ?? null, select: null });
      }
      return { rows, note: rows.length ? null : "the model declares no pattern or time series" };
    }
    case "recorders": {
      const global = recorderDecls(defs, defs.recorders, "recorders");
      for (const r of defs.recorders) rows.push(recorderRow(r, "global", global.get(r.path)));
      for (const st of defs.stages) {
        const staged = recorderDecls(defs, st.recorders, `stages/stage_${String(st.index).padStart(3, "0")}/recorders`);
        for (const r of st.recorders) rows.push(recorderRow(r, stageSection(st), staged.get(r.path)));
      }
      return { rows, note: rows.length ? null : "the model declares no recorder" };
    }
    default: {
      const unknown: never = kind;
      throw new Error(`windowContentOf: unknown window ${JSON.stringify(unknown)}`);
    }
  }
}
