// Pure declaration search engine and reverse dependency index (ADR 0112 D6).
//
// Everything here reads `State` alone (never the BlobStore, never the DOM) and
// caches its per-model index on `s.decls` via a WeakMap so repeated queries in
// Spotlight (`Ctrl+K`) and the Search window (`windows/search.ts`) run in
// sub-millisecond time without duplicating state in the snapshot.

import { elementLinks, type Lookup } from "./decls.ts";
import { roleOfElement, roleOfGroup } from "./roles.ts";
import type { Decl, DeclKind, DeclPath, DeclRecord, DeclValue, OpsFamily, SearchClause, SearchField, SearchOp, State } from "./types.ts";

function formatDeclValue(v: DeclValue): string {
  if (v === null || typeof v === "string" || typeof v === "number" || typeof v === "boolean") return String(v);
  if (Array.isArray(v)) return `[${v.map(formatDeclValue).join(", ")}]`;
  if (typeof v["$decl"] === "string") return `-> ${v["$decl"]}`;
  if (typeof v["$struct"] === "string" && v.fields && typeof v.fields === "object" && !Array.isArray(v.fields)) {
    const body = Object.entries(v.fields)
      .map(([k, x]) => `${k}=${formatDeclValue(x)}`)
      .join(", ");
    return `${v["$struct"]}(${body})`;
  }
  return `{${Object.entries(v)
    .map(([k, x]) => `${k}: ${formatDeclValue(x as DeclValue)}`)
    .join(", ")}}`;
}

/** Cached per-`s.decls` reverse index of element memberships and definition-chain references. */
export interface ModelIndex {
  /** All declarations in deterministic path order. */
  readonly allDecls: readonly Decl[];
  /** Non-element declarations (`physical_group`, `label`, materials, sections, integrations, transforms). */
  readonly namedDecls: readonly Decl[];
  /** Element declarations only. */
  readonly elementDecls: readonly Decl[];
  /** Elements belonging to a `physical_group` or `label`, or using a material/section/integration/transf (transitively). */
  readonly elementsByDecl: ReadonlyMap<DeclPath, readonly DeclPath[]>;
  /** Direct + transitive definition-chain targets (`DeclPath`s) used by an element or object declaration. */
  readonly usedByDecl: ReadonlyMap<DeclPath, ReadonlySet<DeclPath>>;
  /** Named parameters (`name -> numeric or string value`) for each declaration path, joined from `/opensees/decl_params` and `d.fields`. */
  readonly paramsByDecl: ReadonlyMap<DeclPath, ReadonlyMap<string, number | string>>;
  /** Distinct OpenSees / cell types present in the model. */
  readonly types: readonly string[];
  /** Distinct physical group and label names present in the model. */
  readonly groups: readonly { path: DeclPath; name: string; kind: "physical_group" | "label"; count: number }[];
  /** Distinct chain target declarations (materials, sections, integrations, transforms) present in the model. */
  readonly chainTargets: readonly { path: DeclPath; name: string; kind: DeclKind; type: string; elements: number }[];
  /** Distinct parameter names present in the model. */
  readonly paramNames: readonly string[];
  /** Facet string lists for the Search window's autocomplete suggestions. */
  readonly facets: {
    readonly groups: readonly string[];
    readonly types: readonly string[];
    readonly kinds: readonly string[];
    readonly materialsAndSections: readonly string[];
    readonly roles: readonly string[];
    readonly paramKeys: readonly string[];
  };
}

const INDEX_CACHE = new WeakMap<State["decls"], ModelIndex>();

function buildLookup(decls: State["decls"]): Lookup {
  const byTag = new Map<string, DeclPath[]>();
  const byH5 = new Map<string, DeclPath>();
  for (const d of Object.values(decls)) {
    if (d.tag === null) continue;
    const key = `${d.kind}:${d.tag}`;
    const list = byTag.get(key);
    if (list) list.push(d.path);
    else byTag.set(key, [d.path]);
    byH5.set(d.h5, d.path);
  }
  return {
    byTag: (family: OpsFamily, tag: number) => byTag.get(`${family}:${tag}`) ?? [],
    byH5: (path: string) => byH5.get(path) ?? null,
  };
}

function collectDeclParams(s: State): Map<DeclPath, Map<string, number | string>> {
  const out = new Map<DeclPath, Map<string, number | string>>();
  const ensure = (p: DeclPath): Map<string, number | string> => {
    let m = out.get(p);
    if (!m) {
      m = new Map();
      out.set(p, m);
    }
    return m;
  };
  for (const d of Object.values(s.decls)) {
    const m = ensure(d.path);
    for (const f of d.fields) {
      if (f.label === "type" || f.label === "tag" || /^params\[\d+\]$/.test(f.label)) continue;
      const num = Number(f.value);
      m.set(f.label, Number.isFinite(num) && f.value.trim() !== "" ? num : f.value);
    }
  }
  const recs: readonly DeclRecord[] = s.definitions?.decls?.records ?? [];
  for (const r of recs) {
    if (!(r.key in s.decls)) continue;
    const m = ensure(r.key);
    for (const owner of r.params) {
      for (const [k, v] of Object.entries(owner.params)) {
        if (typeof v === "number" || typeof v === "string") m.set(k, v);
        else if (typeof v === "boolean") m.set(k, String(v));
        else if (v !== null && typeof v === "object") m.set(k, formatDeclValue(v as DeclValue));
      }
    }
  }
  return out;
}

/** Build or retrieve the cached `ModelIndex` for `s`. */
export function indexOfModel(s: State): ModelIndex {
  const cached = INDEX_CACHE.get(s.decls);
  if (cached) return cached;

  const allDecls = Object.values(s.decls);
  const namedDecls: Decl[] = [];
  const elementDecls: Decl[] = [];
  const lookup = buildLookup(s.decls);
  const hasOpensees = s.artifacts.model?.zones["opensees"]?.status === "ready";

  const directRefs = new Map<DeclPath, DeclPath[]>();
  const elementsByList = new Map<DeclPath, DeclPath[]>();
  const ensureList = (p: DeclPath): DeclPath[] => {
    let l = elementsByList.get(p);
    if (!l) {
      l = [];
      elementsByList.set(p, l);
    }
    return l;
  };

  // Also collect `$decl` references from `/opensees/decl_params` (e.g. FiberSection patches referencing materials).
  const declParamRefs = new Map<DeclPath, Set<DeclPath>>();
  const visitVal = (v: DeclValue, acc: Set<DeclPath>): void => {
    if (!v || typeof v !== "object") return;
    if (Array.isArray(v)) {
      for (const x of v) visitVal(x, acc);
      return;
    }
    if (typeof v["$decl"] === "string") {
      if (v["$decl"] in s.decls) acc.add(v["$decl"]);
      return;
    }
    if (typeof v["$struct"] === "string" && v.fields && typeof v.fields === "object" && !Array.isArray(v.fields)) {
      for (const x of Object.values(v.fields)) visitVal(x, acc);
      return;
    }
    for (const x of Object.values(v)) visitVal(x as DeclValue, acc);
  };
  for (const r of s.definitions?.decls?.records ?? []) {
    if (!(r.key in s.decls)) continue;
    const set = new Set<DeclPath>();
    for (const owner of r.params) {
      for (const list of Object.values(owner.refs)) {
        for (const target of list) if (target in s.decls) set.add(target);
      }
      for (const val of Object.values(owner.params)) visitVal(val, set);
    }
    if (set.size > 0) declParamRefs.set(r.key, set);
  }

  const typeSet = new Set<string>();
  const kindSet = new Set<string>();
  const roleSet = new Set<string>();

  for (const d of allDecls) {
    kindSet.add(d.kind);
    if (d.type) typeSet.add(d.type);
    if (d.kind === "element") {
      elementDecls.push(d);
      if (d.element?.cell?.alias) typeSet.add(d.element.cell.alias);
      for (const m of d.element?.metas ?? []) if (m.type) typeSet.add(m.type);
      const ownRefs = Object.values(d.refs);
      const refs = ownRefs.length > 0 ? ownRefs : d.element ? Object.values(elementLinks(d.element, hasOpensees, lookup, true).refs) : [];
      directRefs.set(d.path, refs);
      for (const g of d.element?.groups ?? []) ensureList(g).push(d.path);
    } else {
      namedDecls.push(d);
      if (d.kind === "physical_group") {
        const r = roleOfGroup(d);
        if (r) roleSet.add(r);
      }
      const extra = declParamRefs.get(d.path);
      const refs = new Set<DeclPath>(Object.values(d.refs));
      if (extra) for (const t of extra) refs.add(t);
      directRefs.set(d.path, [...refs]);
    }
  }

  // Compute transitive closure of `refs` for every declaration.
  const usedByDecl = new Map<DeclPath, ReadonlySet<DeclPath>>();
  const resolveUsed = (p: DeclPath, visiting: Set<DeclPath>): ReadonlySet<DeclPath> => {
    const hit = usedByDecl.get(p);
    if (hit) return hit;
    const out = new Set<DeclPath>();
    for (const target of directRefs.get(p) ?? []) {
      out.add(target);
      if (!visiting.has(target)) {
        visiting.add(target);
        for (const sub of resolveUsed(target, visiting)) out.add(sub);
        visiting.delete(target);
      }
    }
    usedByDecl.set(p, out);
    return out;
  };

  for (const d of allDecls) {
    const used = resolveUsed(d.path, new Set([d.path]));
    if (d.kind === "element") {
      for (const target of used) ensureList(target).push(d.path);
    }
  }

  const paramsByDecl = collectDeclParams(s);
  const paramNameSet = new Set<string>();
  for (const m of paramsByDecl.values()) {
    for (const k of m.keys()) paramNameSet.add(k);
  }

  const groups = namedDecls
    .filter((d): d is Decl & { kind: "physical_group" | "label" } => d.kind === "physical_group" || d.kind === "label")
    .map((d) => ({
      path: d.path,
      name: d.name,
      kind: d.kind,
      count: (elementsByList.get(d.path) ?? []).length || (d.group?.count ?? 0),
    }));

  const chainTargets = namedDecls
    .filter((d) => d.kind !== "physical_group" && d.kind !== "label")
    .map((d) => ({
      path: d.path,
      name: d.name,
      kind: d.kind,
      type: d.type,
      elements: (elementsByList.get(d.path) ?? []).length,
    }));

  const sortedTypes = [...typeSet].sort();
  const sortedParams = [...paramNameSet].sort();
  const index: ModelIndex = {
    allDecls,
    namedDecls,
    elementDecls,
    elementsByDecl: elementsByList,
    usedByDecl,
    paramsByDecl,
    types: sortedTypes,
    groups,
    chainTargets,
    paramNames: sortedParams,
    facets: {
      groups: groups.map((g) => g.name),
      types: sortedTypes,
      kinds: [...kindSet].sort(),
      materialsAndSections: chainTargets.map((c) => c.name),
      roles: [...roleSet].sort(),
      paramKeys: sortedParams,
    },
  };
  INDEX_CACHE.set(s.decls, index);
  return index;
}

export const indexOf = indexOfModel;

/**
 * Expand a declaration path into the element `DeclPath`s it represents in the
 * 3D viewport:
 * - an `element` declaration -> `[path]`
 * - a `physical_group` or `label` -> all elements in that group
 * - a material, section, beamIntegration or geomTransf -> all elements whose
 *   definition chain references it (directly or transitively).
 */
export function elementsOfDecl(s: State, path: DeclPath): readonly DeclPath[] {
  const d = s.decls[path];
  if (!d) return [];
  if (d.kind === "element") return [path];
  return indexOfModel(s).elementsByDecl.get(path) ?? [];
}

/**
 * Every element `DeclPath` that the current `s.selection.decls` highlights and
 * frames in the 3D viewport (expanding any selected group, label, material,
 * section, integration or transform into its member/using elements).
 */
export function selectedElementsOf(s: State): DeclPath[] {
  if (s.selection.decls.length === 0) return [];
  const out: DeclPath[] = [];
  const seen = new Set<DeclPath>();
  for (const p of s.selection.decls) {
    for (const e of elementsOfDecl(s, p)) {
      if (!seen.has(e)) {
        seen.add(e);
        out.push(e);
      }
    }
  }
  return out;
}

/**
 * The primary colour group of an element declaration: the smallest element-side
 * physical group holding it, ties broken by name (`mesh/build.ts` `colourGroups`
 * and `renderer/caps.ts` `colourGroupOf`).
 */
function colourGroupOfElement(s: State, d: Decl): DeclPath | null {
  let best: Decl | null = null;
  for (const p of d.element?.groups ?? []) {
    const g = s.decls[p];
    if (!g?.group) continue;
    if (!best || g.group.count < best.group!.count || (g.group.count === best.group!.count && g.name.localeCompare(best.name) < 0)) {
      best = g;
    }
  }
  return best?.path ?? null;
}

/**
 * Every element `DeclPath` hidden by `s.visibility.hidden` via non-legend
 * declarations (individual elements, materials, sections, etc.) or via its
 * primary `physical_group` colour group.
 */
export function hiddenElementsOf(s: State): ReadonlySet<DeclPath> {
  if (s.visibility.hidden.length === 0) return new Set();
  const out = new Set<DeclPath>();
  const hiddenGroups = new Set<DeclPath>();
  for (const p of s.visibility.hidden) {
    const d = s.decls[p];
    if (!d) continue;
    if (d.kind === "physical_group") {
      hiddenGroups.add(p);
      continue;
    }
    for (const e of elementsOfDecl(s, p)) out.add(e);
  }
  if (hiddenGroups.size > 0) {
    for (const p of hiddenGroups) {
      for (const e of elementsOfDecl(s, p)) {
        const ed = s.decls[e];
        if (ed && colourGroupOfElement(s, ed) === p) {
          out.add(e);
        }
      }
    }
  }
  return out;
}

// ---- Query AST & Parser -----------------------------------------------------

export interface ParsedTerm {
  readonly join: "and" | "or";
  readonly negate: boolean;
  readonly field: SearchField;
  readonly op: SearchOp;
  readonly key?: string;
  readonly value: string;
}

export type SearchTerm = ParsedTerm;

function normStr(s: string): string {
  return s
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase();
}

/** Human-readable chip label for a structured `SearchClause` or `ParsedTerm`. */
export function clauseLabel(c: SearchClause | ParsedTerm): string {
  const neg = c.negate ? "NOT " : "";
  if (c.field === "param") {
    if (c.key) return `${neg}${c.key} ${c.op === ":" || c.op === "contains" ? ":" : c.op} ${c.value}`;
    const sep = c.value.indexOf(":");
    if (sep >= 0) {
      const k = c.value.slice(0, sep);
      const v = c.value.slice(sep + 1);
      return `${neg}${k} ${c.op === ":" || c.op === "contains" ? ":" : c.op} ${v}`;
    }
  }
  if (c.field === "any") return `${neg}${c.value}`;
  const opStr = c.op === ":" || c.op === "contains" ? ":" : c.op;
  return `${neg}${c.field}${opStr}${c.value}`;
}

/** Convert a `ParsedTerm` into a `SearchClause` with a given `id`. */
export function clauseToSearchClause(t: ParsedTerm, id = "c-0"): SearchClause {
  return {
    id,
    join: t.join,
    negate: t.negate,
    field: t.field,
    op: t.op,
    ...(t.key !== undefined ? { key: t.key } : {}),
    value: t.value,
  };
}

const FIELD_ALIASES: Readonly<Record<string, SearchField>> = {
  any: "any",
  text: "any",
  name: "name",
  kind: "kind",
  family: "kind",
  type: "type",
  shape: "type",
  group: "group",
  pg: "group",
  label: "group",
  role: "role",
  uses: "uses",
  use: "uses",
  mat: "uses",
  material: "uses",
  sec: "uses",
  section: "uses",
  transf: "uses",
  integration: "uses",
  id: "id",
  tag: "id",
  femid: "id",
  param: "param",
  stage: "stage",
  vis: "visibility",
  visibility: "visibility",
};

const FILLER_WORDS = new Set([
  "select",
  "find",
  "filter",
  "show",
  "hide",
  "isolate",
  "search",
  "that",
  "are",
  "is",
  "controlled",
  "based",
  "by",
  "of",
  "with",
  "in",
  "the",
  "all",
]);

/**
 * Parse a human search query string into boolean `ParsedTerm` clauses.
 *
 * Examples:
 * - `"beams and dispBeamColumn"` -> two `and` terms
 * - `"select beams and displacement controlled elements"` -> `beams` AND `displacement` AND `elements`
 * - `"type:dispBeamColumn or type:forceBeamColumn"` -> two `or` terms
 * - `"group:Arch and not tag:1..10"` -> `group = Arch` AND NOT `id = 1..10`
 * - `"uses:Concrete02 E>2e10"` -> `uses = Concrete02` AND `param E > 2e10`
 */
export function parseSearchText(raw: string): ParsedTerm[] {
  const rawTokens = raw.trim().match(/(?:[^\s"]+|"[^"]*")+/g) ?? [];
  if (rawTokens.length === 0) return [];

  const terms: ParsedTerm[] = [];
  let join: "and" | "or" = "and";
  let negate = false;

  for (let i = 0; i < rawTokens.length; i++) {
    const rawTok = rawTokens[i]!;
    const quoted = rawTok.startsWith('"') && rawTok.endsWith('"');
    const hasTrailingComma = !quoted && rawTok.length > 1 && rawTok.endsWith(",");
    const strippedTok = hasTrailingComma ? rawTok.slice(0, -1) : rawTok;
    const tok = quoted ? strippedTok.slice(1, -1) : strippedTok;
    const low = normStr(tok);
    if (!quoted) {
      if (low === "and" || low === "&" || low === "&&") {
        join = "and";
        continue;
      }
      if (low === "or" || low === "|" || low === "||" || low === "," || low === "+") {
        join = "or";
        continue;
      }
      if (low === "not" || low === "!") {
        negate = !negate;
        continue;
      }
      if (FILLER_WORDS.has(low)) {
        continue;
      }
    }

    let termNeg = negate;
    negate = false;
    let body = tok;
    if (!quoted && (body.startsWith("!") || (body.startsWith("-") && body.length > 1 && !/^-\d/.test(body)))) {
      termNeg = !termNeg;
      body = body.slice(1);
    }

    const nextJoin: "and" | "or" = hasTrailingComma ? "or" : "and";

    // Check `field:value` or `field=value` or `paramOp` (`E>2e8`, `fc=-28`).
    const colon = /^([^\s:=<>!]+):(.+)$/.exec(body);
    if (colon) {
      const fKey = normStr(colon[1]!);
      const rhs = colon[2]!.replace(/^"+|"+$/g, "");
      const mapped = FIELD_ALIASES[fKey];
      if (mapped) {
        if (mapped === "param") {
          const pm = /^([A-Za-z0-9_.-]+)(!=|<=|>=|=|<|>)(.+)$/.exec(rhs);
          if (pm) {
            terms.push({
              join,
              negate: termNeg,
              field: "param",
              op: pm[2] as SearchOp,
              key: pm[1]!,
              value: pm[3]!.replace(/^"+|"+$/g, ""),
            });
          } else {
            terms.push({ join, negate: termNeg, field: "param", op: ":", value: rhs });
          }
        } else {
          terms.push({ join, negate: termNeg, field: mapped, op: ":", value: rhs });
        }
        join = nextJoin;
        continue;
      }
    }

    const cmp = /^([^\s:=<>!]+)(!=|<=|>=|=|<|>)(.+)$/.exec(body);
    if (cmp) {
      const lhs = cmp[1]!;
      const op = cmp[2] as SearchOp;
      const rhs = cmp[3]!.replace(/^"+|"+$/g, "");
      const mapped = FIELD_ALIASES[normStr(lhs)];
      if (mapped && mapped !== "param" && (op === "=" || op === "!=")) {
        terms.push({ join, negate: op === "!=" ? !termNeg : termNeg, field: mapped, op: "=", value: rhs });
      } else {
        terms.push({ join, negate: termNeg, field: "param", op, key: lhs, value: rhs });
      }
      join = nextJoin;
      continue;
    }

    if (body.startsWith("@") && body.length > 1) {
      terms.push({ join, negate: termNeg, field: "id", op: "=", value: body.slice(1) });
      join = nextJoin;
      continue;
    }

    terms.push({ join, negate: termNeg, field: "any", op: ":", value: body });
    join = nextJoin;
  }
  return terms;
}

function matchString(haystack: string, needle: string, op: SearchOp): boolean {
  const h = normStr(haystack);
  const n = normStr(needle).trim();
  if (!n) return true;
  if (op === "=") return h === n;
  if (op === "!=") return h !== n;
  if (h.includes(n)) return true;
  // Support sub-token matching across `_` and `-` (e.g. `beam_V50x70` matches `BeamO_V50x70`, `BeamX_V50x70`, `BeamY_V50x70`).
  if (/[_-]/.test(n)) {
    const parts = n.split(/[_-]+/).filter(Boolean);
    if (parts.length > 1 && parts.every((p) => h.includes(p))) return true;
  }
  return false;
}

function matchId(n: number | null, query: string): boolean {
  if (n === null) return false;
  const q = query.trim().replace(/^@|^#/, "");
  const range = /^(-?\d+)\.\.(-?\d+)$/.exec(q);
  if (range) {
    const a = Number(range[1]), b = Number(range[2]);
    return n >= Math.min(a, b) && n <= Math.max(a, b);
  }
  if (/^-?\d+$/.test(q)) return n === Number(q);
  return String(n).includes(q);
}

function matchNumberOp(actual: number | string | undefined, op: SearchOp, targetRaw: string): boolean {
  if (actual === undefined) return false;
  const targetNum = Number(targetRaw);
  const actualNum = typeof actual === "number" ? actual : Number(actual);
  if (Number.isFinite(targetNum) && Number.isFinite(actualNum)) {
    switch (op) {
      case "=":
      case ":":
      case "contains":
        return Math.abs(actualNum - targetNum) <= 1e-9 * Math.max(1, Math.abs(targetNum));
      case "!=":
        return Math.abs(actualNum - targetNum) > 1e-9 * Math.max(1, Math.abs(targetNum));
      case "<":
        return actualNum < targetNum;
      case "<=":
        return actualNum <= targetNum;
      case ">":
        return actualNum > targetNum;
      case ">=":
        return actualNum >= targetNum;
    }
  }
  return matchString(String(actual), targetRaw, op);
}

function matchesBeamSynonym(s: State, d: Decl): boolean {
  if (d.kind === "element") {
    if (/beam|frame/i.test(d.type) || roleOfElement(s, d.path) === "beam") return true;
    return (d.element?.groups ?? []).some((g) => /beam|girder/i.test(s.decls[g]?.name ?? g));
  }
  if (d.kind === "physical_group" || d.kind === "label") {
    return roleOfGroup(d) === "beam" || /beam|girder/i.test(d.name);
  }
  return false;
}

function matchesColumnSynonym(s: State, d: Decl): boolean {
  if (d.kind === "element") {
    if (roleOfElement(s, d.path) === "column") return true;
    return (d.element?.groups ?? []).some((g) => /col/i.test(s.decls[g]?.name ?? g));
  }
  if (d.kind === "physical_group" || d.kind === "label") {
    return roleOfGroup(d) === "column" || /col/i.test(d.name);
  }
  return false;
}

function matchesShellOrWallSynonym(s: State, d: Decl): boolean {
  if (d.kind === "element") {
    if (/shell/i.test(d.type) || roleOfElement(s, d.path) === "wall") return true;
    return (d.element?.groups ?? []).some((g) => /wall|slab|shell/i.test(s.decls[g]?.name ?? g));
  }
  if (d.kind === "physical_group" || d.kind === "label") {
    return roleOfGroup(d) === "wall" || /wall|slab|shell/i.test(d.name);
  }
  return false;
}

function matchTermOnDecl(s: State, idx: ModelIndex, hiddenSet: ReadonlySet<DeclPath>, d: Decl, term: Omit<ParsedTerm, "join" | "negate">): boolean {
  const val = term.value.trim();
  if (!val && !term.key) return true;
  const low = normStr(val);

  switch (term.field) {
    case "name":
      return matchString(d.name, val, term.op) || matchString(d.path, val, term.op);
    case "kind": {
      if (low === "beam" || low === "beams" || low === "frame" || low === "frames") {
        return matchesBeamSynonym(s, d);
      }
      if (low === "column" || low === "columns" || low === "col" || low === "cols") {
        return matchesColumnSynonym(s, d);
      }
      if (low === "element" || low === "elements" || low === "elem" || low === "elems") {
        return d.kind === "element";
      }
      if (low === "shell" || low === "shells" || low === "wall" || low === "walls" || low === "slab" || low === "slabs") {
        return matchesShellOrWallSynonym(s, d);
      }
      if (low === "solid" || low === "solids" || low === "brick" || low === "bricks") {
        return d.kind === "element" && /brick|tet|hex|solid|ssp/i.test(d.type);
      }
      if (low === "truss" || low === "trusses") {
        return d.kind === "element" && /truss|corottruss/i.test(d.type);
      }
      if (low === "material" || low === "materials") {
        return d.kind === "uniaxialMaterial" || d.kind === "nDMaterial";
      }
      if (low === "section" || low === "sections") {
        return d.kind === "section";
      }
      if (low === "group" || low === "groups") {
        return d.kind === "physical_group" || d.kind === "label";
      }
      return matchString(d.kind, val, term.op);
    }
    case "type": {
      const targetVal = low === "displacement" || low === "disp" ? "disp" : low === "force" ? "force" : val;
      if (matchString(d.type, targetVal, term.op)) return true;
      if (d.element?.cell && matchString(d.element.cell.alias, targetVal, term.op)) return true;
      if (d.element?.metas.some((m) => matchString(m.type, targetVal, term.op))) return true;
      return false;
    }
    case "group": {
      if (d.kind === "physical_group" || d.kind === "label") {
        return matchString(d.name, val, term.op) || matchString(d.path, val, term.op);
      }
      if (!d.element) return false;
      return d.element.groups.some((g) => {
        const gd = s.decls[g];
        return matchString(g, val, term.op) || (gd !== undefined && matchString(gd.name, val, term.op));
      });
    }
    case "role": {
      const r = d.kind === "element" ? roleOfElement(s, d.path) : d.kind === "physical_group" ? roleOfGroup(d) : null;
      return r !== null && matchString(r, val, term.op);
    }
    case "uses": {
      const used = idx.usedByDecl.get(d.path);
      if (!used || used.size === 0) return false;
      for (const targetPath of used) {
        const td = s.decls[targetPath];
        if (!td) continue;
        if (
          matchString(td.name, val, term.op) ||
          matchString(td.type, val, term.op) ||
          matchString(td.path, val, term.op) ||
          (td.tag !== null && matchId(td.tag, val))
        ) {
          return true;
        }
      }
      return false;
    }
    case "id": {
      if (d.tag !== null && matchId(d.tag, val)) return true;
      if (d.element) {
        if (matchId(d.element.femId, val)) return true;
        if (d.element.metas.some((m) => matchId(m.tag, val))) return true;
      }
      if (d.group && matchId(d.group.tag, val)) return true;
      return matchString(d.name, val, "=") || matchString(d.path, val, "=");
    }
    case "param": {
      let pName = term.key ?? "";
      let pOp: SearchOp = term.op;
      let pVal = val;
      if (!pName) {
        const sep = val.indexOf(":");
        if (sep >= 0) {
          pName = val.slice(0, sep).trim();
          pVal = val.slice(sep + 1).trim().replace(/^"(.*)"$/, "$1");
        } else {
          const m = /^([A-Za-z0-9_.-]+)\s*(!=|>=|<=|=|>|<)\s*(.+)$/.exec(val);
          if (m) {
            pName = m[1]!;
            pOp = m[2] as SearchOp;
            pVal = m[3]!.trim().replace(/^"(.*)"$/, "$1");
          } else {
            pName = val;
            pVal = "";
          }
        }
      }
      const checkParams = (p: DeclPath): boolean => {
        const map = idx.paramsByDecl.get(p);
        if (!map) return false;
        for (const [k, v] of map.entries()) {
          if (normStr(k) === normStr(pName) || (!pVal && matchString(k, pName, ":"))) {
            if (!pVal) return true;
            if (matchNumberOp(v, pOp, pVal)) return true;
          }
        }
        return false;
      };
      if (checkParams(d.path)) return true;
      // If `d` is an element or section, also match if any declaration in its chain has that parameter!
      for (const target of idx.usedByDecl.get(d.path) ?? []) {
        if (checkParams(target)) return true;
      }
      return false;
    }
    case "stage": {
      if (s.stageElements.status !== "ready") return false;
      const femId = d.element?.femId ?? null;
      const tags = d.element ? d.element.metas.map((m) => m.tag) : [];
      for (const [key, list] of Object.entries(s.stageElements.sets)) {
        if (matchString(key, val, term.op)) {
          if ((femId !== null && list.includes(femId)) || tags.some((t) => list.includes(t))) return true;
        }
      }
      return false;
    }
    case "visibility": {
      const isHid = s.visibility.hidden.includes(d.path) || (d.kind === "element" && hiddenSet.has(d.path));
      if (low === "hidden" || low === "off") {
        return isHid;
      }
      if (low === "visible" || low === "shown" || low === "on") {
        return !isHid;
      }
      return false;
    }
    case "any": {
      if (low === "beam" || low === "beams" || low === "frame" || low === "frames") {
        if (matchesBeamSynonym(s, d)) return true;
      }
      if (low === "column" || low === "columns" || low === "col" || low === "cols") {
        if (matchesColumnSynonym(s, d)) return true;
      }
      if (low === "element" || low === "elements" || low === "elem" || low === "elems") {
        return d.kind === "element";
      }
      if (low === "shell" || low === "shells" || low === "wall" || low === "walls" || low === "slab" || low === "slabs") {
        if (matchesShellOrWallSynonym(s, d)) return true;
      }
      if (low === "solid" || low === "solids" || low === "brick" || low === "bricks") {
        if (d.kind === "element" && /brick|tet|hex|solid|ssp/i.test(d.type)) return true;
      }
      if (low === "truss" || low === "trusses") {
        if (d.kind === "element" && /truss|corottruss/i.test(d.type)) return true;
      }
      if (low === "displacement" || low === "disp") {
        if (matchString(d.type, "disp", ":") || d.element?.metas.some((m) => matchString(m.type, "disp", ":"))) return true;
      }
      if (low === "force") {
        if (matchString(d.type, "force", ":") || d.element?.metas.some((m) => matchString(m.type, "force", ":"))) return true;
      }
      if (low === "hidden") {
        return s.visibility.hidden.includes(d.path) || (d.kind === "element" && hiddenSet.has(d.path));
      }
      if (low === "visible" || low === "shown") {
        return !(s.visibility.hidden.includes(d.path) || (d.kind === "element" && hiddenSet.has(d.path)));
      }
      if (matchString(d.name, val, ":") || matchString(d.type, val, ":") || matchString(d.path, val, ":") || matchString(d.kind, val, "=")) {
        return true;
      }
      if (d.tag !== null && matchId(d.tag, val)) return true;
      if (d.element) {
        if (matchId(d.element.femId, val)) return true;
        if (d.element.cell && matchString(d.element.cell.alias, val, ":")) return true;
        if (d.element.metas.some((m) => matchString(m.type, val, ":") || matchId(m.tag, val))) return true;
        for (const g of d.element.groups) {
          const gd = s.decls[g];
          if (gd && matchString(gd.name, val, ":")) return true;
        }
        const r = roleOfElement(s, d.path);
        if (r && matchString(r, val, ":")) return true;
      }
      // Also check if the element uses a material/section matching `val`
      const used = idx.usedByDecl.get(d.path);
      if (used) {
        for (const target of used) {
          const td = s.decls[target];
          if (td && (matchString(td.name, val, ":") || matchString(td.type, val, ":"))) return true;
        }
      }
      return false;
    }
  }
}

/** Combine structured `SearchClause[]` and free-text `text` into one evaluated predicate list. */
export function termsOfSearch(s: State): ParsedTerm[] {
  const structured: ParsedTerm[] = s.search.clauses.map((c: SearchClause) => ({
    join: c.join,
    negate: c.negate,
    field: c.field,
    op: c.op,
    ...(c.key !== undefined ? { key: c.key } : {}),
    value: c.value,
  }));
  return [...structured, ...parseSearchText(s.search.text)];
}

export interface SearchResultRow {
  readonly path: DeclPath;
  readonly kind: DeclKind;
  readonly name: string;
  readonly type: string;
  readonly detail: string;
  /** Number of 3D elements this declaration represents (1 for an element, N for a group/material/section). */
  readonly elementCount: number;
}

export interface SearchSummary {
  readonly terms: readonly ParsedTerm[];
  /** Total matched declarations. */
  readonly totalDecls: number;
  /** Alias for `totalDecls`. */
  readonly total: number;
  /** Distinct 3D element `DeclPath`s covered by the matched declarations (ready for batch Select / Hide / Isolate). */
  readonly elementDecls: readonly DeclPath[];
  /** Number of distinct 3D elements covered (`elementDecls.length`). */
  readonly elementCount: number;
  /** All matched declarations in deterministic order. */
  readonly matched: readonly Decl[];
  /** Capped result rows for rendering in the Search window or Spotlight. */
  readonly rows: readonly SearchResultRow[];
  /** Breakdown by `DeclKind`. */
  readonly byKind: Readonly<Record<string, number>>;
}

export const SEARCH_ROW_CAP = 80;

function detailOfDecl(s: State, idx: ModelIndex, d: Decl): { detail: string; elementCount: number } {
  if (d.kind === "element") {
    const groups = (d.element?.groups ?? []).map((g) => s.decls[g]?.name ?? g.split("/").pop()!);
    const meta = d.element?.metas[0];
    const used = [...(idx.usedByDecl.get(d.path) ?? [])]
      .map((p) => s.decls[p])
      .filter((x): x is Decl => x !== undefined && (x.kind === "section" || x.kind === "uniaxialMaterial" || x.kind === "nDMaterial"))
      .map((x) => `${x.name} (${x.type})`);
    const parts = [
      d.element?.cell ? `cell ${d.element.cell.alias}` : "OpenSees-only",
      meta ? `@${meta.tag}` : "",
      groups.length ? `in ${groups.join(", ")}` : "",
      used.length ? `uses ${used.slice(0, 2).join(", ")}` : "",
    ].filter(Boolean);
    return { detail: parts.join(" · "), elementCount: 1 };
  }
  if (d.kind === "physical_group" || d.kind === "label") {
    const elems = idx.elementsByDecl.get(d.path) ?? [];
    const role = roleOfGroup(d);
    const parts = [`dim ${d.group?.dim ?? "?"}`, `tag ${d.group?.tag ?? "?"}`, `${elems.length} elements`, role ? `role ${role}` : ""].filter(Boolean);
    return { detail: parts.join(" · "), elementCount: elems.length };
  }
  const elems = idx.elementsByDecl.get(d.path) ?? [];
  const params = idx.paramsByDecl.get(d.path);
  const pText = params
    ? [...params.entries()]
        .slice(0, 3)
        .map(([k, v]) => `${k}=${v}`)
        .join(" · ")
    : "";
  const parts = [d.tag !== null ? `tag ${d.tag}` : "", `${elems.length} elements`, pText].filter(Boolean);
  return { detail: parts.join(" · "), elementCount: elems.length };
}

const GENERIC_SYNONYMS = new Set([
  "beam",
  "beams",
  "frame",
  "frames",
  "column",
  "columns",
  "col",
  "cols",
  "element",
  "elements",
  "elem",
  "elems",
  "shell",
  "shells",
  "wall",
  "walls",
  "slab",
  "slabs",
  "solid",
  "solids",
  "brick",
  "bricks",
  "truss",
  "trusses",
  "displacement",
  "disp",
  "force",
  "hidden",
  "off",
  "visible",
  "shown",
  "on",
]);

function isGroupSelectorTerm(s: State, idx: ModelIndex, hiddenSet: ReadonlySet<DeclPath>, t: ParsedTerm): boolean {
  if (t.negate) return false;
  if (t.field === "group" || t.field === "name" || t.field === "id") return true;
  if (t.field !== "any") return false;
  const low = normStr(t.value.trim());
  if (!low || GENERIC_SYNONYMS.has(low)) return false;
  for (const d of idx.allDecls) {
    if ((d.kind === "physical_group" || d.kind === "label") && matchTermOnDecl(s, idx, hiddenSet, d, t)) {
      return true;
    }
  }
  return false;
}

/**
 * Evaluate a list of `ParsedTerm`s against `s` and return the summary, the
 * covered 3D element paths, and the display rows.
 */
export function runSearchTerms(s: State, terms: readonly ParsedTerm[], cap = SEARCH_ROW_CAP): SearchSummary {
  const idx = indexOfModel(s);
  const hiddenSet = hiddenElementsOf(s);
  const byKind: Record<string, number> = {};
  const matched: Decl[] = [];

  // Group `terms` into OR-separated conjunction (AND) groups so `A and B or C and D`
  // evaluates as `(A and B) or (C and D)`.
  const orGroups: ParsedTerm[][] = [];
  for (let i = 0; i < terms.length; i++) {
    const t = terms[i]!;
    if (i === 0 || t.join === "or") {
      orGroups.push([t]);
    } else {
      orGroups[orGroups.length - 1]!.push(t);
    }
  }

  // Pre-analyse each conjunction: if a conjunction contains 2+ positive group/name
  // selectors (e.g. `select beam_V50x70 and beam_V95x70` or two pinned `group:` chips)
  // whose strict intersection across elements is empty, treat those group selectors
  // as a union (`g1 OR g2 ...`) while still intersecting (`AND`) with all other filters.
  const evaluatedGroups = orGroups.map((group) => {
    let hasStrictElementHit = false;
    for (const d of idx.allDecls) {
      if (d.kind !== "element") continue;
      let ok = true;
      for (const t of group) {
        const rawHit = matchTermOnDecl(s, idx, hiddenSet, d, t);
        if (t.negate ? rawHit : !rawHit) {
          ok = false;
          break;
        }
      }
      if (ok) {
        hasStrictElementHit = true;
        break;
      }
    }
    if (hasStrictElementHit) {
      return { unionSelectors: [] as ParsedTerm[], mustMatch: group };
    }
    const unionSelectors: ParsedTerm[] = [];
    const mustMatch: ParsedTerm[] = [];
    for (const t of group) {
      if (isGroupSelectorTerm(s, idx, hiddenSet, t)) {
        unionSelectors.push(t);
      } else {
        mustMatch.push(t);
      }
    }
    if (unionSelectors.length >= 2) {
      return { unionSelectors, mustMatch };
    }
    return { unionSelectors: [] as ParsedTerm[], mustMatch: group };
  });

  for (const d of idx.allDecls) {
    let ok = evaluatedGroups.length === 0;
    for (const eg of evaluatedGroups) {
      let groupOk = true;
      if (eg.unionSelectors.length > 0) {
        const anyUnionHit = eg.unionSelectors.some((t) => matchTermOnDecl(s, idx, hiddenSet, d, t));
        if (!anyUnionHit) groupOk = false;
      }
      if (groupOk) {
        for (const t of eg.mustMatch) {
          const rawHit = matchTermOnDecl(s, idx, hiddenSet, d, t);
          const hit = t.negate ? !rawHit : rawHit;
          if (!hit) {
            groupOk = false;
            break;
          }
        }
      }
      if (groupOk) {
        ok = true;
        break;
      }
    }
    if (ok) {
      matched.push(d);
      byKind[d.kind] = (byKind[d.kind] ?? 0) + 1;
    }
  }

  // Order: physical_group first, then sections/materials/integrations, then labels, then elements.
  const kindRank = (k: DeclKind): number => {
    if (k === "physical_group") return 0;
    if (k === "section" || k === "uniaxialMaterial" || k === "nDMaterial" || k === "beamIntegration" || k === "geomTransf") return 1;
    if (k === "label") return 2;
    if (k === "element") return 4;
    return 3;
  };
  matched.sort((a, b) => {
    const rDiff = kindRank(a.kind) - kindRank(b.kind);
    if (rDiff !== 0) return rDiff;
    return a.path.localeCompare(b.path);
  });

  const elemSet = new Set<DeclPath>();
  for (const d of matched) {
    if (d.kind === "element") elemSet.add(d.path);
    else for (const e of idx.elementsByDecl.get(d.path) ?? []) elemSet.add(e);
  }
  const elementDecls = [...elemSet];

  const rows: SearchResultRow[] = matched.slice(0, cap).map((d) => {
    const { detail, elementCount } = detailOfDecl(s, idx, d);
    return {
      path: d.path,
      kind: d.kind,
      name: d.name,
      type: d.type,
      detail,
      elementCount,
    };
  });

  return {
    terms,
    totalDecls: matched.length,
    total: matched.length,
    elementDecls,
    elementCount: elementDecls.length,
    matched,
    rows,
    byKind,
  };
}

/** Evaluate the store's current `s.search` state. */
export function searchResultsOf(s: State, cap = SEARCH_ROW_CAP): SearchSummary {
  return runSearchTerms(s, termsOfSearch(s), cap);
}
