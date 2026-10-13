// The materials and sections window (V3c4, #1645; ADR 0112 D6, D7): every
// uniaxial and nD material, every section and every beam integration (the
// declarations that place sections) by its handle, with its type, its tag as
// evidence, and its parameters by field name from /opensees/decl_params.
//
// A `{"$decl": key}` value is a link: a click selects that declaration through
// the store (it never calls another panel). Each row also says who uses it:
// the elements and objects whose definition-chain refs point at it (the
// store's `refs`, the inspector's own data), and the declarations whose
// decl_params name it. The window lists; the inspector shows one chain (D7).
//
// A uniaxial material can also be named by g.reinforce LadrunoEmbeddedRebar
// ties (`-bond <tag>`, /opensees/constraints/embeddedRebar, opensees 2.29;
// V4m #1846). Ties are not elements and have no declaration row, so they are
// counted from `definitions.embeddedRebar` by tag, as "N embeddedRebar ties".
// When any tie's bond material cannot be read (no tag, a non-integer one,
// `-bond` twice, or a tag no uniaxial material of the file carries), a
// uniaxial row says "users unknown: referenced by embeddedRebar ties", never
// 0 and never "used by nothing".
//
// Parameters come from decl_params only: a file without it says so, and the
// positional argv is never re-read here as names. Names are handles, tags
// evidence (D2c); an unnamed declaration shows its `#k` key.
//
// `materialsOf` is the pure (state) -> data half; `view` turns it into DOM.

import type { Event, State } from "../../state/store.ts";
import { declOfTag } from "../../state/selectors.ts";
import type { Decl, DeclPath, DeclRecord, DeclValue, Definitions } from "../../state/types.ts";
import { el } from "../../ui/dom.ts";
import type { WindowView } from "./view.ts";

/** The families the window lists, in display order, with each group's heading. */
export const GROUPS = [
  { kind: "uniaxialMaterial", title: "uniaxial materials" },
  { kind: "nDMaterial", title: "nD materials" },
  { kind: "section", title: "sections" },
  { kind: "beamIntegration", title: "beam integrations (section placement)" },
] as const;

/** A list shows this many items; the rest are counted. */
export const LIST_CAP = 8;

/** One parameter value, ready to draw. */
export type Val =
  | { k: "text"; text: string }
  | { k: "null" }
  | { k: "ref"; ref: Ref }
  | { k: "struct"; cls: string; fields: [string, Val][] }
  | { k: "opaque"; cls: string }
  | { k: "list"; items: Val[]; more: number }
  | { k: "object"; fields: [string, Val][] };

/** A link to another declaration: what it reads as, and the store declaration a click selects (null with the reason when there is none). */
export interface Ref {
  /** the `/opensees/decls` key */
  key: string;
  /** `<family> <handle>`, the handle being the name or `#k` */
  label: string;
  target: DeclPath | null;
  why: string | null;
}

export interface MaterialRow {
  path: DeclPath;
  kind: string;
  /** the OpenSees type as the store read it */
  type: string;
  /** the name, or `#k` when unnamed */
  handle: string;
  tag: number | null;
  /** the `/opensees/decls` key that owns the tag; null when none does */
  declKey: string | null;
  h5: string;
  /** one entry per decl_params owner (one for a material or section) */
  owners: { type: string; fields: [string, Val][] }[];
  /** why there are no parameters by field name, or null */
  paramsNote: string | null;
  /** from the definition chain: elements and objects whose refs point here; `ties`, the embeddedRebar ties whose `-bond` tag is this material's */
  usedBy: { elements: number; objects: Ref[]; ties: number };
  /** a uniaxial material's tie users cannot all be known: why (one entry per reason, with its tie count, null when unknown); null when they can */
  tiesUnknown: { why: string; ties: number | null }[] | null;
  /** from decl_params: declarations whose parameters reference this one */
  referencedBy: { ref: Ref; type: string; elements: number }[];
}

export interface MaterialsData {
  groups: { kind: string; title: string; rows: MaterialRow[] }[];
  /** a sentence for the whole window (nothing to list, or a zone not in this file); null when there is none */
  note: string | null;
}

const LISTED: ReadonlySet<string> = new Set(GROUPS.map((g) => g.kind));

/** `opensees/<family>/<name|#k>` -> `<name|#k>` */
const keyHandle = (key: string): string => key.split("/").slice(2).join("/");

function fmtNumber(n: number): string {
  if (Number.isInteger(n) && Math.abs(n) < 1e6) return String(n);
  const a = Math.abs(n);
  if (a !== 0 && (a >= 1e6 || a < 1e-3)) return n.toExponential().replace(/\.?0+e/, "e").replace("e+", "e");
  return String(Number(n.toPrecision(6)));
}

interface Ctx {
  s: State;
  byKey: Map<string, DeclRecord>;
}

function refOf(c: Ctx, key: string): Ref {
  const rec = c.byKey.get(key);
  // The reader checks every $decl key against /opensees/decls; one that is
  // missing here is a broken invariant, not a case to draw around.
  if (!rec) throw new Error(`materials window: $decl ${key} is not an /opensees/decls key`);
  const label = `${rec.family} ${rec.name || keyHandle(rec.key)}`;
  const target = c.s.decls[key] ? key : null;
  return { key, label, target, why: target ? null : `${key} is not a declaration the app lists (the inspector shows materials, sections, transforms, integrations and elements)` };
}

function valOf(c: Ctx, v: DeclValue): Val {
  if (v === null) return { k: "null" };
  if (typeof v === "number") return { k: "text", text: fmtNumber(v) };
  if (typeof v === "string" || typeof v === "boolean") return { k: "text", text: String(v) };
  if (Array.isArray(v)) return { k: "list", items: v.slice(0, LIST_CAP).map((x) => valOf(c, x)), more: Math.max(0, v.length - LIST_CAP) };
  if (typeof v["$decl"] === "string") return { k: "ref", ref: refOf(c, v["$decl"]) };
  if (typeof v["$struct"] === "string") {
    const f = v["fields"];
    const fields = f && typeof f === "object" && !Array.isArray(f) ? Object.entries(f).map(([k, x]): [string, Val] => [k, valOf(c, x)]) : [];
    return { k: "struct", cls: v["$struct"], fields };
  }
  if (typeof v["$opaque"] === "string") return { k: "opaque", cls: v["$opaque"] };
  return { k: "object", fields: Object.entries(v).map(([k, x]): [string, Val] => [k, valOf(c, x)]) };
}

/** Every `$decl` key inside a value. */
function declKeys(v: DeclValue, out: Set<string>): void {
  if (v === null || typeof v !== "object") return;
  if (Array.isArray(v)) {
    for (const x of v) declKeys(x, out);
    return;
  }
  if (typeof v["$decl"] === "string") {
    out.add(v["$decl"]);
    return;
  }
  for (const x of Object.values(v)) declKeys(x, out);
}

// Reverse indexes, built once per loaded model (the frame re-renders on each
// window event; the decls and definitions objects change only on a re-read).
const chainUsers = new WeakMap<object, Map<DeclPath, { elements: number; objects: DeclPath[] }>>();
const paramUsers = new WeakMap<object, Map<string, DeclRecord[]>>();

function chainUsersOf(s: State): Map<DeclPath, { elements: number; objects: DeclPath[] }> {
  let m = chainUsers.get(s.decls);
  if (m) return m;
  m = new Map();
  for (const d of Object.values(s.decls)) {
    for (const to of new Set(Object.values(d.refs))) {
      const u = m.get(to) ?? m.set(to, { elements: 0, objects: [] }).get(to)!;
      if (d.kind === "element") u.elements++;
      else u.objects.push(d.path);
    }
  }
  chainUsers.set(s.decls, m);
  return m;
}

function paramUsersOf(defs: Definitions): Map<string, DeclRecord[]> {
  let m = paramUsers.get(defs);
  if (m) return m;
  m = new Map();
  for (const r of defs.decls?.records ?? []) {
    const keys = new Set<string>();
    for (const o of r.params) for (const v of Object.values(o.params)) declKeys(v, keys);
    for (const k of keys) (m.get(k) ?? m.set(k, []).get(k)!).push(r);
  }
  paramUsers.set(defs, m);
  return m;
}

interface TieUsers {
  /** uniaxial material path -> ties whose bond tag is its tag */
  byPath: Map<DeclPath, number>;
  /** the ties no uniaxial material can be named for, by reason; [] when there are none */
  unknown: { why: string; ties: number | null }[];
}
const tieUsers = new WeakMap<object, WeakMap<object, TieUsers>>();

/**
 * The embeddedRebar ties' bond tags matched to the uniaxial materials of this
 * file by tag (the tag is what the tie row carries). A tag that matches no
 * uniaxial material, or more than one, is unknown, as is every tie the
 * reader could not read.
 */
function tieUsersOf(s: State, defs: Definitions | null): TieUsers | null {
  const er = defs?.embeddedRebar;
  if (!defs || !er) return null;
  const inner = tieUsers.get(defs) ?? tieUsers.set(defs, new WeakMap()).get(defs)!;
  let t = inner.get(s.decls);
  if (t) return t;
  const byTag = new Map<number, DeclPath[]>();
  for (const d of Object.values(s.decls)) {
    if (d.kind === "uniaxialMaterial" && d.tag !== null) (byTag.get(d.tag) ?? byTag.set(d.tag, []).get(d.tag)!).push(d.path);
  }
  t = { byPath: new Map(), unknown: er.unresolved.map((u) => ({ why: u.why, ties: u.ties })) };
  for (const b of er.bond) {
    const hit = byTag.get(b.tag) ?? [];
    if (hit.length === 1) t.byPath.set(hit[0]!, (t.byPath.get(hit[0]!) ?? 0) + b.ties);
    else t.unknown.push({ why: hit.length ? `bond tag ${b.tag} names ${hit.length} uniaxial materials` : `bond tag ${b.tag} is no uniaxial material of this file`, ties: b.ties });
  }
  inner.set(s.decls, t);
  return t;
}

/** How many elements a declaration owns (its `element` tag runs). */
function elementsOwned(defs: Definitions, rec: DeclRecord): number {
  return (defs.decls?.tags ?? []).filter((t) => t.decl === rec.row && t.kind === "element").reduce((n, t) => n + t.count, 0);
}

function opsVersion(s: State): string {
  const z = s.artifacts.model?.zones["opensees"];
  return z?.status === "ready" ? z.version : "?";
}

function rowOf(c: Ctx, defs: Definitions | null, d: Decl): MaterialRow {
  const rec = defs ? declOfTag(defs, d.kind, d.tag) : null;
  let paramsNote: string | null = null;
  if (!defs?.decls) paramsNote = null; // the window note says it once
  else if (!rec) paramsNote = `no /opensees/decls tag run owns ${d.kind} tag ${d.tag ?? "?"}`;
  else if (defs.decls.hasParams && rec.params.length === 0) paramsNote = `/opensees/decl_params has no row for ${rec.key}`;
  const owners = rec ? rec.params.map((o) => ({ type: o.type, fields: Object.entries(o.params).map(([k, v]): [string, Val] => [k, valOf(c, v)]) })) : [];
  const chain = chainUsersOf(c.s).get(d.path);
  const objects = (chain?.objects ?? []).map((p): Ref => {
    const u = c.s.decls[p]!;
    return { key: p, label: `${u.kind} ${keyHandle(p) || u.name}`, target: p, why: null };
  });
  const tu = d.kind === "uniaxialMaterial" ? tieUsersOf(c.s, defs) : null;
  let tiesUnknown: MaterialRow["tiesUnknown"] = tu && tu.unknown.length ? tu.unknown : null;
  if (tu && d.tag === null && defs!.embeddedRebar!.bond.length) {
    tiesUnknown = [...(tiesUnknown ?? []), { why: "this material's tag is not in the file, so no tie's bond tag can be matched to it", ties: null }];
  }
  const referencedBy =
    defs && rec
      ? (paramUsersOf(defs).get(rec.key) ?? []).map((u) => ({ ref: refOf(c, u.key), type: u.params[0]?.type ?? u.family, elements: elementsOwned(defs, u) }))
      : [];
  return {
    path: d.path,
    kind: d.kind,
    type: d.type,
    handle: rec ? rec.name || keyHandle(rec.key) : d.name,
    tag: d.tag,
    declKey: rec?.key ?? null,
    h5: d.h5,
    owners,
    paramsNote,
    usedBy: { elements: chain?.elements ?? 0, objects, ties: tu?.byPath.get(d.path) ?? 0 },
    tiesUnknown,
    referencedBy,
  };
}

/** What the window lists, from the state alone. */
export function materialsOf(s: State): MaterialsData {
  const model = s.artifacts.model;
  if (!model || model.status !== "ready") return { groups: [], note: "no model is open" };
  const defs = s.definitions;
  const c: Ctx = { s, byKey: new Map((defs?.decls?.records ?? []).map((r) => [r.key, r])) };
  const decls = Object.values(s.decls).filter((d) => LISTED.has(d.kind));
  const groups = GROUPS.map((g) => ({
    kind: g.kind,
    title: g.title,
    rows: decls.filter((d) => d.kind === g.kind).map((d) => rowOf(c, defs, d)),
  })).filter((g) => g.rows.length > 0);
  if (groups.length === 0) return { groups, note: defs ? "the model declares no material or section" : "the model has no /opensees zone: it declares no material or section" };
  const v = opsVersion(s);
  const note =
    !defs?.decls ? `opensees ${v}: this file has no /opensees/decls or /opensees/decl_params (written before 2.25.0), so parameters by field name are not in this file; the inspector shows each object as read` :
    !defs.decls.hasParams ? `opensees ${v}: this file has no /opensees/decl_params (written before 2.26.0), so parameters by field name are not in this file; the inspector shows each object as read` :
    null;
  return { groups, note };
}

// ---- DOM --------------------------------------------------------------------

const select = (act: (e: Event) => void, decl: DeclPath): void => act({ type: "select", pick: { decl, at: null, nodes: null } });

function refEl(r: Ref, act: (e: Event) => void): HTMLElement {
  const a = el("span", "window-ref", r.label);
  a.dataset["decl"] = r.key;
  if (r.target) {
    const target = r.target;
    a.title = `select ${target}`;
    a.style.cssText = "color: var(--accent); text-decoration: underline; cursor: pointer;";
    a.addEventListener("click", (e) => {
      e.stopPropagation();
      select(act, target);
    });
  } else {
    a.title = r.why ?? "";
    a.style.cssText = "color: var(--muted); text-decoration: underline dotted;";
  }
  return a;
}

const muted = (text: string): HTMLElement => {
  const e = el("span", undefined, text);
  e.style.cssText = "color: var(--muted);";
  return e;
};

function valEl(v: Val, act: (e: Event) => void): HTMLElement {
  switch (v.k) {
    case "text":
      return el("span", undefined, v.text);
    case "null":
      return muted("none");
    case "ref":
      return refEl(v.ref, act);
    case "opaque": {
      const e = muted(v.cls);
      e.style.fontStyle = "italic";
      return e;
    }
    case "struct":
    case "object": {
      const e = el("span");
      if (v.k === "struct") e.append(el("span", undefined, v.cls));
      e.append(muted("("));
      v.fields.forEach(([k, x], i) => {
        if (i) e.append(muted(", "));
        e.append(muted(`${k}=`), valEl(x, act));
      });
      e.append(muted(")"));
      return e;
    }
    case "list": {
      // A list of structured items reads one per line; scalars run inline.
      const block = v.items.some((x) => x.k === "struct" || x.k === "object");
      const e = el(block ? "div" : "span");
      v.items.forEach((x, i) => {
        if (block) {
          const line = el("div");
          line.append(valEl(x, act));
          e.append(line);
        } else {
          if (i) e.append(muted(", "));
          e.append(valEl(x, act));
        }
      });
      if (v.items.length === 0) e.append(muted("[]"));
      if (v.more) e.append(muted(block ? `+${v.more} more` : `, +${v.more} more`));
      return e;
    }
    default: {
      const never: never = v;
      throw new Error(`materials window: unknown value ${JSON.stringify(never)}`);
    }
  }
}

function usedByEl(r: MaterialRow, act: (e: Event) => void): HTMLElement {
  const e = el("div", "window-row-detail");
  const n = r.usedBy.elements, objs = r.usedBy.objects, ties = r.usedBy.ties;
  const unknown = r.tiesUnknown;
  if (n === 0 && objs.length === 0 && ties === 0 && unknown) {
    // A tie whose bond material is unread may name this one: not "nothing".
    e.append(muted("users unknown: referenced by embeddedRebar ties"));
  } else if (n === 0 && objs.length === 0 && ties === 0) {
    // The two sources disagree when decl_params names a user the chain does
    // not follow (an element type the chain cannot decode): say so, never a
    // bare "nothing".
    e.append(
      muted(
        r.referencedBy.length
          ? "the definition chain follows no element or object to it, but decl_params names it (below): the two sources disagree"
          : "used by nothing in the definition chain",
      ),
    );
  } else {
    e.append(muted("used by "));
    const parts: (string | HTMLElement)[] = [];
    if (n) parts.push(`${n.toLocaleString()} element${n === 1 ? "" : "s"}`);
    if (ties) parts.push(`${ties.toLocaleString()} embeddedRebar tie${ties === 1 ? "" : "s"}`);
    for (const o of objs) parts.push(refEl(o, act));
    parts.forEach((p, i) => {
      if (i) e.append(muted(" · "));
      e.append(p);
    });
  }
  if (unknown) {
    const d = el("div");
    const head = n === 0 && objs.length === 0 && ties === 0 ? "" : "users unknown beyond these: referenced by embeddedRebar ties; ";
    const list = unknown.map((u) => `${u.ties === null ? "unknown number of" : u.ties.toLocaleString()} tie${u.ties === 1 ? "" : "s"}: ${u.why}`);
    d.append(muted(`${head}${list.join(" · ")}`));
    e.append(d);
  }
  if (r.referencedBy.length) {
    const d = el("div");
    d.append(muted(n === 0 && objs.length === 0 && ties === 0 ? "decl_params names it from " : "named in decl_params by "));
    r.referencedBy.forEach((u, i) => {
      if (i) d.append(muted(" · "));
      d.append(refEl(u.ref, act), muted(` ${u.type}` + (u.elements ? ` (${u.elements.toLocaleString()} element${u.elements === 1 ? "" : "s"})` : "")));
    });
    e.append(d);
  }
  return e;
}

function rowEl(r: MaterialRow, act: (e: Event) => void): HTMLElement {
  const row = el("div", "window-row clickable");
  row.dataset["decl"] = r.path;
  row.title = `select ${r.path}`;
  row.addEventListener("click", () => select(act, r.path));
  const head = el("div", "window-row-head");
  head.append(el("span", "window-row-title", r.handle), el("span", "node-type", r.type));
  if (r.tag !== null) head.append(el("span", "window-row-decl", `@${r.tag}`));
  if (r.declKey) head.append(el("span", "window-row-decl", r.declKey));
  row.append(head);
  for (const [i, o] of r.owners.entries()) {
    if (r.owners.length > 1 || o.type !== r.type) row.append(el("div", "window-row-detail", r.owners.length > 1 ? `owner ${i + 1} · ${o.type}` : o.type));
    if (o.fields.length === 0) {
      row.append(el("div", "window-row-detail", "no parameters"));
      continue;
    }
    const table = el("table", "fields");
    for (const [k, v] of o.fields) {
      const tr = el("tr");
      const td = el("td", "f-value");
      td.append(valEl(v, act));
      tr.append(el("td", "f-label", k), td);
      table.append(tr);
    }
    row.append(table);
  }
  if (r.paramsNote) row.append(el("div", "window-note", r.paramsNote));
  row.append(usedByEl(r, act), el("div", "f-src", r.h5));
  return row;
}

export const view: WindowView = (s, act) => {
  const data = materialsOf(s);
  const body = el("div", "window-list");
  if (data.note) body.append(el("div", "window-note", data.note));
  for (const g of data.groups) {
    body.append(el("div", "window-section", `${g.title} (${g.rows.length})`));
    for (const r of g.rows) body.append(rowEl(r, act));
  }
  return body;
};
