// The analysis window (V3c1, #1642; ADR 0112 D6, D7).
//
// For each stage in order (or the one implicit stage of an unstaged deck,
// `/opensees/analysis`): system, numberer, constraints handler, integrator,
// algorithm, test and analysis, each as its token and its parameters by
// field name from `/opensees/decl_params` (`LoadControl{dlam}`,
// `NormDispIncr{tol, max_iter}`), never from the positional `*_args`. Then
// the stage's steps and dt from `/sequence`; a deck whose analysis loop is
// user Tcl (`steps = -1`) shows no steps and says why.
//
// The store has no stage selection, so every stage's chain is listed, in
// stage order. The join of a chain to its declarations: the archive's
// `stage_NNN/analysis` carries no declaration link (#1646), so the k-th
// declaration of a chain family (`integrator`, `test`, ...) is the k-th chain
// that sets that token, in file order. The join is used only when the counts
// agree and every declaration's type is the token the chain names; otherwise
// the window says it cannot join, and lists no parameters.
//
// `analysisOf` is the pure read of the state; `view` builds the DOM. A click
// on a declaration (or on a `{$decl}` ref inside its parameters) dispatches
// `requestSource` and nothing else. Tags are evidence, names are handles: an
// unnamed declaration shows its `#k` key.

import { shortValue, sourceFor } from "../../state/selectors.ts";
import type { Event, State } from "../../state/store.ts";
import type { AnalysisDef, DeclRecord, DeclValue, Definitions, Table } from "../../state/types.ts";
import { el } from "../../ui/dom.ts";
import { dtText, handleOf, stepsText } from "./stages.ts";
import type { WindowView } from "./view.ts";

/** A declaration, and whether go-to-source can reach it (else why not). */
export interface DeclLink {
  key: string;
  label: string;
  source: string | null;
  why: string | null;
}

export interface ParamLine {
  field: string;
  text: string;
  ref: DeclLink | null;
}

/** One member of the chain: its role, token, declaration and parameters by field name. */
export interface ChainPart {
  /** `system`, `numberer`, `constraints`, `integrator`, `algorithm`, `test`, `analysis` */
  role: string;
  /** the archive attribute it is read from (`handler` for the constraints handler) */
  attr: string;
  /** the token (`LoadControl`), or null when the chain does not set it */
  token: string | null;
  decl: DeclLink | null;
  params: ParamLine[];
  /** why no parameter is listed, or null */
  note: string | null;
}

export interface ChainCard {
  /** `stage 0 · gravity`, or `implicit stage (unstaged deck)` */
  title: string;
  h5: string;
  parts: ChainPart[];
  /** steps and dt in words, from /sequence (or the stage's attributes when the file has none) */
  steps: string | null;
  dt: string | null;
}

export interface AnalysisModel {
  note: string | null;
  chains: ChainCard[];
}

/** The chain in the order the card names it: role, archive attribute, `/opensees/decls` family. */
export const CHAIN: readonly { role: string; attr: string; family: string }[] = [
  { role: "system", attr: "system", family: "system" },
  { role: "numberer", attr: "numberer", family: "numberer" },
  { role: "constraints", attr: "handler", family: "constraints" },
  { role: "integrator", attr: "integrator", family: "integrator" },
  { role: "algorithm", attr: "algorithm", family: "algorithm" },
  { role: "test", attr: "test", family: "test" },
  { role: "analysis", attr: "analysis", family: "analysis" },
];

const isRef = (v: DeclValue): v is { $decl: string } => typeof v === "object" && v !== null && !Array.isArray(v) && typeof v["$decl"] === "string";

function linkOf(s: State, defs: Definitions, key: string): DeclLink {
  const rec = defs.decls?.records.find((r) => r.key === key);
  const where = sourceFor(s, key);
  const why = where.ok ? null :
    !(key in s.decls) && !s.provenance.some((p) => p.key === key) ? `no /provenance record for ${key}` + (s.provenance.length ? "" : " (the file records no sources)") :
    where.reason;
  return { key, label: rec?.name || handleOf(key), source: where.ok ? where.label : null, why };
}

/**
 * The declaration of each chain, per family: the k-th declaration of the
 * family to the k-th chain that sets its token, when the counts agree and
 * every type matches; else the reason the join is refused.
 */
function joinChains(defs: Definitions, chains: readonly AnalysisDef[]): Map<string, (DeclRecord | null)[] | string> {
  const out = new Map<string, (DeclRecord | null)[] | string>();
  const records = defs.decls?.records ?? [];
  for (const { attr, family } of CHAIN) {
    const recs = records.filter((r) => r.family === family);
    const setting = chains.map((c, i) => ({ i, token: c.attrs[attr] })).filter((x) => typeof x.token === "string");
    if (recs.length !== setting.length) {
      out.set(family, `cannot join: /opensees/decls has ${recs.length} ${family} declaration${recs.length === 1 ? "" : "s"} for ${setting.length} chain${setting.length === 1 ? "" : "s"} that set it (the archive records no chain↔declaration link, #1646)`);
      continue;
    }
    const bad = setting.findIndex((x, k) => recs[k]!.params[0] !== undefined && recs[k]!.params[0]!.type !== x.token);
    if (bad >= 0) {
      out.set(family, `cannot join: ${recs[bad]!.key} is a ${recs[bad]!.params[0]!.type}, the chain names ${String(setting[bad]!.token)} (the archive records no chain↔declaration link, #1646)`);
      continue;
    }
    const per: (DeclRecord | null)[] = chains.map(() => null);
    setting.forEach((x, k) => (per[x.i] = recs[k]!));
    out.set(family, per);
  }
  return out;
}

const STAGE_COLUMNS = ["source", "steps", "dt", "analysis"] as const;

/** Steps and dt per stage source, from /sequence/stages, or null when the zone is not ready or lacks a column. */
function sequenceSteps(s: State): Map<string, { steps: string; dt: string }> | null {
  const seq = s.sequence;
  if (seq.status !== "ready") return null;
  const t: Table = seq.stages;
  const at = STAGE_COLUMNS.map((c) => t.columns.indexOf(c));
  if (at.some((i) => i < 0)) return null;
  const num = (v: unknown): number => (typeof v === "number" ? v : v === null || v === undefined ? Number.NaN : Number(v));
  const out = new Map<string, { steps: string; dt: string }>();
  for (const r of t.rows) {
    const analysis = typeof r[at[3]!] === "string" ? (r[at[3]!] as string) : "";
    out.set(String(r[at[0]!]), { steps: stepsText(num(r[at[1]!])), dt: dtText(num(r[at[2]!]), analysis) });
  }
  return out;
}

/** What the analysis window shows, from the store alone. */
export function analysisOf(s: State): AnalysisModel {
  const model = s.artifacts.model;
  if (!model || model.status !== "ready") return { note: "no model is open", chains: [] };
  const defs = s.definitions;
  if (!defs) return { note: "the model has no /opensees zone", chains: [] };

  const list: { title: string; a: AnalysisDef; attrs: Record<string, unknown> | null }[] = [];
  if (defs.analysis) list.push({ title: defs.stages.length ? "global" : "implicit stage (unstaged deck)", a: defs.analysis, attrs: defs.analysis.attrs });
  for (const st of defs.stages) if (st.analysis) list.push({ title: `stage ${st.index} · ${st.name}`, a: st.analysis, attrs: st.attrs });
  if (!list.length) return { note: "the model declares no analysis", chains: [] };

  const joins = joinChains(defs, list.map((x) => x.a));
  const fromSeq = sequenceSteps(s);
  const noParams =
    !defs.decls ? "parameters by field name: not in this file (no /opensees/decls, opensees < 2.25)" :
    !defs.decls.hasParams ? "parameters by field name: not in this file (no /opensees/decl_params, opensees < 2.26)" : null;

  const chains = list.map(({ title, a, attrs }, i): ChainCard => {
    const parts = CHAIN.map(({ role, attr, family }): ChainPart => {
      const tok = a.attrs[attr];
      const token = typeof tok === "string" ? tok : null;
      if (token === null) return { role, attr, token, decl: null, params: [], note: `not set in ${a.path}` };
      if (noParams && !defs.decls) return { role, attr, token, decl: null, params: [], note: noParams };
      const join = joins.get(family)!;
      if (typeof join === "string") return { role, attr, token, decl: null, params: [], note: join };
      const rec = join[i] ?? null;
      if (!rec) return { role, attr, token, decl: null, params: [], note: `no /opensees/decls ${family} declaration` };
      const p = rec.params[0];
      if (!p) return { role, attr, token, decl: linkOf(s, defs, rec.key), params: [], note: noParams ?? `${rec.key} has no /opensees/decl_params row` };
      const params = Object.entries(p.params).map(([field, v]): ParamLine =>
        isRef(v) ? { field, text: v.$decl, ref: linkOf(s, defs, v.$decl) } : { field, text: shortValue(v), ref: null },
      );
      return { role, attr, token, decl: linkOf(s, defs, rec.key), params, note: null };
    });
    // Steps and dt: /sequence when the file has it; else the stage's own attributes, said as such.
    const sq = fromSeq?.get(a.path === "/opensees/analysis" ? a.path : a.path.replace(/\/analysis$/, ""));
    let steps: string | null = sq?.steps ?? null, dt: string | null = sq?.dt ?? null;
    if (!sq && attrs) {
      const n = attrs["analyze_steps"], d = attrs["analyze_dt"];
      steps = typeof n === "number" ? stepsText(n) : "no analyze_steps archived: the deck runs its own analysis loop, so no step count is known";
      dt = dtText(typeof d === "number" ? d : Number.NaN, typeof a.attrs["analysis"] === "string" ? (a.attrs["analysis"] as string) : "");
    }
    return { title, h5: a.path, parts, steps, dt };
  });
  return { note: null, chains };
}

// ---- DOM ---------------------------------------------------------------------

const LINK = "cursor: pointer; color: var(--accent); text-decoration: underline;";

function refEl(r: DeclLink, act: (e: Event) => void, text: string = r.label): HTMLElement {
  const e = el("span", "window-ref", text);
  if (r.source) {
    e.setAttribute("style", LINK);
    e.title = `go to source: ${r.key} (${r.source})`;
    e.addEventListener("click", (ev) => {
      ev.stopPropagation();
      act({ type: "requestSource", decl: r.key });
    });
  } else {
    e.title = `${r.key}: ${r.why ?? "no source"}`;
  }
  return e;
}

function partEl(p: ChainPart, act: (e: Event) => void): HTMLElement {
  const row = el("div", "window-row-detail");
  row.dataset["role"] = p.role;
  const role = el("span", "chain-role", `${p.role} `);
  role.setAttribute("style", "color: var(--muted);");
  row.append(role);
  if (p.token === null) {
    row.append(el("span", "window-note", p.note ?? "not set"));
    return row;
  }
  // `LoadControl{dlam=0.5, num_iter=null}`: the token, then its fields by name.
  const tok = el("span", "window-row-title", p.token);
  row.append(tok);
  if (p.params.length || !p.note) {
    row.append("{");
    p.params.forEach((l, i) => {
      if (i) row.append(", ");
      row.append(`${l.field}=`, l.ref ? refEl(l.ref, act) : l.text);
    });
    row.append("}");
  }
  if (p.decl) row.append(" ", refEl(p.decl, act, p.decl.source ? `${p.decl.label} ↗` : p.decl.label));
  if (p.note) {
    const n = el("div", "window-note", p.note);
    n.setAttribute("style", "margin: 0 0 2px 12px;");
    row.append(n);
  }
  return row;
}

export const view: WindowView = (s, act) => {
  const m = analysisOf(s);
  const body = el("div", "window-list");
  if (m.note) body.append(el("div", "window-note", m.note));
  for (const c of m.chains) {
    body.append(el("div", "window-section", c.title));
    const card = el("div", "window-row");
    card.dataset["h5"] = c.h5;
    for (const p of c.parts) card.append(partEl(p, act));
    if (c.steps) card.append(el("div", "window-row-detail", c.steps));
    if (c.dt) card.append(el("div", "window-row-detail", c.dt));
    card.append(el("div", "f-src", c.h5));
    body.append(card);
  }
  return body;
};
