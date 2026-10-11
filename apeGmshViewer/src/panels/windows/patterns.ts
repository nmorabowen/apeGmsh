// The patterns and time series window (V3c2, #1643; ADR 0112 D6, D7).
//
// Every load pattern by name, under the stage that adds it (or `global`):
// its type, its time series (the `series` field's `{$decl}` ref), its other
// parameters by field name from `/opensees/decl_params`, the archive tables
// it carries (loads, sps, element loads, held DOFs) with their row counts,
// and the targets the archive names. Then every time series with its
// parameters and a small plot of its samples (plot/series.ts).
//
// `patternsOf` is the pure read of the state (selectors only); `view` builds
// the DOM from it. A click on a referenced declaration dispatches
// `requestSource` (go-to-source), the only store action a series or pattern
// supports: neither is a declaration of the store, so it cannot be selected.
// Tags are evidence, names are handles (D2c): a row is titled by its name, or
// by its `#k` key when unnamed; the tag is shown beside the type.

import { declOfTag, shortValue, sourceFor } from "../../state/selectors.ts";
import type { Event, State } from "../../state/store.ts";
import type { DeclRecord, DeclValue, Definitions, DefObject, StageDef } from "../../state/types.ts";
import { el } from "../../ui/dom.ts";
import { archivedSamples, sampleSeries, seriesSvg, type SeriesPlot } from "./plot/series.ts";
import type { WindowView } from "./view.ts";

/** A referenced declaration: its key, and whether go-to-source can reach it (else why not). */
export interface RefLink {
  key: string;
  /** the declaration's handle: its name, or the key after `opensees/<family>/` */
  label: string;
  /** `file:line` of its source, or null when go-to-source cannot jump */
  source: string | null;
  why: string | null;
}

/** One parameter of a declaration, by field name. */
export interface ParamLine {
  field: string;
  text: string;
  /** set when the value is a `{$decl}` ref */
  ref: RefLink | null;
}

export interface PatternCard {
  h5: string;
  /** the stage that adds the pattern (`stage 0 · gravity`), or `global` */
  stage: string;
  title: string;
  /** the `/opensees/decls` key, or null when the file has no decls row for it */
  self: RefLink | null;
  synth: boolean;
  type: string;
  tag: number | null;
  /** the attributes that say what the pattern is for (`role`) */
  role: string | null;
  series: RefLink | null;
  /** the series as the archive attribute names it, when no decl_params ref does */
  seriesH5: string | null;
  /** the `series` field as written when it is not a `{$decl}` ref (an inline value, null), shown verbatim */
  seriesValue: string | null;
  params: ParamLine[];
  /** each archive table with its row count (`loads: 258 rows`) */
  tables: string[];
  /** the targets the archive names (`pg Col`), else the node / element summary */
  targets: string[];
  /** why the parameters are not listed, or null */
  paramsNote: string | null;
}

export interface SeriesCard {
  h5: string;
  title: string;
  self: RefLink | null;
  synth: boolean;
  type: string;
  tag: number | null;
  params: ParamLine[];
  paramsNote: string | null;
  plot: SeriesPlot;
  /** the patterns that drive this series, with their stage */
  usedBy: string[];
}

export interface PatternsModel {
  /** a sentence when there is nothing (or something missing) to say why */
  note: string | null;
  sections: { name: string; patterns: PatternCard[] }[];
  series: SeriesCard[];
}

/** The handle of a declaration key: `opensees/pattern/#1` -> `#1`, `opensees/timeSeries/support:gravity/hold` -> `support:gravity/hold`. */
export const handleOf = (key: string): string => key.split("/").slice(2).join("/") || key;

function refOf(s: State, defs: Definitions, key: string): RefLink {
  const rec = defs.decls?.records.find((r) => r.key === key);
  const where = sourceFor(s, key);
  // A series or pattern is not a declaration of the store: when /provenance has no record for it, say that.
  const why = where.ok ? null :
    !(key in s.decls) && !s.provenance.some((p) => p.key === key) ? `no /provenance record for ${key}` + (s.provenance.length ? "" : " (the file records no sources)") :
    where.reason;
  return { key, label: rec?.name || handleOf(key), source: where.ok ? where.label : null, why };
}

const isRef = (v: DeclValue): v is { $decl: string } => typeof v === "object" && v !== null && !Array.isArray(v) && typeof v["$decl"] === "string";
/** `_loads_`, `_sps_`, ...: the record lists a pattern carries; listed as targets and counts, not as parameters. */
const isRecordField = (f: string): boolean => f.startsWith("_") && f.endsWith("_");

function paramLines(s: State, defs: Definitions, p: Record<string, DeclValue>, skip: ReadonlySet<string>): ParamLine[] {
  return Object.entries(p)
    .filter(([f]) => !skip.has(f) && !isRecordField(f))
    .map(([field, v]) => (isRef(v) ? { field, text: v.$decl, ref: refOf(s, defs, v.$decl) } : { field, text: shortValue(v), ref: null }));
}

const NO_PARAMS = (d: DeclRecord | null, defs: Definitions): string =>
  !defs.decls ? "parameters by field name: not in this file (no /opensees/decls, opensees < 2.25)" :
  !d ? "parameters by field name: no /opensees/decls row owns this tag" :
  "parameters by field name: not in this file (no /opensees/decl_params, opensees < 2.26)";

/** `pg Col`, `node 8`: a record's target, when it has one. */
function targetOf(kind: unknown, target: unknown): { named: boolean; text: string } | null {
  if (kind === undefined || target === undefined) return null;
  const k = String(kind), t = String(target);
  return { named: k !== "node" && k !== "element", text: `${k} ${t}` };
}

function targetsOf(p: DefObject, rec: Record<string, DeclValue> | null): string[] {
  const named = new Set<string>(), bare = new Set<string>();
  // The declaration's record lists (`_loads_`, `_sps_`): the targets as the user named them.
  for (const [f, v] of Object.entries(rec ?? {})) {
    if (!isRecordField(f) || !Array.isArray(v)) continue;
    for (const r of v) {
      const fields = typeof r === "object" && r !== null && !Array.isArray(r) ? r["fields"] : undefined;
      if (typeof fields !== "object" || fields === null || Array.isArray(fields)) continue;
      const t = targetOf(fields["target_kind"], fields["target"]);
      if (t) (t.named ? named : bare).add(t.text);
    }
  }
  // The archive tables: a row whose target is a group keeps its name; node / element rows are counted by kind.
  const rows = new Map<string, number>();
  for (const table of Object.values(p.tables)) {
    const k = table.columns.indexOf("target_kind"), g = table.columns.indexOf("target");
    if (k < 0 || g < 0) continue;
    for (const row of table.rows) {
      const t = targetOf(row[k], row[g]);
      if (!t) continue;
      if (t.named) named.add(t.text);
      else rows.set(String(row[k]), (rows.get(String(row[k])) ?? 0) + 1);
    }
  }
  if (named.size) return [...named];
  if (bare.size) return bare.size <= 4 ? [...bare] : [`${bare.size} nodes or elements (no group name in the archive)`];
  return [...rows].map(([kind, n]) => `${n} ${kind} row${n === 1 ? "" : "s"} (no group name in the archive)`);
}

/** What a pattern's archive table holds (h5-schema.md, `/opensees/patterns/<P>`); another table is listed by its name as `rows`. */
const TABLES: Readonly<Record<string, string>> = {
  loads: "nodal loads",
  sps: "single-point constraints",
  element_loads: "element loads",
  sp_holds: "held DOFs",
};

function patternCard(s: State, defs: Definitions, p: DefObject, stage: string): PatternCard {
  const d = declOfTag(defs, "pattern", p.tag);
  const rec = d?.params[0] ?? null;
  const seriesV = rec?.params["series"];
  const role = p.attrs["role"];
  const seriesRef = p.attrs["series_ref"];
  return {
    h5: p.path,
    stage,
    title: d ? d.name || handleOf(d.key) : p.groupName,
    self: d ? refOf(s, defs, d.key) : null,
    synth: d?.synth ?? false,
    type: rec?.type ?? p.type,
    tag: p.tag,
    role: typeof role === "string" ? role : null,
    series: seriesV !== undefined && isRef(seriesV) ? refOf(s, defs, seriesV.$decl) : null,
    seriesH5: seriesV === undefined && typeof seriesRef === "string" ? seriesRef : null,
    seriesValue: seriesV !== undefined && !isRef(seriesV) ? shortValue(seriesV) : null,
    params: rec ? paramLines(s, defs, rec.params, new Set(["series"])) : [],
    tables: Object.entries(p.tables).map(([name, t]) => {
      const n = t.rows.length, what = TABLES[name] ?? "rows";
      return `${n} ${n === 1 ? what.replace(/s$/, "") : what} (${name})`;
    }),
    targets: targetsOf(p, rec?.params ?? null),
    paramsNote: rec ? null : NO_PARAMS(d, defs),
  };
}

const stageName = (st: StageDef): string => `stage ${st.index} · ${st.name}`;

/** What the patterns window shows, from the store alone. */
export function patternsOf(s: State): PatternsModel {
  const model = s.artifacts.model;
  if (!model || model.status !== "ready") return { note: "no model is open", sections: [], series: [] };
  const defs = s.definitions;
  if (!defs) return { note: "the model has no /opensees zone", sections: [], series: [] };
  const sections: PatternsModel["sections"] = [];
  if (defs.patterns.length) sections.push({ name: "global", patterns: defs.patterns.map((p) => patternCard(s, defs, p, "global")) });
  for (const st of defs.stages) {
    if (st.patterns.length) sections.push({ name: stageName(st), patterns: st.patterns.map((p) => patternCard(s, defs, p, stageName(st))) });
  }
  const patterns = sections.flatMap((x) => x.patterns);
  const series = defs.timeSeries.map((t): SeriesCard => {
    const d = declOfTag(defs, "timeSeries", t.tag);
    const rec = d?.params[0] ?? null;
    const usedBy = patterns
      .filter((p) => (d && p.series?.key === d.key) || p.seriesH5 === t.path)
      .map((p) => `${p.title} (${p.stage})`);
    return {
      h5: t.path,
      title: d ? d.name || handleOf(d.key) : t.groupName,
      self: d ? refOf(s, defs, d.key) : null,
      synth: d?.synth ?? false,
      type: rec?.type ?? t.type,
      tag: t.tag,
      params: rec ? paramLines(s, defs, rec.params, new Set()) : [],
      paramsNote: rec ? null : NO_PARAMS(d, defs),
      plot: archivedSamples(t.tables["time"], t.tables["values"]) ??
        (rec ? sampleSeries(rec.type, rec.params) : { ok: false, reason: "no plot: the series' parameters by field name are not in this file" }),
      usedBy,
    };
  });
  const note =
    !patterns.length && !series.length ? "the model declares no load pattern or time series" :
    !patterns.length ? "the model declares no load pattern" :
    !series.length ? "the model declares no time series" : null;
  return { note, sections, series };
}

// ---- DOM ---------------------------------------------------------------------

const LINK = "cursor: pointer; color: var(--accent); text-decoration: underline;";

/** A ref as text; clickable (go-to-source) when its source is recorded, else muted with the reason as its tooltip. */
function refEl(r: RefLink, act: (e: Event) => void, text: string = r.label): HTMLElement {
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

function head(title: string, self: RefLink | null, synth: boolean, act: (e: Event) => void): HTMLElement {
  const h = el("div", "window-row-head");
  h.append(el("span", "window-row-title", title));
  if (synth) h.append(el("span", "window-row-decl", "synthesised"));
  if (self) {
    h.append(el("span", "window-row-decl", self.key));
    if (self.source) h.append(refEl(self, act, `↗ ${self.source}`));
  }
  return h;
}

function detail(parts: (string | Node)[]): HTMLElement {
  const d = el("div", "window-row-detail");
  parts.forEach((p, i) => {
    if (i) d.append(" · ");
    d.append(p);
  });
  return d;
}

function paramsEl(lines: readonly ParamLine[], note: string | null, act: (e: Event) => void): HTMLElement | null {
  if (note) return el("div", "window-note", note);
  if (!lines.length) return null;
  return detail(
    lines.map((l) => {
      const span = el("span");
      span.append(`${l.field}=`, l.ref ? refEl(l.ref, act) : l.text);
      return span;
    }),
  );
}

function patternEl(p: PatternCard, act: (e: Event) => void): HTMLElement {
  const row = el("div", "window-row");
  row.dataset["h5"] = p.h5;
  row.append(head(p.title, p.self, p.synth, act));
  const series = el("span");
  series.append("series ", p.series ? refEl(p.series, act) : (p.seriesValue ?? p.seriesH5 ?? "none"));
  row.append(detail([`${p.type} · tag ${p.tag ?? "?"}`, ...(p.role ? [`role ${p.role}`] : []), series]));
  const params = paramsEl(p.params, p.paramsNote, act);
  if (params) row.append(params);
  row.append(detail([p.tables.length ? p.tables.join(" · ") : "no load or constraint table"]));
  if (p.targets.length) row.append(detail([`targets ${p.targets.join(", ")}`]));
  row.append(el("div", "f-src", p.h5));
  return row;
}

function seriesEl(t: SeriesCard, act: (e: Event) => void): HTMLElement {
  const row = el("div", "window-row");
  row.dataset["h5"] = t.h5;
  row.append(head(t.title, t.self, t.synth, act));
  row.append(detail([`${t.type} · tag ${t.tag ?? "?"}`, t.usedBy.length ? `drives ${t.usedBy.join(", ")}` : "drives no pattern"]));
  const params = paramsEl(t.params, t.paramsNote, act);
  if (params) row.append(params);
  if (t.plot.ok) {
    // The plot, its caption beside it (style.css is not this window's file, so the layout is inline).
    const fig = el("div", "series-figure");
    fig.setAttribute("style", "display: flex; align-items: center; gap: 10px; margin: 2px 0;");
    const caption = el("div", "window-note", t.plot.samples.caption);
    caption.setAttribute("style", "margin: 0;");
    fig.append(seriesSvg(t.plot.samples), caption);
    row.append(fig);
  } else {
    row.append(el("div", "window-note", t.plot.reason));
  }
  row.append(el("div", "f-src", t.h5));
  return row;
}

export const view: WindowView = (s, act) => {
  const m = patternsOf(s);
  const body = el("div", "window-list");
  if (m.note) body.append(el("div", "window-note", m.note));
  for (const sec of m.sections) {
    body.append(el("div", "window-section", `patterns · ${sec.name}`));
    for (const p of sec.patterns) body.append(patternEl(p, act));
  }
  if (m.series.length) {
    body.append(el("div", "window-section", "time series"));
    for (const t of m.series) body.append(seriesEl(t, act));
  }
  return body;
};
