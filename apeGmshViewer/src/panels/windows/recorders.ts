// The recorders window (V3c3, ADR 0112 D6, D7): every recorder of the file,
// global and per stage, by its handle (the user's name, or the `#k` of its
// /opensees/decls key when unnamed; D2c: tags are evidence, names are
// handles), with its type, what it records, its targets, its output, its
// parameters by field name from /opensees/decl_params, and the stage that
// adds it (and removes it, when the file records a removal).
//
// `recordersOf` is the window's content as plain data, from the store and the
// selectors alone; `view` renders it. A row whose recorder targets a group the
// store declares selects that group on click (`select`); every other row is
// not clickable. Go to source is the store's `requestSource`, offered when the
// recorder's key reaches a /provenance frame (selectors.sourceFor says why
// not otherwise). Parameters are never re-derived from the positional
// `params` / `params_str` argv: a file without decl_params says so.
//
// `label` (opensees 2.28.0, ADR 0115 amendment R2-f) is not an OpenSees
// parameter: it is the names a `label=` declaration was given, kept as
// provenance beside the ids they resolved to. The window shows it on the
// targets line (`label left`), never among the parameters or the unset
// fields, and a null label (a `pg=` or explicit-id declaration) shows nothing.
// It is never re-resolved, so it selects nothing.

import type { Event, State } from "../../state/store.ts";
import { recorderDecls, shortValue, sourceFor } from "../../state/selectors.ts";
import type { DeclPath, DeclRecord, DeclValue, DefObject, StageDef } from "../../state/types.ts";
import { el } from "../../ui/dom.ts";
import type { WindowView } from "./view.ts";

/** One parameter of a recorder, by field name. */
export interface RecorderParam {
  field: string;
  value: string;
}

/** One recorder as the window shows it. */
export interface RecorderEntry {
  /** the recorder's HDF5 group */
  h5: string;
  /** `global`, or `stage <NNN> · <name>` */
  section: string;
  /** the user's name; the `#k` of its decls key when unnamed; the HDF5 group name when the file joins no decl */
  handle: string;
  /** the `/opensees/decls` key, or null when the recorder is not joined to one */
  declKey: string | null;
  /** the archive's `type`, and the decl_params class when it differs (`ladruno (Ladruno)`) */
  type: string;
  /** what it records: responses and dofs, by field name */
  records: string;
  /** what it records them on: node and element counts, groups by name, regions */
  targets: string;
  /** the output file and the format fields */
  output: string;
  /** the stage that adds it, and what the file says about its removal */
  lifetime: string;
  /** the non-empty parameters by field name; [] when there are none to show */
  params: RecorderParam[];
  /** the fields that are null or empty */
  unset: string[];
  /** why the parameters are missing (no decl_params, no decl join); null when they are shown */
  paramsNote: string | null;
  /** the group a click selects, or null */
  select: DeclPath | null;
  /** why a click selects nothing, when the recorder names a group the store lacks; else null */
  selectNote: string | null;
  /** go to source: the decl key, the button label and its tooltip; `off` when it cannot jump */
  source: { decl: string; label: string; title: string } | { off: string } | null;
}

export interface RecordersContent {
  entries: RecorderEntry[];
  /** a sentence over the list: why it is empty, or what it is read from */
  note: string;
}

const isEmpty = (v: DeclValue | undefined): boolean => v === undefined || v === null || (Array.isArray(v) && v.length === 0);
const list = (v: DeclValue): string => (Array.isArray(v) ? v.map((x) => shortValue(x)).join(", ") : shortValue(v));
const plural = (n: number, one: string): string => `${n} ${one}${n === 1 ? "" : "s"}`;

/** The response fields: `response`, `responses`, `*_responses`, and `dofs`. */
function recordsOf(p: Record<string, DeclValue>): string {
  const parts: string[] = [];
  for (const [k, v] of Object.entries(p)) {
    if (isEmpty(v)) continue;
    if (k === "response" || k === "responses") parts.push(list(v));
    else if (k.endsWith("_responses")) parts.push(`${k.slice(0, -"_responses".length)}: ${list(v)}`);
  }
  if (!isEmpty(p["dofs"])) parts.push(`dofs ${list(p["dofs"]!)}`);
  return parts.join(" · ");
}

/** The 2.28.0 provenance field: the names a `label=` declaration was given; not a parameter. */
const LABEL_FIELD = "label";

/** A field that names a group: `pg`, or `<what>_pg`. */
const isGroupField = (k: string): boolean => k === "pg" || k.endsWith("_pg");

/** The target fields: `nodes` / `elements` lists as counts, `*pg` as group names, regions as given, and the declaration's label. */
function targetsOf(p: Record<string, DeclValue>): { text: string; groups: string[] } {
  const parts: string[] = [];
  const groups: string[] = [];
  for (const [k, v] of Object.entries(p)) {
    if (isEmpty(v) || typeof v === "boolean") continue;
    if (k === "nodes" || k === "elements") {
      parts.push(Array.isArray(v) ? plural(v.length, k.slice(0, -1)) : `${k} ${shortValue(v)}`);
    } else if (isGroupField(k)) {
      const names = (Array.isArray(v) ? v : [v]).filter((x): x is string => typeof x === "string");
      groups.push(...names);
      const what = k === "pg" ? "" : `${k.slice(0, -"_pg".length)} of `;
      parts.push(`${what}group ${names.length ? names.join(", ") : shortValue(v)}`);
    } else if (/region/.test(k)) {
      parts.push(`${k} ${list(v)}`);
    } else if (k === LABEL_FIELD) {
      parts.push(`label ${list(v)}`);
    }
  }
  return { text: parts.length ? parts.join(" · ") : "no target field is set (nodes, elements, pg, region)", groups };
}

/** The output: the file, and every field whose name says format. */
function outputOf(r: DefObject, p: Record<string, DeclValue> | null): string {
  const file = p && typeof p["file"] === "string" ? p["file"] : r.attrs["file"];
  const parts = [typeof file === "string" && file ? `file ${file}` : "no file field"];
  if (p) for (const [k, v] of Object.entries(p)) if (/format/.test(k) && !isEmpty(v)) parts.push(`${k} ${shortValue(v)}`);
  const kind = r.attrs["kind"];
  if (typeof kind === "string") parts.push(`archive kind ${kind}`);
  return parts.join(" · ");
}

/** The `#k` (or name) at the end of a decls key. */
const keyHandle = (key: string): string => key.slice(key.lastIndexOf("/") + 1);

/** The group a click selects: an element-side physical group, else a label, of that name. */
function groupDecl(s: State, name: string): DeclPath | null {
  const paths = s.names[name] ?? [];
  return paths.find((p) => s.decls[p]?.kind === "physical_group") ?? paths.find((p) => s.decls[p]?.kind === "label") ?? null;
}

/** What /sequence says about the removal of recorder `handle`. */
function removalOf(s: State, handle: string | null): string {
  const seq = s.sequence;
  if (seq.status === "absent") return "removal not in this file (no /sequence zone)";
  if (seq.status === "refused") return `removal: unknown (/sequence ${seq.version} was not read: ${seq.reason})`;
  const col = (c: string) => seq.changes.columns.indexOf(c);
  const [iStage, iOp, iWhat, iName] = [col("stage"), col("op"), col("what"), col("name")];
  if ([iStage, iOp, iWhat, iName].some((i) => i < 0)) return "removal: unknown (/sequence/changes lacks a stage, op, what or name column)";
  if (handle === null) return "removal: unknown (/sequence names recorders by name; this one is not joined to a declaration)";
  // /sequence writes "" for every unnamed recorder, so an empty name cannot say which `#k` it removes.
  if (handle.startsWith("#")) return "removal: unknown (unnamed; /sequence matches by name)";
  const removed = seq.changes.rows.filter((r) => r[iWhat] === "recorder" && r[iOp] === "remove" && r[iName] === handle);
  return removed.length ? `removed by stage ${removed.map((r) => String(r[iStage])).join(", ")}` : "never removed (/sequence)";
}

function entryOf(s: State, r: DefObject, d: DeclRecord | undefined, st: StageDef | null, staged: boolean, why: string): RecorderEntry {
  const owner = d?.params[0];
  const p = owner?.params ?? null;
  const handle = d ? d.name || keyHandle(d.key) : r.groupName;
  const t = p ? targetsOf(p) : { text: "not in this file (no parameters by field name)", groups: [] };
  const sel = t.groups.map((g) => ({ g, decl: groupDecl(s, g) }));
  const select = sel.find((x) => x.decl !== null)?.decl ?? null;
  const missing = sel.filter((x) => x.decl === null).map((x) => x.g);
  const added = st ? `added by stage ${st.index} · ${st.name}` : staged ? "global, added before the first stage: every stage records it" : "global, unstaged model: the whole analysis records it";
  const params: RecorderParam[] = [];
  const unset: string[] = [];
  for (const o of d?.params ?? []) {
    for (const [k, v] of Object.entries(o.params)) {
      if (k === LABEL_FIELD) continue; // provenance, on the targets line
      if (isEmpty(v)) unset.push(k);
      else params.push({ field: (d!.params.length > 1 ? `#${o.row} ` : "") + k, value: shortValue(v) });
    }
  }
  let source: RecorderEntry["source"] = null;
  if (d) {
    // The decls key and the /provenance key are both the writer's, so even an
    // unnamed `#k` joins by key here (the app's own #k numbering is not involved).
    const zone = s.artifacts.model?.zones["provenance"];
    if (!s.provenance.some((x) => x.key === d.key)) {
      source = { off: !zone || zone.status === "absent" ? "the model file has no /provenance zone" : zone.status === "ready" ? `no /provenance record for ${d.key}` : `the /provenance zone was not read (${zone.status})` };
    } else {
      const where = sourceFor(s, d.key);
      source = where.ok ? { decl: d.key, label: where.label, title: `open ${where.source.file}:${where.source.line} in the editor` } : { off: where.reason };
    }
  }
  return {
    h5: r.path,
    section: st ? `stage ${st.index} · ${st.name}` : "global",
    handle,
    declKey: d?.key ?? null,
    type: owner && owner.type !== r.type ? `${r.type} (${owner.type})` : r.type,
    records: p ? recordsOf(p) || "no response field is set" : "not in this file (no parameters by field name)",
    targets: t.text,
    output: outputOf(r, p),
    lifetime: `${added}; ${removalOf(s, d ? handle : null)}`,
    params,
    unset,
    paramsNote: d ? (owner ? null : "parameters by field name: not in this file (it has no /opensees/decl_params, opensees < 2.26.0)") : `parameters by field name: not joined (${why})`,
    select,
    selectNote: missing.length ? `group ${missing.join(", ")} is not a group of this file, so a click selects nothing` : null,
    source,
  };
}

/**
 * Why recorder `r` of `store` joins no declaration, following the checks of
 * selectors.recorderDecls in its order: no decls, no rows, a count mismatch,
 * a group name without a trailing `_<index>`, then a row past the records.
 */
function joinGap(s: State, recs: readonly DefObject[], r: DefObject, store: string): string {
  const decls = s.definitions?.decls;
  if (!decls) return "the file has no /opensees/decls (opensees < 2.25.0)";
  const rows = decls.rows[store];
  if (!rows) return `/opensees/decls/rows/${store} is not in this file`;
  if (rows.length !== recs.length) return `/opensees/decls/rows/${store} has ${rows.length} rows for ${recs.length} recorders`;
  const idx = recs.map((x) => Number(/_(\d+)$/.exec(x.groupName)?.[1] ?? Number.NaN));
  const bad = recs.filter((_, i) => !Number.isInteger(idx[i]));
  if (bad.length) return `the group name ${bad.map((x) => x.groupName).join(", ")} has no trailing _<index>, so /opensees/decls/rows/${store} cannot be ordered`;
  const k = [...idx].sort((a, b) => a - b).indexOf(idx[recs.indexOf(r)]!);
  return `/opensees/decls/rows/${store}[${k}] = ${rows[k]} is not a row of /opensees/decls (${decls.records.length} records)`;
}

/** The recorders window's content, from the store alone. */
export function recordersOf(s: State): RecordersContent {
  const model = s.artifacts.model;
  if (!model || model.status !== "ready") return { entries: [], note: "no model is open" };
  const defs = s.definitions;
  if (!defs) return { entries: [], note: "the model has no /opensees zone, so it declares no recorder" };
  const staged = defs.stages.length > 0;
  const entries: RecorderEntry[] = [];
  const add = (recs: readonly DefObject[], store: string, st: StageDef | null) => {
    const joined = recorderDecls(defs, recs, store);
    // In declaration order (the decls row) when joined; the file's group order otherwise.
    const order = recs.map((r, i) => ({ r, d: joined.get(r.path), i })).sort((a, b) => (a.d?.row ?? Infinity) - (b.d?.row ?? Infinity) || a.i - b.i);
    for (const { r, d } of order) entries.push(entryOf(s, r, d, st, staged, d ? "" : joinGap(s, recs, r, store)));
  };
  add(defs.recorders, "recorders", null);
  for (const st of defs.stages) add(st.recorders, `stages/stage_${String(st.index).padStart(3, "0")}/recorders`, st);
  if (!entries.length) return { entries, note: "the model declares no recorder (no /opensees/recorders, and no stage adds one)" };
  const where = defs.decls?.hasParams ? "parameters by field name from /opensees/decl_params" : "this file has no /opensees/decl_params, so no parameters by field name";
  return { entries, note: `${plural(entries.length, "recorder")} · ${where}` };
}

function line(label: string, text: string): HTMLElement {
  const row = el("div", "window-row-detail");
  const b = el("span", "f-label", `${label} `);
  b.style.color = "var(--muted)";
  row.append(b, text);
  return row;
}

export const view: WindowView = (s: State, act: (e: Event) => void) => {
  const content = recordersOf(s);
  const body = el("div", "window-list");
  body.append(el("div", "window-note", content.note));
  let section: string | null = null;
  for (const r of content.entries) {
    if (r.section !== section) {
      section = r.section;
      body.append(el("div", "window-section", section));
    }
    const row = el("div", "window-row" + (r.select ? " clickable" : ""));
    row.dataset["recorder"] = r.h5;
    const head = el("div", "window-row-head");
    head.append(el("span", "window-row-title", r.handle), el("span", "node-type", r.type));
    if (r.declKey) head.append(el("span", "window-row-decl", r.declKey));
    if (r.source) {
      const go = el("button", "source", "off" in r.source ? "source" : r.source.label);
      if ("off" in r.source) {
        go.title = r.source.off;
        go.disabled = true;
      } else {
        const { decl, title } = r.source;
        go.title = title;
        go.addEventListener("click", (e) => {
          e.stopPropagation();
          act({ type: "requestSource", decl });
        });
      }
      head.append(go);
    }
    row.append(head, line("records", r.records), line("targets", r.targets), line("output", r.output), line("stage", r.lifetime));
    if (r.params.length) {
      const table = el("table", "fields");
      table.style.fontSize = "12px";
      for (const p of r.params) {
        const tr = el("tr");
        tr.append(el("td", "f-label", p.field), el("td", "f-value", p.value));
        table.append(tr);
      }
      row.append(table);
    }
    if (r.unset.length) row.append(el("div", "window-note", `unset: ${r.unset.join(", ")}`));
    if (r.paramsNote) row.append(el("div", "window-note", r.paramsNote));
    if (r.selectNote) row.append(el("div", "window-note", r.selectNote));
    row.append(el("div", "f-src", r.h5));
    const decl = r.select;
    if (decl) {
      // The frame highlights the row while the store selects its group (frame.ts markSelected).
      row.dataset["decl"] = decl;
      row.title = `select ${decl}`;
      row.addEventListener("click", () => act({ type: "select", pick: { decl, at: null, nodes: null } }));
    }
    body.append(row);
  }
  return body;
};
