// The stage diagram (V3c1, #1642; ADR 0112 D2b, D6, D7).
//
// The model's sequence as the root `/sequence` zone records it (V3a): the
// stages in order on a horizontal timeline, each with its name, analysis
// type, integrator / algorithm / test, steps and dt, and the changes it makes
// (`add` / `remove` / `activate` / `set` x pattern, recorder, fix, ...), each
// change by name (or `#k` key) and count. The declarations outside any stage
// are listed once, before the timeline: `/sequence` gives them stage 0
// because they are in force from the first analysis. An unstaged deck shows
// its one implicit stage. A file without `/sequence` says it predates
// readable stages; the window never rebuilds a sequence from `/opensees`.
//
// `stagesOf` is the pure read of the state (selectors only); `view` builds
// the DOM from it. A click on a stage card selects that stage (`selectStage`,
// V4f; a click on the selected card clears it), and the card the store names
// in `State.stage` is highlighted; the stage is not a phase, so the viewport
// does not change. A click on a change whose declaration has a recorded
// source asks for go-to-source (`requestSource`) and does not select the
// stage. Tags are evidence, names are handles (D2c).

import { declOfTag, sourceFor } from "../../state/selectors.ts";
import type { Event, State } from "../../state/store.ts";
import type { Definitions, DefObject, Table } from "../../state/types.ts";
import { el } from "../../ui/dom.ts";
import type { WindowView } from "./view.ts";

/** A declaration a change names: its key, and whether go-to-source can reach it (else why not). */
export interface DeclLink {
  key: string;
  /** `file:line` of its source, or null when go-to-source cannot jump */
  source: string | null;
  why: string | null;
}

/** One `/sequence/changes` row, as the diagram shows it. */
export interface ChangeItem {
  /** the declaration's name, else its `#k` handle, else `unnamed` */
  label: string;
  count: number;
  decl: DeclLink | null;
  h5: string;
}

/** The rows of one `(op, what)` pair, in file order. */
export interface ChangeGroup {
  op: string;
  what: string;
  items: ChangeItem[];
  /** why the pair is not in the /sequence 1.0.0 vocabulary (an unknown op or what), or null */
  unknown: string | null;
}

/** The `/sequence/changes` vocabulary (h5-schema.md, `/sequence`, "Changes"). */
export const OPS: readonly string[] = ["add", "remove", "activate", "set"];
export const WHATS: readonly string[] = [
  "pattern", "recorder", "fix", "mass", "element", "group", "region",
  "rayleigh", "initial_stress", "absorbing", "material_stage", "constraint",
];

function unknownOf(op: string, what: string): string | null {
  const bad = [!OPS.includes(op) ? `op '${op}'` : "", !WHATS.includes(what) ? `what '${what}'` : ""].filter(Boolean);
  return bad.length ? `unknown ${bad.join(" and ")}: not in the /sequence 1.0.0 vocabulary` : null;
}

export interface StageCard {
  /** the row index in `/sequence/stages`: what `selectStage` and `State.stage` name */
  index: number;
  /** this is the store's selected stage */
  selected: boolean;
  order: number;
  /** `gravity`, `stage 2` for an unnamed stage, `implicit stage` for an unstaged deck */
  title: string;
  analysis: string;
  /** `LoadControl · Newton · NormDispIncr`; `not set` for a missing token */
  chain: string;
  /** steps in words: `2 steps`, or why there are none */
  steps: string;
  /** dt in words: `dt 0.01`, `static (no dt)`, or `no analyze_dt archived` */
  dt: string;
  /** `set_time 0`, or null when the stage does not set it */
  setTime: string | null;
  domainChange: boolean;
  changes: ChangeGroup[];
  h5: string;
}

export interface StagesModel {
  /** a sentence when there is nothing (or something missing) to say why */
  note: string | null;
  /** `/sequence 1.0.0 · staged (2 stages)` */
  header: string | null;
  /** the declarations outside any stage (in force from the first analysis) */
  global: ChangeGroup[];
  stages: StageCard[];
}

/**
 * The stage verbs that write no `/sequence` row (h5-schema.md, `/sequence`,
 * "Not projected"; archive gap #1646). The diagram says so on every file
 * that has a sequence, so an absent row is never read as an absent action.
 */
export const NOT_RECORDED = ["update_parameter", "zero_velocities", "set_creep_on", "pre_analyze_reset"] as const;
export const NOT_RECORDED_NOTE =
  `Not recorded in /sequence: ${NOT_RECORDED.join(", ")} (archive gap #1646). ` +
  "A stage may still make these actions; the diagram cannot show them.";

const STAGE_COLUMNS = ["order", "name", "analysis", "integrator", "algorithm", "test", "steps", "dt", "set_time", "domain_change", "source"] as const;
const CHANGE_COLUMNS = ["stage", "op", "what", "name", "count", "source"] as const;

/** Column accessors for a flat table, or the name of the first missing column. */
function columns<K extends string>(t: Table, names: readonly K[]): { get: (row: unknown[], k: K) => unknown } | { missing: string } {
  const at = new Map<string, number>(t.columns.map((c, i) => [c, i]));
  const absent = names.find((n) => !at.has(n));
  if (absent !== undefined) return { missing: absent };
  return { get: (row, k) => row[at.get(k)!] };
}

const str = (v: unknown): string => (typeof v === "string" ? v : v === null || v === undefined ? "" : String(v));
/** A float column: h5wasm gives NaN as null in `json_value`. */
const num = (v: unknown): number => (typeof v === "number" ? v : v === null || v === undefined ? Number.NaN : Number(v));
const fmt = (x: number): string => String(Number(x.toPrecision(6)));

/** `opensees/pattern/#1` -> `#1`, `opensees/pattern/support:gravity` -> `support:gravity`. */
export const handleOf = (key: string): string => key.split("/").slice(2).join("/") || key;

/** Every DefObject the definitions carry, by its HDF5 path. */
function objectsByPath(defs: Definitions): Map<string, DefObject> {
  const out = new Map<string, DefObject>();
  for (const o of [...defs.patterns, ...defs.recorders, ...defs.stages.flatMap((st) => [...st.patterns, ...st.recorders])]) out.set(o.path, o);
  return out;
}

/**
 * The `/opensees/decls` key a change row names, read the way V3a's deriver
 * wrote it (sequence_h5.py): a pattern by its group's tag; `<store>[i]` by
 * the `i`-th entry of `decls/rows/<store>`; a grouped store (`bcs/fix`,
 * `remove_sp`, ...) by the `j`-th distinct declaration of `rows/<store>` for
 * its `j`-th row. A key whose name disagrees with the row's is dropped, so a
 * stale join never names the wrong declaration. Null when there is none.
 */
function keyResolver(defs: Definitions): (source: string, what: string, name: string) => string | null {
  const objects = objectsByPath(defs);
  const seen = new Map<string, number>();
  const decls = defs.decls;
  const checked = (row: number | undefined, name: string): string | null => {
    const r = row === undefined ? undefined : decls?.records[row];
    return r && (name === "" || r.name === name) ? r.key : null;
  };
  return (source, what, name) => {
    if (!decls || !source.startsWith("/opensees/")) return null;
    if (what === "pattern") {
      const p = objects.get(source);
      const d = p ? declOfTag(defs, "pattern", p.tag) : null;
      return d && (name === "" || d.name === name) ? d.key : null;
    }
    const m = /^\/opensees\/(.+?)(?:\[(\d+)\])?$/.exec(source);
    if (!m) return null;
    const rows = decls.rows[m[1]!];
    if (!rows) return null;
    if (m[2] !== undefined) return checked(rows[Number(m[2])], name);
    const j = seen.get(source) ?? 0;
    seen.set(source, j + 1);
    return checked([...new Set(rows)][j], name);
  };
}

function linkOf(s: State, key: string): DeclLink {
  const where = sourceFor(s, key);
  if (where.ok) return { key, source: where.label, why: null };
  const why = !(key in s.decls) && !s.provenance.some((p) => p.key === key)
    ? `no /provenance record for ${key}` + (s.provenance.length ? "" : " (the file records no sources)")
    : where.reason;
  return { key, source: null, why };
}

/** `steps = -1` is the archive's "no analyze_steps" (h5-schema.md, `/sequence`): said in words, never shown as a count. */
export function stepsText(steps: number): string {
  if (steps === -1) return "no analyze_steps archived (steps = -1): the deck runs its own analysis loop, so no step count is known";
  if (!Number.isInteger(steps) || steps < 0) return `steps ${steps}: not a step count`;
  return `${steps} step${steps === 1 ? "" : "s"}`;
}

export function dtText(dt: number, analysis: string): string {
  if (Number.isFinite(dt)) return `dt ${fmt(dt)}`;
  return analysis === "Static" ? "static (no dt)" : "no analyze_dt archived (dt = NaN)";
}

/** What the stage diagram shows, from the store alone. */
export function stagesOf(s: State): StagesModel {
  const empty = (note: string): StagesModel => ({ note, header: null, global: [], stages: [] });
  const model = s.artifacts.model;
  if (!model || model.status !== "ready") return empty("no model is open");
  const seq = s.sequence;
  if (seq.status === "absent") {
    return empty(
      "This file predates readable stages: it has no /sequence zone (apeGmsh writes one from sequence schema 1.0.0 on). " +
        "Re-write the model with a current apeGmsh to see its stages here.",
    );
  }
  if (seq.status === "refused") return empty(`/sequence ${seq.version} was not read: ${seq.reason} (this app reads ${seq.accepted})`);

  const sc = columns(seq.stages, STAGE_COLUMNS);
  if ("missing" in sc) return empty(`/sequence/stages has no '${sc.missing}' column: the diagram cannot be drawn from this file`);
  const cc = columns(seq.changes, CHANGE_COLUMNS);
  if ("missing" in cc) return empty(`/sequence/changes has no '${cc.missing}' column: the diagram cannot be drawn from this file`);

  const staged = Number(seq.attrs["staged"]) === 1;
  const resolve = s.definitions ? keyResolver(s.definitions) : () => null;

  // Group a stage's change rows by (op, what), in first-appearance order.
  const groupsOf = (rows: unknown[][]): ChangeGroup[] => {
    const out: ChangeGroup[] = [];
    for (const r of rows) {
      const op = str(cc.get(r, "op")), what = str(cc.get(r, "what")), name = str(cc.get(r, "name")), h5 = str(cc.get(r, "source"));
      const key = resolve(h5, what, name);
      let g = out.find((x) => x.op === op && x.what === what);
      if (!g) out.push((g = { op, what, items: [], unknown: unknownOf(op, what) }));
      g.items.push({ label: name || (key ? handleOf(key) : "unnamed"), count: num(cc.get(r, "count")), decl: key ? linkOf(s, key) : null, h5 });
    }
    return out;
  };
  // `/sequence` gives the declarations outside any stage stage 0; their source is outside /opensees/stages.
  const isStaged = (r: unknown[]): boolean => str(cc.get(r, "source")).startsWith("/opensees/stages/");
  const global = groupsOf(seq.changes.rows.filter((r) => !isStaged(r)));

  const stages = seq.stages.rows.map((r, index): StageCard => {
    const order = num(sc.get(r, "order"));
    const name = str(sc.get(r, "name"));
    const analysis = str(sc.get(r, "analysis"));
    const token = (k: "integrator" | "algorithm" | "test"): string => str(sc.get(r, k)) || `no ${k}`;
    const setTime = num(sc.get(r, "set_time"));
    return {
      index,
      selected: s.stage === index,
      order,
      title: name || (staged ? `stage ${order}` : "implicit stage"),
      analysis: analysis || "no analysis",
      chain: [token("integrator"), token("algorithm"), token("test")].join(" · "),
      steps: stepsText(num(sc.get(r, "steps"))),
      dt: dtText(num(sc.get(r, "dt")), analysis),
      setTime: Number.isFinite(setTime) ? `set_time ${fmt(setTime)}` : null,
      domainChange: num(sc.get(r, "domain_change")) === 1,
      changes: groupsOf(seq.changes.rows.filter((c) => isStaged(c) && num(cc.get(c, "stage")) === order)),
      h5: str(sc.get(r, "source")),
    };
  });

  const n = stages.length;
  const header = `/sequence ${seq.version} · ` + (staged ? `staged, ${n} stage${n === 1 ? "" : "s"}` : n ? "unstaged deck: one implicit stage, from /opensees/analysis" : "no stage");
  const note = n ? null : "the model declares no analysis: /sequence has no stage";
  return { note, header, global, stages };
}

// ---- DOM ---------------------------------------------------------------------

const LINK = "cursor: pointer; color: var(--accent); text-decoration: underline;";

function itemEl(it: ChangeItem, act: (e: Event) => void): HTMLElement {
  const e = el("span", "window-ref", it.label + (it.count === 1 ? "" : ` ×${it.count}`));
  const d = it.decl;
  if (d?.source) {
    e.setAttribute("style", LINK);
    e.title = `go to source: ${d.key} (${d.source}) · ${it.h5}`;
    e.addEventListener("click", (ev) => {
      ev.stopPropagation();
      act({ type: "requestSource", decl: d.key });
    });
  } else {
    e.title = d ? `${d.key}: ${d.why ?? "no source"} · ${it.h5}` : `${it.h5}: no /opensees/decls row names it`;
  }
  return e;
}

function changesEl(groups: readonly ChangeGroup[], act: (e: Event) => void, none: string): HTMLElement {
  const box = el("div", "stage-changes");
  if (!groups.length) {
    box.append(el("div", "window-note", none));
    return box;
  }
  for (const g of groups) {
    const line = el("div", "window-row-detail");
    line.dataset["change"] = `${g.op} ${g.what}`;
    const verb = el("span", "stage-verb", `${g.op} ${g.what}`);
    verb.setAttribute("style", "color: var(--muted);");
    line.append(verb, " ");
    if (g.unknown) {
      // Flagged, never drawn like a known change.
      const flag = el("span", "stage-unknown", `[${g.unknown}] `);
      flag.setAttribute("style", "color: var(--error); font-weight: 600;");
      line.dataset["unknown"] = "1";
      line.append(flag);
    }
    g.items.forEach((it, i) => {
      if (i) line.append(", ");
      line.append(itemEl(it, act));
    });
    box.append(line);
  }
  return box;
}

const CARD = "flex: 0 0 186px; padding: 6px 8px; border: 1px solid var(--line); border-radius: 8px; background: var(--faint); cursor: pointer;";
/** The selected card: the office accent, as a selected row is drawn (frame.ts SELECTED_STYLE). */
const CARD_SELECTED =
  "flex: 0 0 186px; padding: 6px 8px; border: 1px solid var(--accent); border-radius: 8px; background: var(--accent-wash); box-shadow: inset 3px 0 0 var(--accent); cursor: pointer;";

function stageEl(st: StageCard, act: (e: Event) => void): HTMLElement {
  const card = el("div", st.selected ? "stage-card selected" : "stage-card");
  card.dataset["h5"] = st.h5;
  card.dataset["stage"] = String(st.index);
  card.setAttribute("style", st.selected ? CARD_SELECTED : CARD);
  card.title = st.selected ? `${st.title}: selected (click to clear)` : `select ${st.title}`;
  card.addEventListener("click", () => act({ type: "selectStage", index: st.selected ? null : st.index }));
  const head = el("div", "window-row-head");
  head.append(el("span", "window-row-decl", String(st.order)), el("span", "window-row-title", st.title));
  card.append(head);
  card.append(el("div", "window-row-detail", st.analysis));
  card.append(el("div", "window-row-detail", st.chain));
  card.append(el("div", "window-row-detail", st.steps));
  card.append(el("div", "window-row-detail", [st.dt, ...(st.setTime ? [st.setTime] : []), ...(st.domainChange ? ["domain change"] : [])].join(" · ")));
  card.append(changesEl(st.changes, act, "no change of its own recorded in /sequence"));
  card.append(el("div", "f-src", st.h5));
  return card;
}

export const view: WindowView = (s, act) => {
  const m = stagesOf(s);
  const body = el("div", "window-list");
  if (m.note) body.append(el("div", "window-note", m.note));
  if (!m.header) return body;
  body.append(el("div", "window-note", m.header));
  if (m.global.length) {
    body.append(el("div", "window-section", "in force from the first stage (outside any stage)"));
    body.append(changesEl(m.global, act, ""));
  }
  if (m.stages.length) {
    body.append(el("div", "window-section", "stages"));
    const line = el("div", "stage-timeline");
    line.setAttribute("style", "display: flex; align-items: stretch; gap: 4px; overflow-x: auto; padding-bottom: 4px;");
    m.stages.forEach((st, i) => {
      if (i) {
        const arrow = el("div", "stage-arrow", "→");
        arrow.setAttribute("style", "align-self: center; color: var(--muted);");
        line.append(arrow);
      }
      line.append(stageEl(st, act));
    });
    body.append(line);
  }
  const gap = el("div", "window-note", NOT_RECORDED_NOTE);
  gap.dataset["gap"] = "1646";
  body.append(gap);
  return body;
};
