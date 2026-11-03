// Modal table panel (ADR 0130 D6, Amendment 1 A6; slice VU7a): the `/modes`
// table of the set's `.user.h5`, shown beside the mode list while the Modes
// family is on the phase bar's results axis. One line per row: the mode, its
// period, its effective mass ratio and the cumulative ratio per component, and
// the row's A6 verdict against the paired `.ladruno` (its lambda against the
// recorded LAMBDA). A stale row says so, and why, in its own cell; nothing is
// shown as current on no proof. The row of the mode shown is marked. A mode
// the table has no row for is shown without one, not refused (D6).
//
// The panel only reads the store; it dispatches nothing. Its card is created
// here and placed in `#rail` after the offers (index.html is not this
// panel's), so its look is inline, with the theme's variables only.

import type { State, Store } from "../state/store.ts";
import type { StaleMark, UserModeRow } from "../state/types.ts";
import { byId, el } from "../ui/dom.ts";

const baseName = (path: string): string => path.split(/[\\/]/).pop() ?? path;

/** One line of the table, as drawn. */
export interface ModalLine {
  mode: number;
  /** "1.234" (s) */
  period: string;
  /** one per component: the effective mass ratio, then the cumulative ratio, in percent */
  ratios: string[];
  cumulative: string[];
  stale: boolean;
  /** "current", or "stale" */
  verdict: string;
  /** the row's tooltip: its address, lambda, generalised mass, participation factors, and the verdict's reason */
  title: string;
  /** the mode the viewport shows */
  shown: boolean;
}

/** What the panel draws, or null when it is hidden. */
export interface ModalTableView {
  file: string;
  components: string[];
  /** the table's lines; [] when the zone is refused */
  lines: ModalLine[];
  /** e.g. "12 rows · 2 stale", "12 rows · current", "refused" */
  status: string;
  /** a whole-zone note: refused or stale, with its reason; null when none */
  note: string | null;
  /** the zone itself is refused or stale */
  bad: boolean;
  /** the mode shown has no row in the table (D6), else null */
  missing: number | null;
}

/** "—" for NaN (h5-schema.md: not defined here). */
const num = (x: number, f: (v: number) => string): string => (Number.isNaN(x) ? "—" : f(x));
const pct = (x: number) => num(x, (v) => (v * 100).toFixed(1));
const sig = (x: number) => num(x, (v) => (Math.abs(v) >= 1e-3 && Math.abs(v) < 1e5 ? v.toPrecision(4).replace(/(\.\d*?)0+$/, "$1").replace(/\.$/, "") : v.toExponential(3)));

const verdictText = (m: StaleMark) => (m.stale ? "stale" : "current");

/** The mode index the viewport shows, when it shows a mode stage; else null. */
function shownMode(s: State): number | null {
  const at = s.phase.at;
  if (at?.kind !== "results") return null;
  const stage = s.results?.stages[at.stage];
  return stage?.kind === "mode" && stage.mode ? stage.mode.index : null;
}

function lineOf(r: UserModeRow, components: string[], shown: number | null, unique: boolean): ModalLine {
  const gamma = components.map((c, i) => `${c} ${sig(r.participation[i]!)}`).join(", ");
  const title = [
    `mode ${r.mode} (${r.kind}) of ${r.host}, ${r.step === -1 ? "its only set" : `STEP_${r.step}`}`,
    `lambda ${sig(r.lambda)}, period ${sig(r.period)} s, generalised mass ${sig(r.generalizedMass)}`,
    `participation factor: ${gamma}`,
    r.verdict.stale ? `stale: ${r.verdict.reason || "the reader gave no reason"}` : "current: its lambda equals the results file's",
  ].join("\n");
  return {
    mode: r.mode,
    period: sig(r.period),
    ratios: r.massRatio.map(pct),
    cumulative: r.cumulativeRatio.map(pct),
    stale: r.verdict.stale,
    verdict: verdictText(r.verdict),
    title,
    shown: unique && shown === r.mode,
  };
}

/**
 * The modal table on offer now: while the viewport shows a mode stage and the
 * set's `.user.h5` holds `/modes`. Null otherwise.
 */
export function modalTableOf(s: State): ModalTableView | null {
  if (s.user?.status !== "ready") return null;
  const m = s.user.info.modes;
  const shown = shownMode(s);
  if (m === null || shown === null) return null;
  const file = baseName(s.user.path);
  if (!m.ok) return { file, components: [], lines: [], status: "refused", note: `the modal table of ${file} is refused: ${m.reason}`, bad: true, missing: null };
  // A mode number names one row only when the table holds one set; with several, no row is marked.
  const unique = m.rows.filter((r) => r.mode === shown).length === 1;
  const lines = m.rows.map((r) => lineOf(r, m.components, shown, unique));
  const stale = lines.filter((l) => l.stale).length;
  const rows = `${lines.length} ${lines.length === 1 ? "row" : "rows"}`;
  const status = stale === 0 ? `${rows} · current` : stale === lines.length ? `${rows} · stale` : `${rows} · ${stale} stale`;
  const note = m.stale.stale ? `the modal table of ${file} is stale: ${m.stale.reason || "the reader gave no reason"}` : null;
  return { file, components: m.components, lines, status, note, bad: m.stale.stale, missing: m.rows.some((r) => r.mode === shown) ? null : shown };
}

const CARD = "flex: 0 0 auto; min-height: 0; max-height: 45%; overflow: auto; padding: 10px 12px; border-radius: 10px; background: var(--panel); border: 1px solid var(--line);";
const HEAD = "display: flex; align-items: center; gap: 8px; margin: 0 0 6px; font-size: 12px; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; color: var(--muted);";
const CELL = "padding: 2px 5px; text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; border-top: 1px solid var(--line);";

export function mountModalTable(store: Store): () => void {
  const panel = el("aside");
  panel.id = "modal-table";
  panel.hidden = true;
  panel.style.cssText = CARD;
  // After the offers (VA5 keeps them at the top of the rail), above the inspector.
  const rail = byId("rail");
  const offers = rail.querySelector("#offers");
  if (offers) offers.after(panel);
  else rail.prepend(panel);
  const render = (s: State, prev: State | null) => {
    if (prev && s.user === prev.user && s.phase === prev.phase && s.results === prev.results) return;
    const v = modalTableOf(s);
    panel.hidden = v === null;
    if (v === null) {
      panel.replaceChildren();
      return;
    }
    const head = el("h3", undefined, "Modal table");
    head.style.cssText = HEAD;
    const status = el("span", "modal-status", v.status);
    status.style.cssText = `text-transform: none; letter-spacing: 0; color: var(${v.bad || v.lines.some((l) => l.stale) ? "--interp-text" : "--muted"});`;
    head.append(status);
    const parts: HTMLElement[] = [head];
    const sub = el("div", "modal-file", `${v.file}: effective mass ratio and cumulative, % of the total`);
    sub.style.cssText = "font-size: 12px; color: var(--muted); margin: 0 0 4px;";
    parts.push(sub);
    if (v.note) {
      const note = el("div", "modal-note", v.note);
      note.style.cssText = `font-size: 12px; margin: 0 0 4px; overflow-wrap: anywhere; color: var(${v.lines.length ? "--interp-text" : "--error"});`;
      parts.push(note);
    }
    if (v.missing !== null) {
      const miss = el("div", "modal-missing", `mode ${v.missing} has no row in this table`);
      miss.style.cssText = "font-size: 12px; margin: 0 0 4px; color: var(--muted);";
      parts.push(miss);
    }
    if (v.lines.length) parts.push(table(v));
    panel.replaceChildren(...parts);
  };
  const table = (v: ModalTableView): HTMLElement => {
    const t = el("table", "modal-rows");
    t.style.cssText = "width: 100%; border-collapse: collapse; font-size: 12px;";
    const hr = el("tr");
    const th = (text: string, title: string) => {
      const c = el("th", undefined, text);
      c.title = title;
      c.style.cssText = "padding: 2px 5px; text-align: right; font-weight: 600; color: var(--muted); white-space: nowrap;";
      hr.append(c);
    };
    th("mode", "1-based mode number");
    th("T (s)", "period");
    for (const c of v.components) th(`${c} %`, `effective mass ratio along ${c}`);
    for (const c of v.components) th(`Σ${c} %`, `cumulative effective mass ratio along ${c}`);
    th("status", "current: the row's lambda equals the results file's LAMBDA of that mode (1e-9 relative); stale otherwise, or when it cannot be compared (ADR 0130 A6)");
    t.append(hr);
    for (const l of v.lines) {
      const tr = el("tr", l.shown ? "modal-row shown" : "modal-row");
      tr.dataset["mode"] = String(l.mode);
      tr.title = l.title;
      if (l.shown) tr.style.cssText = "background: var(--accent-wash); font-weight: 700;";
      const td = (text: string, extra = "") => {
        const c = el("td", undefined, text);
        c.style.cssText = CELL + extra;
        tr.append(c);
        return c;
      };
      td(String(l.mode));
      td(l.period);
      for (const x of l.ratios) td(x);
      for (const x of l.cumulative) td(x);
      const verdict = td(l.verdict, ` font-weight: 600; color: var(${l.stale ? "--interp-text" : "--muted"});`);
      verdict.className = l.stale ? "modal-verdict stale" : "modal-verdict";
      t.append(tr);
    }
    return t;
  };
  render(store.get(), null);
  const unsubscribe = store.subscribe(render);
  return () => {
    unsubscribe();
    panel.remove();
  };
}
