// Combine panel (VU9b, ADR 0130 A3, A4, A7.3; HG9 on #2693): the human writes
// a User Combination (its name, rule, terms and factors) and saves it into the
// window set's `<stem>.user.h5`. The phase bar's New combination opens it.
//
// While the human types, the draft is checked (`door.check`, the effects'
// `checkCombinationDraft`): the readers' rules beside the file's rows, then
// VU4's evaluator for one quantity. Only a draft that stands is offered for
// save; a refused one shows its reason and never reaches main. Save carries
// the row and the token of the read on screen (`expect`) to main's
// `combinations:save` (`door.save`), which copies the file, adds a generation
// and replaces the original (main/combinations.ts). Revert puts the last save
// back, once. A save another program blocks shows the typed refusal naming
// the file (HG9 rule 3). The panel never names a path.
//
// Its draft, check and save in flight are its own (state/types.ts
// `CombineState`); the file it writes into is the store's. A save that lands
// hands the store the file as main read it again (`userRead`), so the phase
// bar lists what it wrote at once.

import type { State, Store } from "../state/store.ts";
import { COMBINATION_ENVELOPES, COMBINATION_RULES, combineCannotSave, combineEntryOf, draftKeyOf, draftSourcesOf, EMPTY_DRAFT, TERM_KINDS } from "../state/selectors.ts";
import type { CombinationDraft, CombinationEdit, CombinationSaveIn, CombineState, DraftCheck, DraftTerm } from "../state/types.ts";
import { el, listen } from "../ui/dom.ts";

/** What the panel asks of the app (renderer/app.ts passes the effects' check and main's door). */
export interface CombineDoor {
  check(draft: CombinationDraft): Promise<DraftCheck | null>;
  save(edit: CombinationEdit): Promise<CombinationSaveIn>;
}

const baseName = (path: string): string => path.split(/[\\/]/).pop() ?? path;
const fmt = (x: number): string => (Number.isInteger(x) ? String(x) : Math.abs(x) >= 1e-3 && Math.abs(x) < 1e4 ? x.toPrecision(4) : x.toExponential(3));

/** The check in words, and whether it lets the draft be saved (tested in Node). */
export function checkLine(check: DraftCheck | null, key: string | null): { text: string; tone: "muted" | "ok" | "refused" } {
  if (key === null) return { text: "no user file is read", tone: "muted" };
  if (check === null || check.key !== key || check.status === "checking") return { text: "checking…", tone: "muted" };
  if (check.status === "refused") return { text: `refused (${check.reason}): ${check.detail}`, tone: "refused" };
  const p = check.preview;
  if (p === null) return { text: `stands. ${check.note ?? ""}`.trim(), tone: "ok" };
  const range = p.range ? `${fmt(p.range.min)} to ${fmt(p.range.max)}` : "no finite value";
  return { text: `stands: ${p.label} over ${p.nodes} nodes, ${range}, ${p.signed ? "signed" : "unsigned: contour only"}${check.note ? `. ${check.note}` : ""}`, tone: "ok" };
}

/** What the panel states, as plain lines (tested in Node; the DOM below shows them). Null with no user file to add to. */
export function combineLines(
  s: State,
  c: CombineState,
): { title: string; generation: string; check: { text: string; tone: "muted" | "ok" | "refused" }; canSave: boolean; why: string | null; canRevert: boolean; revertWhy: string | null; notice: { text: string; held: boolean } | null; last: string | null } | null {
  const entry = combineEntryOf(s);
  if (entry === null || s.user?.status !== "ready") return null;
  const u = s.user;
  const g = u.info.generations ?? null;
  const rows = u.info.table?.rows.length ?? 0;
  const generation = g === null ? "no /combinations yet: the first save creates it" : `g${g.current} in force (${rows} ${rows === 1 ? "row" : "rows"}); ${g.held.length} ${g.held.length === 1 ? "generation" : "generations"} kept`;
  const why = combineCannotSave(s, c, "add");
  const revertWhy = combineCannotSave(s, c, "revert");
  const last = c.last === null || c.last.file !== u.path ? null : c.last.op === "add" ? `Saved ${JSON.stringify(c.last.name)} as g${c.last.generation}.` : `Reverted: g${c.last.generation} is in force again.`;
  return {
    title: `New combination · ${baseName(u.path)}`,
    generation,
    check: checkLine(c.check, draftKeyOf(s, c.draft)),
    canSave: why === null,
    why,
    canRevert: revertWhy === null,
    revertWhy,
    notice: c.notice === null ? null : { text: c.notice.held ? c.notice.detail : `Not saved (${c.notice.reason}): ${c.notice.detail}`, held: c.notice.held },
    last,
  };
}

const BOX = "flex: 0 0 auto; min-height: 0; max-height: 60%; overflow: auto; padding: 10px 12px; border-radius: 10px; background: var(--panel); border: 1px solid var(--line);";
const HEAD = "display: flex; align-items: center; gap: 8px; margin: 0 0 6px; font-size: 12px; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; color: var(--muted);";
const BUTTON = "padding: 1px 8px; border-radius: 5px; cursor: pointer; font: 600 12px var(--font); color: var(--text); background: var(--faint); border: 1px solid var(--line);";
const FIELD = "min-width: 0; padding: 2px 6px; border-radius: 5px; font: 12px var(--font); color: var(--text); background: var(--faint); border: 1px solid var(--line);";

function button(text: string, title: string, disabled: boolean, on: () => void, unlisten: (() => void)[]): HTMLButtonElement {
  const b = el("button", "combine-button", text);
  b.type = "button";
  b.title = title;
  b.disabled = disabled;
  b.style.cssText = BUTTON + (disabled ? " cursor: default; color: var(--muted); opacity: 0.6;" : "");
  unlisten.push(listen(b, "click", on));
  return b;
}

function select(options: readonly string[], value: string, role: string, title: string): HTMLSelectElement {
  const s = el("select");
  s.dataset["role"] = role;
  s.title = title;
  s.style.cssText = FIELD;
  for (const o of options) {
    const opt = el("option", undefined, o === "" ? "(none)" : o);
    opt.value = o;
    opt.selected = o === value;
    s.append(opt);
  }
  return s;
}

function input(value: string, role: string, title: string, width?: string): HTMLInputElement {
  const i = el("input");
  i.type = "text";
  i.value = value;
  i.dataset["role"] = role;
  i.title = title;
  i.style.cssText = FIELD + (width ? ` width: ${width};` : "");
  return i;
}

/** A typed number, or NaN for text that is none (the readers refuse it with their reason). */
const num = (text: string): number => (text.trim() === "" ? Number.NaN : Number(text));
const optional = (text: string): number | null => (text.trim() === "" ? null : Number(text));
const shownNumber = (x: number | null): string => (x === null || Number.isNaN(x) ? "" : String(x));

/**
 * Mount the panel; `open()` shows it (the phase bar's New combination).
 * `receive(answer)` shows an answer main gives a save of the draft on screen,
 * as if the save had returned it: the capture harness shows the held refusal
 * with it (renderer/harness/combine.ts), which a capture window, which never
 * writes, cannot get from main.
 */
export function mountCombine(store: Store, door: CombineDoor): { open(): void; receive(answer: CombinationSaveIn): void; dispose(): void } {
  const panel = el("aside");
  panel.id = "combine";
  panel.hidden = true;
  panel.style.cssText = BOX;
  const rail = document.getElementById("rail");
  if (!rail) throw new Error("#rail is missing from index.html");
  const after = document.getElementById("sets") ?? document.getElementById("variants") ?? document.getElementById("offers");
  if (after && after.parentElement === rail) after.after(panel);
  else rail.prepend(panel);
  let open = false;
  let c: CombineState = { draft: EMPTY_DRAFT, check: null, saving: null, notice: null, last: null };
  let unlisten: (() => void)[] = [];
  let timer: ReturnType<typeof setTimeout> | null = null;
  /** The key a check is running for, so the same draft is not checked twice at once. */
  let checking: string | null = null;

  const set = (next: Partial<CombineState>) => {
    c = { ...c, ...next };
    render();
  };
  const setDraft = (draft: CombinationDraft, redraw: boolean) => {
    c = { ...c, draft, check: null };
    // Typing in a field redraws only the check line and the buttons, so the field keeps its focus.
    if (redraw) render();
    else refreshStatus();
    schedule();
  };
  /** Check the draft on screen once the typing pauses; a check of an earlier draft (or file) is dropped. */
  const schedule = () => {
    if (timer !== null) clearTimeout(timer);
    timer = setTimeout(runCheck, 150);
  };
  const runCheck = () => {
    timer = null;
    const s = store.get();
    const key = draftKeyOf(s, c.draft);
    if (!open || key === null || c.check?.key === key || checking === key) return;
    checking = key;
    const draft = c.draft;
    door.check(draft).then(
      (check) => {
        if (checking === key) checking = null;
        if (check !== null && check.key === draftKeyOf(store.get(), c.draft)) set({ check });
      },
      (err: unknown) => {
        if (checking === key) checking = null;
        if (key === draftKeyOf(store.get(), c.draft)) set({ check: { key, status: "refused", reason: "failed", detail: `the check failed: ${err instanceof Error ? err.message : String(err)}` } });
      },
    );
  };
  const save = (op: "add" | "revert") => {
    const s = store.get();
    if (combineCannotSave(s, c, op) !== null || s.user?.status !== "ready" || !s.user.token) return;
    const expect = s.user.token;
    if (op === "revert" && (c.last === null || c.last.target === null)) return;
    const edit: CombinationEdit =
      op === "add"
        ? { op, row: { name: c.draft.name, rule: c.draft.rule, envelope: c.draft.envelope, fraction: c.draft.fraction, damping: c.draft.damping, terms: c.draft.terms.map((t) => ({ ...t })) }, expect }
        : { op, target: c.last!.target!, expect };
    const path = s.user.path;
    // The generation in force when this save is read: what a revert of it goes back to.
    const target = s.user.info.generations?.current ?? null;
    set({ saving: op, notice: null });
    door.save(edit).then(
      (a) => take(a, edit, path, target),
      (err: unknown) => set({ saving: null, notice: { reason: "failed", detail: `the save failed: ${err instanceof Error ? err.message : String(err)}`, file: path, held: false } }),
    );
  };
  /** Main's answer to a save made on the file at `path`. */
  const take = (a: CombinationSaveIn, edit: CombinationEdit, path: string, target: number | null) => {
    // An answer about another file than the one on screen now is never shown as this set's.
    const now = store.get().user?.path ?? null;
    if (now !== path || (a.file !== null && a.file !== path)) return set({ saving: null });
    if (!a.ok) return set({ saving: null, notice: { reason: a.reason, detail: a.detail, file: a.file, held: a.held } });
    // The file as main read it after the replace: the phase bar lists what the save wrote at once.
    store.dispatch({ type: "userRead", path: a.file, read: a.read });
    const token = a.read.ok ? (a.read.token ?? null) : null;
    set({
      saving: null,
      notice: null,
      check: null,
      last: { op: a.op, name: edit.op === "add" ? edit.row.name : null, generation: a.generation, file: a.file, target: a.op === "add" ? target : null, token },
      draft: a.op === "add" ? EMPTY_DRAFT : c.draft,
    });
    schedule();
  };

  let status: HTMLElement | null = null;
  let actions: HTMLElement | null = null;
  /** The check line and the Save / Revert buttons, redrawn without the fields. */
  let statusUnlisten: (() => void)[] = [];
  const refreshStatus = () => {
    for (const u of statusUnlisten) u();
    statusUnlisten = [];
    const lines = combineLines(store.get(), c);
    if (lines === null || status === null || actions === null) return;
    status.textContent = lines.check.text;
    status.dataset["tone"] = lines.check.tone;
    status.style.color = `var(${lines.check.tone === "refused" ? "--error" : lines.check.tone === "ok" ? "--text" : "--muted"})`;
    const saveB = button(c.saving === "add" ? "Saving…" : "Save", lines.why ?? "Save this combination into the user file (a copy of it, then a replace)", !lines.canSave, () => save("add"), statusUnlisten);
    saveB.dataset["role"] = "combine-save";
    const revertB = button(c.saving === "revert" ? "Reverting…" : "Revert last save", lines.revertWhy ?? "Put the generation before the last save back in force (one revert)", !lines.canRevert, () => save("revert"), statusUnlisten);
    revertB.dataset["role"] = "combine-revert";
    actions.replaceChildren(saveB, revertB);
  };

  const render = () => {
    for (const u of unlisten) u();
    unlisten = [];
    const s = store.get();
    const lines = open ? combineLines(s, c) : null;
    panel.hidden = lines === null;
    if (lines === null) {
      panel.replaceChildren();
      status = actions = null;
      return;
    }
    const head = el("h3", undefined, "New combination");
    head.style.cssText = HEAD;
    head.append(
      button("Close", "Close the panel (the draft is kept)", false, () => {
        open = false;
        render();
      }, unlisten),
    );
    const parts: HTMLElement[] = [head, el("div", "offer-note", lines.title.replace(/^New combination · /, "into ")), el("div", "offer-note", lines.generation)];
    const d = c.draft;
    const grid = el("div", "combine-form");
    grid.style.cssText = "display: grid; grid-template-columns: auto minmax(0, 1fr); gap: 4px 6px; align-items: center; margin: 6px 0;";
    const name = input(d.name, "combine-name", "The combination's name (unique across the user file and the model)");
    name.placeholder = "Name";
    // As typed: a name with leading or trailing space is refused by the readers' name rule, and the check says so.
    unlisten.push(listen(name, "input", () => setDraft({ ...c.draft, name: name.value }, false)));
    const rule = select(COMBINATION_RULES, d.rule, "combine-rule", "How the terms combine (h5-schema.md, Rules)");
    unlisten.push(listen(rule, "change", () => setDraft({ ...c.draft, rule: rule.value, envelope: rule.value === "envelope" ? c.draft.envelope || "max" : "", fraction: rule.value === "directional" ? c.draft.fraction : null, damping: rule.value === "cqc" ? c.draft.damping : null }, true)));
    grid.append(el("span", "offer-caption", "Name"), name, el("span", "offer-caption", "Rule"), rule);
    if (d.rule === "envelope") {
      const env = select(COMBINATION_ENVELOPES, d.envelope, "combine-envelope", "Which envelope");
      unlisten.push(listen(env, "change", () => setDraft({ ...c.draft, envelope: env.value }, true)));
      grid.append(el("span", "offer-caption", "Envelope"), env);
    }
    if (d.rule === "directional") {
      const p = input(shownNumber(d.fraction), "combine-fraction", "The directional fraction p, in [0, 1]");
      unlisten.push(listen(p, "input", () => setDraft({ ...c.draft, fraction: optional(p.value) }, false)));
      grid.append(el("span", "offer-caption", "Fraction"), p);
    }
    if (d.rule === "cqc") {
      const z = input(shownNumber(d.damping), "combine-damping", "A damping ratio in (0, 1); empty takes each case's own");
      unlisten.push(listen(z, "input", () => setDraft({ ...c.draft, damping: optional(z.value) }, false)));
      grid.append(el("span", "offer-caption", "Damping"), z);
    }
    parts.push(grid);
    // The terms: one source address and a factor each (A3).
    const sources = draftSourcesOf(s);
    const terms = el("div", "combine-terms");
    terms.style.cssText = "display: flex; flex-direction: column; gap: 4px;";
    terms.append(el("div", "offer-caption", "Terms: kind · source · step · mode · factor"));
    d.terms.forEach((t, i) => {
      const row = el("div", "combine-term");
      row.dataset["term"] = String(i);
      row.style.cssText = "display: grid; grid-template-columns: auto minmax(0, 1fr) 3em 3em 4em auto; gap: 4px; align-items: center;";
      const put = (patch: Partial<DraftTerm>, redraw: boolean) => setDraft({ ...c.draft, terms: c.draft.terms.map((x, j) => (j === i ? { ...x, ...patch } : x)) }, redraw);
      const kind = select(TERM_KINDS, t.kind, "term-kind", "What the term reads");
      unlisten.push(listen(kind, "change", () => put({ kind: kind.value, step: -1, mode: kind.value === "mode" ? 1 : 0 }, true)));
      const src = input(t.source, "term-source", `The ${t.kind} it reads${sources[t.kind]?.length ? `: ${sources[t.kind]!.join(", ")}` : ""}`);
      const list = el("datalist");
      list.id = `combine-sources-${i}`;
      for (const x of sources[t.kind] ?? []) {
        const o = el("option");
        o.value = x;
        list.append(o);
      }
      src.setAttribute("list", list.id);
      unlisten.push(listen(src, "input", () => put({ source: src.value }, false)));
      const step = input(String(t.step), "term-step", "The step (a time index; -1 the last)");
      unlisten.push(listen(step, "input", () => put({ step: num(step.value) }, false)));
      const mode = input(String(t.mode), "term-mode", "The mode (1-based for a mode term, an increment for a spectrum term, 0 otherwise)");
      unlisten.push(listen(mode, "input", () => put({ mode: num(mode.value) }, false)));
      const factor = input(String(t.factor), "term-factor", "The factor");
      unlisten.push(listen(factor, "input", () => put({ factor: num(factor.value) }, false)));
      const drop = button("×", "Remove this term", d.terms.length < 2, () => setDraft({ ...c.draft, terms: c.draft.terms.filter((_, j) => j !== i) }, true), unlisten);
      row.append(kind, src, step, mode, factor, drop, list);
      terms.append(row);
    });
    const more = button("Add a term", "Add a term to the combination", false, () => setDraft({ ...c.draft, terms: [...c.draft.terms, { kind: c.draft.terms.at(-1)?.kind ?? "stage", source: "", step: -1, mode: 0, factor: 1 }] }, true), unlisten);
    more.dataset["role"] = "combine-add-term";
    terms.append(more);
    parts.push(terms);
    status = el("div", "combine-check");
    status.dataset["role"] = "combine-check";
    status.style.cssText = "margin: 6px 0; font-size: 12px; overflow-wrap: anywhere;";
    actions = el("div", "combine-actions");
    actions.style.cssText = "display: flex; gap: 6px; flex-wrap: wrap;";
    parts.push(status, actions);
    if (lines.notice) {
      const n = el("div", lines.notice.held ? "sets-held" : "offer-reason", lines.notice.text);
      n.dataset["role"] = "combine-notice";
      parts.push(n);
    }
    if (lines.last) parts.push(el("div", "offer-note", lines.last));
    panel.replaceChildren(...parts);
    refreshStatus();
  };

  let prev: State | null = null;
  const unsubscribe = store.subscribe((s) => {
    if (prev !== null && s.user === prev.user && s.results === prev.results) return;
    prev = s;
    // Another read of the file (a save, a watcher's re-read, a set switch): the check is for the file on screen.
    if (open) {
      render();
      schedule();
    }
  });
  return {
    open() {
      open = true;
      render();
      schedule();
      panel.scrollIntoView?.({ block: "nearest" });
    },
    receive(answer: CombinationSaveIn) {
      const u = store.get().user;
      if (u?.status !== "ready") return;
      take(answer, { op: "add", row: { ...c.draft, terms: c.draft.terms.map((t) => ({ ...t })) }, expect: u.token ?? "" }, u.path, u.info.generations?.current ?? null);
    },
    dispose() {
      if (timer !== null) clearTimeout(timer);
      unsubscribe();
      for (const u of [...unlisten, ...statusUnlisten]) u();
      panel.remove();
    },
  };
}
