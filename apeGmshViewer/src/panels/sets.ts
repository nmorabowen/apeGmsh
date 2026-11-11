// Sets panel (VU9a, ADR 0132 D3-D7): what the user makes, kept in the window
// set's `<stem>.workspace.h5`. It names a set from the selection (Nodes: the
// node of each clicked element nearest the click; Elements: the OpenSees
// elements selected), renames and deletes sets, and shows each set's verdict
// against the model on screen with its reason: current, moved, partial ("k of
// n", with Accept for the survivors) or lost. Rows that refuse alone are
// listed with their reason, never dropped; stored plots and views are listed
// by name (drawing a plot is a later slice). A save another program blocks
// shows the typed refusal naming the file (HG9 rule 2).
//
// The panel never names a path: it dispatches `workspaceSave` with the edit,
// and the effects carry it to main, which writes the set's own file (option A,
// #2753). It takes no room with no set open.

import type { State, Store } from "../state/store.ts";
import { pickSummaryOf, workspaceCannotSave } from "../state/selectors.ts";
import type { SetVerdictIn, WorkspaceSetIn, WorkspaceState } from "../state/types.ts";
import { byId, el, listen } from "../ui/dom.ts";

const baseName = (path: string): string => path.split(/[\\/]/).pop() ?? path;
const fmt = (d: number) => (d === 0 ? "0" : d.toExponential(2));

/** A set's chip and its reason, in words (tested in Node). */
export function verdictLine(set: WorkspaceSetIn): { chip: string; verdict: string; reason: string } {
  const v: SetVerdictIn | null = set.verdict;
  if (v === null) return { chip: "refused", verdict: "refused", reason: "the model cannot judge it (see the refusal below)" };
  switch (v.verdict) {
    case "current":
      return { chip: "current", verdict: "current", reason: "every identity hash equals the model's" };
    case "moved":
      return {
        chip: "moved",
        verdict: "moved",
        reason: `the model's hashes changed, and every member sits at its stored position (largest distance ${fmt(v.maxDistance ?? 0)}); the next save records the new hashes`,
      };
    case "partial":
      return {
        chip: `partial · ${v.matched} of ${v.total}`,
        verdict: "partial",
        reason: `${v.matched} of ${v.total} members sit at their stored positions; a plot that reads this set refuses until you accept the ${v.matched}`,
      };
    case "lost":
      return { chip: `lost · 0 of ${v.total}`, verdict: "lost", reason: "no member sits at its stored position in this model: pick the set again, or delete it" };
  }
}

/** What the panel states, as plain lines (tested in Node; the DOM below shows them). */
export function setsLines(s: State): {
  title: string;
  status: string | null;
  readOnly: string | null;
  canSave: boolean;
  why: string | null;
  kind: WorkspaceState["kind"];
  pick: { count: number; reason: string | null; ids: readonly number[] | null };
  sets: { name: string; kind: string; members: number; chip: string; verdict: string; reason: string; canAccept: boolean }[];
  refused: string[];
  plots: string[];
  views: string[];
  notice: { text: string; held: boolean } | null;
  saving: boolean;
} | null {
  const w = s.workspace;
  const r = w.read;
  if (r === null && s.artifacts.model !== null) {
    // A set is opening (a switch, a drop): the last set's sets are gone, and nothing is saved until this one's file is read.
    const model = s.artifacts.model.path.split(/[\\/]/).pop() ?? s.artifacts.model.path;
    return {
      title: model.replace(/\.h5$/i, ".workspace.h5"),
      status: `Reading the workspace of ${model}, the set now open: nothing can be saved until it is read.`,
      readOnly: null,
      canSave: false,
      why: workspaceCannotSave(w),
      kind: w.kind,
      pick: pickSummaryOf(s),
      sets: [],
      refused: [],
      plots: [],
      views: [],
      notice: null,
      saving: false,
    };
  }
  if (r === null || r.status === "none") return null;
  const why = workspaceCannotSave(w);
  const base = {
    title: `${baseName(r.file)}${r.status === "ready" ? ` · revision ${r.revision}` : ""}`,
    readOnly: r.status === "ready" ? r.readOnly : null,
    canSave: why === null,
    why,
    kind: w.kind,
    pick: pickSummaryOf(s),
    notice: w.notice === null ? null : { text: w.notice.held ? w.notice.detail : `Not saved (${w.notice.reason}): ${w.notice.detail}`, held: w.notice.held },
    saving: w.saving !== null,
  };
  if (r.status !== "ready") {
    const status =
      r.status === "absent"
        ? "No sets yet: the first save creates this file beside the model."
        : r.status === "unreadable"
          ? `Not a readable workspace (${r.reason}): ${r.detail}. A save moves it aside, under a name the banner shows, and never writes over it.`
          : `Refused (${r.reason}): ${r.detail}`;
    return { ...base, status, sets: [], refused: [], plots: [], views: [] };
  }
  return {
    ...base,
    status: r.broken.length ? r.broken.map((b) => `/${b.table} does not read: ${b.reason}`).join("; ") : null,
    sets: r.sets.map((x) => {
      const v = verdictLine(x);
      return { name: x.name, kind: x.kind, members: x.ids.length, ...v, canAccept: x.verdict?.verdict === "partial" };
    }),
    refused: r.refused.map((x) => `${x.table} row ${x.row}${x.name ? ` (${x.name})` : ""} refused, ${x.reason}: ${x.detail}`),
    plots: r.plots.map((p) => p.name),
    views: r.views.map((p) => p.name),
  };
}

const BOX = "flex: 0 0 auto; min-height: 0; max-height: 45%; overflow: auto; padding: 10px 12px; border-radius: 10px; background: var(--panel); border: 1px solid var(--line);";
const HEAD = "display: flex; align-items: center; gap: 8px; margin: 0 0 6px; font-size: 12px; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; color: var(--muted);";
const BUTTON = "padding: 1px 8px; border-radius: 5px; cursor: pointer; font: 600 12px var(--font); color: var(--text); background: var(--faint); border: 1px solid var(--line);";

function button(text: string, title: string, disabled: boolean, on: () => void, unlisten: (() => void)[]): HTMLButtonElement {
  const b = el("button", "sets-button", text);
  b.type = "button";
  b.title = title;
  b.disabled = disabled;
  b.style.cssText = BUTTON + (disabled ? " cursor: default; color: var(--muted); opacity: 0.6;" : "");
  unlisten.push(listen(b, "click", on));
  return b;
}

export function mountSets(store: Store): () => void {
  const panel = el("aside");
  panel.id = "sets";
  panel.hidden = true;
  panel.style.cssText = BOX;
  const rail = byId("rail");
  const after = document.getElementById("variants") ?? document.getElementById("offers");
  if (after && after.parentElement === rail) after.after(panel);
  else rail.prepend(panel);
  let unlisten: (() => void)[] = [];
  /** The name typed for a new set, kept across renders. */
  let draft = "";
  /** The set being renamed, and its new name as typed. */
  let renaming: { from: string; to: string } | null = null;
  const save = (edit: Parameters<typeof dispatchSave>[1]) => dispatchSave(store, edit);
  const render = (s: State, prev: State | null) => {
    if (prev && s.workspace === prev.workspace && s.selection === prev.selection && s.decls === prev.decls && s.artifacts.model === prev.artifacts.model) return;
    for (const u of unlisten) u();
    unlisten = [];
    const lines = setsLines(s);
    panel.hidden = lines === null;
    if (lines === null) {
      panel.replaceChildren();
      return;
    }
    const head = el("h3", undefined, "Sets");
    head.style.cssText = HEAD;
    head.append(el("span", "legend-count", String(lines.sets.length)));
    const parts: HTMLElement[] = [head, el("div", "offer-note", lines.title)];
    if (lines.readOnly) parts.push(el("div", "offer-reason", lines.readOnly));
    if (lines.status) parts.push(el("div", "offer-note", lines.status));
    // The kind a new set takes, and what the selection gives now.
    const kinds = el("div", "sets-kind");
    for (const k of ["nodes", "elements"] as const) {
      const b = button(k === "nodes" ? "Nodes" : "Elements", k === "nodes" ? "A click takes the node of the clicked element nearest the click" : "A click takes the OpenSees elements of the clicked cell", false, () => store.dispatch({ type: "setWorkspaceKind", kind: k }), unlisten);
      b.setAttribute("aria-pressed", String(lines.kind === k));
      // The inline button style wins over style.css, so the pressed kind is drawn here.
      if (lines.kind === k) b.style.cssText += " color: var(--accent-ink); background: var(--accent); border-color: transparent;";
      kinds.append(b);
    }
    parts.push(kinds);
    const noun = lines.kind === "nodes" ? (lines.pick.count === 1 ? "node" : "nodes") : lines.pick.count === 1 ? "element" : "elements";
    // The tags a save would take, as the effects computed them: in Nodes the viewport still highlights the clicked cells.
    const picked = lines.pick.ids ? `${noun} ${lines.pick.ids.join(", ")}${lines.kind === "nodes" ? " (the node of each clicked cell nearest the click)" : ""}` : `${lines.pick.count} ${noun} selected`;
    const note = el("div", "offer-note", lines.pick.reason ?? picked);
    note.dataset["role"] = "set-members";
    parts.push(note);
    const form = el("div", "sets-form");
    const input = el("input");
    input.type = "text";
    input.placeholder = "Name the selection";
    input.value = draft;
    input.dataset["role"] = "set-name";
    unlisten.push(listen(input, "input", () => (draft = input.value)));
    const name = () => draft.trim();
    form.append(
      input,
      button(lines.saving ? "Saving…" : "Save set", lines.why ?? (lines.pick.reason ? lines.pick.reason : "Save the selection as a named set"), !lines.canSave || lines.pick.count === 0, () => {
        if (!name()) return;
        save({ op: "create", name: name() });
        draft = "";
      }, unlisten),
    );
    parts.push(form);
    if (lines.notice) parts.push(el("div", lines.notice.held ? "sets-held" : "offer-reason", lines.notice.text));
    for (const x of lines.sets) {
      const row = el("div", "offer-row");
      row.dataset["set"] = x.name;
      const top = el("div", "offer-top");
      if (renaming?.from === x.name) {
        const r = el("input");
        r.type = "text";
        r.value = renaming.to;
        unlisten.push(listen(r, "input", () => (renaming = { from: x.name, to: r.value })));
        top.append(r);
        top.append(button("OK", `Rename ${x.name}`, !lines.canSave, () => {
          const to = renaming?.to.trim() ?? "";
          renaming = null;
          if (to && to !== x.name) save({ op: "rename", table: "sets", from: x.name, to });
          else render(store.get(), null);
        }, unlisten));
        top.append(button("Cancel", "Keep the name", false, () => {
          renaming = null;
          render(store.get(), null);
        }, unlisten));
      } else {
        top.append(el("span", "offer-name", x.name));
        const chip = el("span", "set-verdict", x.chip);
        chip.dataset["verdict"] = x.verdict;
        top.append(chip);
        top.append(button("Rename", `Rename ${x.name}`, !lines.canSave, () => {
          renaming = { from: x.name, to: x.name };
          render(store.get(), null);
        }, unlisten));
        top.append(button("Delete", `Delete ${x.name} from the file`, !lines.canSave, () => save({ op: "delete", table: "sets", name: x.name }), unlisten));
      }
      row.append(top);
      row.append(el("div", "offer-caption", `${x.members} ${x.kind} · ${x.reason}`));
      if (x.canAccept) row.append(button(`Accept the survivors`, `Keep only the members found, in order, with the model's hashes`, !lines.canSave, () => save({ op: "accept", name: x.name }), unlisten));
      parts.push(row);
    }
    for (const t of lines.refused) parts.push(el("div", "offer-reason", t));
    if (lines.plots.length) parts.push(el("div", "offer-note", `Plots stored: ${lines.plots.join(", ")} (drawing them is a later slice)`));
    if (lines.views.length) parts.push(el("div", "offer-note", `Views stored: ${lines.views.join(", ")}`));
    panel.replaceChildren(...parts);
  };
  render(store.get(), null);
  const unsubscribe = store.subscribe(render);
  return () => {
    unsubscribe();
    for (const u of unlisten) u();
    panel.remove();
  };
}

/** Dispatch one edit; a refusal the reducer raises (a save in flight, a read-only file) is shown, never thrown into the click. */
function dispatchSave(store: Store, edit: Extract<Parameters<Store["dispatch"]>[0], { type: "workspaceSave" }>["edit"]): void {
  if (workspaceCannotSave(store.get().workspace) !== null) return;
  store.dispatch({ type: "workspaceSave", edit });
}
