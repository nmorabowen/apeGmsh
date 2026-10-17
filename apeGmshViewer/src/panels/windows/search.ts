// The Search window (ADR 0112 D6, D7): visual boolean query builder + free-text
// search over the pure search engine (`state/search.ts`), with live facet
// suggestions, filter chips, batch selection/visibility actions, and clickable
// declaration rows. Shares 100 % of its query evaluation with Spotlight (`Ctrl+K`).

import { clauseLabel, indexOf, searchResultsOf } from "../../state/selectors.ts";
import type { Event, State } from "../../state/store.ts";
import type { Decl, SearchField, SearchOp } from "../../state/types.ts";
import { el } from "../../ui/dom.ts";
import type { WindowView } from "./view.ts";

const FIELDS: readonly { field: SearchField; label: string }[] = [
  { field: "group", label: "group" },
  { field: "type", label: "type" },
  { field: "kind", label: "kind" },
  { field: "uses", label: "uses (material / section)" },
  { field: "role", label: "role" },
  { field: "id", label: "id / @tag / a..b" },
  { field: "param", label: "param (e.g. E>2e8)" },
  { field: "stage", label: "stage" },
  { field: "visibility", label: "visibility" },
  { field: "name", label: "name" },
  { field: "any", label: "any text" },
];

const OPS: readonly SearchOp[] = ["contains", "=", "!=", ">", ">=", "<", "<="];

const RESULT_CAP = 120;

function rowDetail(s: State, d: Decl): string {
  const parts: string[] = [d.type];
  if (d.tag !== null) parts.push(`@${d.tag}`);
  if (d.element?.femId !== null && d.element?.femId !== undefined) parts.push(`#${d.element.femId}`);
  if (d.element?.groups.length) {
    parts.push(d.element.groups.map((g) => s.decls[g]?.name ?? g).join(", "));
  }
  if (d.group) parts.push(`${d.group.count} elements`);
  return parts.join(" · ");
}

export const view: WindowView = (s, act) => {
  const body = el("div", "window-list search-window");
  const model = s.artifacts.model;
  if (!model || model.status !== "ready") {
    body.append(el("div", "window-note", "no model is open"));
    return body;
  }

  const res = searchResultsOf(s);
  const idx = indexOf(s);

  // 1. Free-text query bar
  const bar = el("div", "search-bar");
  const input = el("input", "search-input");
  input.type = "text";
  input.value = s.search.text;
  input.placeholder = "Search: beams and dispBeamColumn · uses:Steel · E>2e8 · not hidden";
  input.dataset["focusKey"] = "search-text";
  input.addEventListener("input", () => act({ type: "setSearchText", text: input.value }));
  bar.append(input);

  if (s.search.text || s.search.clauses.length > 0) {
    const clearBtn = el("button", "pin", "clear");
    clearBtn.title = "clear search query and filter chips";
    clearBtn.addEventListener("click", () => act({ type: "clearSearch" }));
    bar.append(clearBtn);
  }
  body.append(bar);

  // 2. Active filter chips
  if (s.search.clauses.length > 0) {
    const chips = el("div", "search-chips");
    s.search.clauses.forEach((c, i) => {
      const chip = el("span", "search-chip");
      if (i > 0) chip.append(el("span", "search-chip-join", c.join.toUpperCase()));
      chip.append(el("span", "search-chip-text", clauseLabel(c)));
      const rm = el("button", "search-chip-rm", "×");
      rm.title = "remove filter";
      rm.addEventListener("click", () => act({ type: "removeSearchClause", id: c.id }));
      chip.append(rm);
      chips.append(chip);
    });
    body.append(chips);
  }

  // 3. Visual clause builder
  const builder = el("div", "search-builder");
  const joinSel = el("select", "search-select");
  for (const j of ["and", "or"] as const) {
    const o = el("option", undefined, j.toUpperCase());
    o.value = j;
    joinSel.append(o);
  }
  const negLabel = el("label", "search-neg");
  const negCheck = el("input");
  negCheck.type = "checkbox";
  negLabel.append(negCheck, el("span", undefined, "NOT"));

  const fieldSel = el("select", "search-select");
  for (const f of FIELDS) {
    const o = el("option", undefined, f.label);
    o.value = f.field;
    fieldSel.append(o);
  }

  const opSel = el("select", "search-select");
  for (const op of OPS) {
    const o = el("option", undefined, op);
    o.value = op;
    opSel.append(o);
  }

  const valInput = el("input", "search-val-input");
  valInput.type = "text";
  valInput.placeholder = "value (e.g. Arch, dispBeamColumn, E=2e11)";
  valInput.setAttribute("list", "search-facet-suggestions");

  const datalist = el("datalist");
  datalist.id = "search-facet-suggestions";
  const suggestions = new Set<string>([
    ...idx.facets.groups,
    ...idx.facets.types,
    ...idx.facets.kinds,
    ...idx.facets.materialsAndSections,
    ...idx.facets.roles,
    ...idx.facets.paramKeys.map((k) => `${k}=`),
  ]);
  for (const sug of [...suggestions].slice(0, 80)) {
    const opt = el("option");
    opt.value = sug;
    datalist.append(opt);
  }

  const addClause = () => {
    const raw = valInput.value.trim();
    if (!raw) return;
    const field = fieldSel.value as SearchField;
    let op = opSel.value as SearchOp;
    let key: string | undefined;
    let value = raw;
    if (field === "param") {
      const m = /^([A-Za-z0-9_.-]+)\s*(!=|>=|<=|=|>|<)\s*(.+)$/.exec(raw);
      if (m) {
        key = m[1]!;
        op = m[2] as SearchOp;
        value = m[3]!.trim();
      }
    }
    act({
      type: "addSearchClause",
      clause: {
        join: joinSel.value as "and" | "or",
        negate: negCheck.checked,
        field,
        op,
        ...(key !== undefined ? { key } : {}),
        value,
      },
    });
    valInput.value = "";
  };

  valInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      addClause();
    }
  });

  const addBtn = el("button", "pin", "+ filter");
  addBtn.title = "add filter clause";
  addBtn.addEventListener("click", addClause);

  builder.append(joinSel, negLabel, fieldSel, opSel, valInput, datalist, addBtn);
  body.append(builder);

  // 4. Summary & Batch Actions
  const summaryRow = el("div", "search-summary");
  summaryRow.append(
    el(
      "span",
      "search-count",
      `${res.total.toLocaleString()} match${res.total === 1 ? "" : "es"} · ${res.elementCount.toLocaleString()} element${res.elementCount === 1 ? "" : "s"}`,
    ),
  );

  const actions = el("div", "search-actions");
  const matchedPaths = res.matched.map((d) => d.path);
  const batchTargets = res.elementDecls.length > 0 ? res.elementDecls : matchedPaths;
  const unhideTargets = [...new Set([...matchedPaths, ...res.elementDecls])];
  const hasMatches = matchedPaths.length > 0;

  const mkBtn = (label: string, title: string, onClick: () => void) => {
    const b = el("button", "pin", label);
    b.title = title;
    b.disabled = !hasMatches;
    b.addEventListener("click", onClick);
    return b;
  };

  actions.append(
    mkBtn("select & frame", "select all matching elements/declarations and frame them in the 3D view", () => {
      act({ type: "selectDecls", decls: batchTargets });
      act({ type: "frameSelection" });
    }),
    mkBtn("+ selection", "add all matching elements/declarations to the current selection", () => {
      act({ type: "selectDecls", decls: batchTargets, add: true });
    }),
    mkBtn("isolate", "isolate matching elements in the 3D view", () => {
      act({ type: "isolate", decls: batchTargets });
    }),
    mkBtn("hide", "hide matching elements in the 3D view", () => {
      act({ type: "setHidden", decls: batchTargets, hidden: true });
    }),
    mkBtn("show", "show matching declarations and elements in the 3D view", () => {
      act({ type: "setHidden", decls: unhideTargets, hidden: false });
    }),
  );
  summaryRow.append(actions);
  body.append(summaryRow);

  // 5. Results list
  if (res.total === 0) {
    body.append(el("div", "window-note", "no declarations match the current query"));
    return body;
  }

  let currentKind = "";
  const shown = res.matched.slice(0, RESULT_CAP);
  for (const d of shown) {
    if (d.kind !== currentKind) {
      currentKind = d.kind;
      const countInKind = res.byKind[d.kind] ?? 0;
      body.append(el("div", "window-section", `${d.kind} (${countInKind.toLocaleString()})`));
    }
    const row = el("div", "window-row clickable");
    row.dataset["decl"] = d.path;
    row.title = `select ${d.path} (Shift+click to add to selection)`;
    row.addEventListener("click", (e) => {
      if ((e as MouseEvent).shiftKey) {
        act({ type: "selectDecls", decls: [d.path], add: true });
      } else {
        act({ type: "select", pick: { decl: d.path, at: null, nodes: null } });
      }
    });
    const head = el("div", "window-row-head");
    head.append(el("span", "window-row-title", d.name), el("span", "node-type", d.type));
    if (d.tag !== null) head.append(el("span", "window-row-decl", `@${d.tag}`));
    row.append(head, el("div", "window-row-detail", rowDetail(s, d)), el("div", "f-src", d.h5));
    body.append(row);
  }
  if (res.total > RESULT_CAP) {
    body.append(el("div", "window-note", `showing first ${RESULT_CAP} of ${res.total.toLocaleString()} matches — refine the query or use batch actions above`));
  }
  return body;
};
