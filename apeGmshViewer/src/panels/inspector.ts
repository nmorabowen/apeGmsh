// Inspector panel: the definition chain of each pinned declaration, then
// of the selection, from the `inspected` selector. A pin button dispatches
// `inspectorPin` / `inspectorUnpin`.

import { declOfH5, inspected, inspectedBatchOf, inspectedGroupsOf, sourceFailure, sourceFor } from "../state/selectors.ts";
import type { State, Store } from "../state/store.ts";
import type { ChainNode, DeclPath, Field } from "../state/types.ts";
import { byId, el, listen } from "../ui/dom.ts";

const ROLE_LABEL: Record<ChainNode["role"], string> = {
  element: "element",
  geomTransf: "geomTransf",
  beamIntegration: "beamIntegration",
  section: "section",
  uniaxialMaterial: "uniaxialMaterial",
  nDMaterial: "nDMaterial",
};

/** Adds what belongs to one node beside the read facts (the go-to-source button). */
type Decorate = (n: ChainNode, depth: number, head: HTMLElement, box: HTMLElement) => void;

function renderNode(n: ChainNode, decorate: Decorate, depth = 0): HTMLElement {
  const box = el("div", "node");
  const head = el("div", "node-head");
  head.append(el("span", "role", ROLE_LABEL[n.role]), el("span", "node-name", n.name), el("span", "node-type", n.type));
  box.append(head, el("div", "node-src", `name from ${n.nameSource}`));
  decorate(n, depth, head, box);
  if (n.via) {
    const via = el("div", "via");
    via.append("via ", el("b", undefined, `${n.via.label} = ${n.via.value}`), ` from ${n.via.source}`);
    if (n.via.interpreted) via.append(el("span", "tag-interp", "interpreted"));
    if (n.via.note) via.title = n.via.note;
    box.append(via);
  }
  const t = el("table", "fields");
  // Positional params read best as one row: "[0] v0 · [1] v1 · ...".
  const isParam = (f: Field) => /^params\[\d+\]$/.test(f.label);
  const params = n.fields.filter(isParam);
  const shown: Field[] = n.fields.filter((f) => !isParam(f));
  if (params.length) {
    shown.splice(Math.min(2, shown.length), 0, {
      label: "params",
      value: params.map((f) => `[${f.label.slice(7, -1)}] ${f.value}`).join(" · "),
      source: `${n.path}@params[0..${params.length - 1}] (strings from @params_str)`,
      interpreted: false,
      note: "positional; the file carries no parameter names",
    });
  }
  for (const f of shown) {
    const tr = el("tr");
    const v = el("td", "f-value", f.value);
    if (f.interpreted) v.append(el("span", "tag-interp", "interpreted"));
    v.append(el("div", "f-src", f.source));
    if (f.note) tr.title = f.note;
    tr.append(el("td", "f-label", f.label), v);
    t.append(tr);
  }
  box.append(t);
  if (n.children.length) {
    const kids = el("div", "children");
    for (const c of n.children) kids.append(renderNode(c, decorate, depth + 1));
    box.append(kids);
  }
  return box;
}

export function mountInspector(store: Store): () => void {
  const inspector = byId("inspector");
  let unlisten: (() => void)[] = [];
  const render = (s: State, prev: State | null) => {
    if (
      prev &&
      s.selection === prev.selection &&
      s.inspector === prev.inspector &&
      s.visibility === prev.visibility &&
      s.decls === prev.decls &&
      s.source === prev.source &&
      s.artifacts === prev.artifacts
    ) return;
    for (const u of unlisten) u();
    unlisten = [];
    const groups = inspectedGroupsOf(s);
    const batch = inspectedBatchOf(s);
    const shown = inspected(s);
    inspector.hidden = groups.length === 0 && batch === null && shown.length === 0;
    const parts: HTMLElement[] = [];

    for (const g of groups) {
      const head = el("h3", undefined, g.pinned ? "Pinned group" : "Group");
      const pin = el("button", "pin" + (g.pinned ? " pinned" : ""), g.pinned ? "unpin" : "pin");
      pin.title = g.decl;
      unlisten.push(listen(pin, "click", () => store.dispatch(g.pinned ? { type: "inspectorUnpin", decl: g.decl } : { type: "inspectorPin", decl: g.decl })));
      head.append(pin);
      parts.push(head);
      const box = el("div", "node");
      const nodeHead = el("div", "node-head");
      nodeHead.append(el("span", "role", g.kind), el("span", "node-name", g.name), el("span", "node-type", `${g.count} elements`));
      const where = sourceFor(s, g.decl);
      const src = el("button", "source", where.ok ? where.label : "source");
      src.title = where.ok ? `open ${where.source.file}:${where.source.line} in the editor` : where.reason;
      if (where.ok) unlisten.push(listen(src, "click", () => store.dispatch({ type: "requestSource", decl: g.decl })));
      else src.disabled = true;
      nodeHead.append(src);
      box.append(nodeHead, el("div", "node-src", g.h5));
      const failed = sourceFailure(s, g.decl);
      if (failed) box.append(el("div", "source-failed", `go to source: ${failed}`));
      const t = el("table", "fields");
      for (const [label, val] of [["dimension", `${g.dim}-D`], ["tag", String(g.tag)], ["elements", String(g.count)]] as const) {
        const tr = el("tr");
        tr.append(el("td", "f-label", label), el("td", "f-value", val));
        t.append(tr);
      }
      box.append(t);
      const actions = el("div", "inspector-actions");
      const iso = el("button", "pin", "isolate");
      unlisten.push(listen(iso, "click", () => store.dispatch({ type: "isolate", decls: [g.decl] })));
      const hideBtn = el("button", "pin", g.hidden ? "show" : "hide");
      unlisten.push(listen(hideBtn, "click", () => store.dispatch({ type: "setHidden", decls: [g.decl], hidden: !g.hidden })));
      actions.append(iso, hideBtn);
      box.append(actions);
      parts.push(box);
    }

    if (batch) {
      const head = el("h3", undefined, "Batch selection");
      const clearBtn = el("button", "pin", "clear");
      unlisten.push(listen(clearBtn, "click", () => store.dispatch({ type: "clearSelection" })));
      head.append(clearBtn);
      parts.push(head);
      const box = el("div", "node");
      const nodeHead = el("div", "node-head");
      nodeHead.append(
        el("span", "role", batch.kinds.join(" / ")),
        el("span", "node-name", `${batch.count.toLocaleString()} selected`),
        el("span", "node-type", `${batch.elementCount.toLocaleString()} elements`),
      );
      box.append(nodeHead);
      const t = el("table", "fields");
      if (batch.types.length) {
        const tr = el("tr");
        tr.append(el("td", "f-label", "types"), el("td", "f-value", batch.types.join(" · ")));
        t.append(tr);
      }
      if (batch.groups.length) {
        const tr = el("tr");
        tr.append(el("td", "f-label", "groups"), el("td", "f-value", batch.groups.join(" · ")));
        t.append(tr);
      }
      box.append(t);
      const actions = el("div", "inspector-actions");
      const frameBtn = el("button", "pin", "frame");
      unlisten.push(listen(frameBtn, "click", () => store.dispatch({ type: "frameSelection" })));
      const isoBtn = el("button", "pin", "isolate");
      unlisten.push(listen(isoBtn, "click", () => store.dispatch({ type: "isolate", decls: [...batch.decls] })));
      const hideBtn = el("button", "pin", "hide");
      unlisten.push(listen(hideBtn, "click", () => store.dispatch({ type: "setHidden", decls: [...batch.decls], hidden: true })));
      const showBtn = el("button", "pin", "show");
      unlisten.push(listen(showBtn, "click", () => store.dispatch({ type: "setHidden", decls: [...batch.decls], hidden: false })));
      actions.append(frameBtn, isoBtn, hideBtn, showBtn);
      box.append(actions);
      parts.push(box);
    }

    for (const { decl, pinned, chain } of shown) {
      const head = el("h3", undefined, pinned ? "Pinned" : "Definition chain");
      const pin = el("button", "pin" + (pinned ? " pinned" : ""), pinned ? "unpin" : "pin");
      pin.title = decl;
      unlisten.push(listen(pin, "click", () => store.dispatch(pinned ? { type: "inspectorUnpin", decl } : { type: "inspectorPin", decl })));
      head.append(pin);
      parts.push(head);
      if (chain.problems.length) {
        const p = el("div", "problems");
        for (const msg of chain.problems) p.append(el("div", undefined, msg));
        parts.push(p);
      }
      // Go to source (ADR 0112 D3) on every declaration of the chain: the root
      // is the inspected decl; a linked node is found by its HDF5 path.
      const decorate: Decorate = (n, depth, nodeHead, box) => {
        const d: DeclPath | null = depth === 0 ? decl : declOfH5(s, n.path);
        if (d === null) return;
        const where = sourceFor(s, d);
        const src = el("button", "source", where.ok ? where.label : "source");
        src.title = where.ok ? `open ${where.source.file}:${where.source.line} in the editor` : where.reason;
        if (where.ok) unlisten.push(listen(src, "click", () => store.dispatch({ type: "requestSource", decl: d })));
        else src.disabled = true;
        nodeHead.append(src);
        const failed = sourceFailure(s, d);
        if (failed) box.append(el("div", "source-failed", `go to source: ${failed}`));
      };
      parts.push(renderNode(chain.root, decorate));
    }
    inspector.replaceChildren(...parts);
  };
  render(store.get(), null);
  const unsubscribe = store.subscribe(render);
  return () => {
    unsubscribe();
    for (const u of unlisten) u();
  };
}
