// Inspector panel: the definition chain of each pinned declaration, then
// of the selection, from the `inspected` selector. A pin button dispatches
// `inspectorPin` / `inspectorUnpin`.

import { declOfH5, inspected, sourceFailure, sourceFor } from "../state/selectors.ts";
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
      s.decls === prev.decls &&
      s.source === prev.source &&
      s.artifacts === prev.artifacts
    ) return;
    for (const u of unlisten) u();
    unlisten = [];
    const shown = inspected(s);
    inspector.hidden = shown.length === 0;
    const parts: HTMLElement[] = [];
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
