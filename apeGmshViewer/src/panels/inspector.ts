// Inspector panel: the definition chain of each pinned declaration, then
// of the selection, from the `inspected` selector. A pin button dispatches
// `inspectorPin` / `inspectorUnpin`.

import { inspected } from "../state/selectors.ts";
import type { State, Store } from "../state/store.ts";
import type { ChainNode, Field } from "../state/types.ts";
import { byId, el, listen } from "../ui/dom.ts";

const ROLE_LABEL: Record<ChainNode["role"], string> = {
  element: "element",
  geomTransf: "geomTransf",
  beamIntegration: "beamIntegration",
  section: "section",
  uniaxialMaterial: "uniaxialMaterial",
  nDMaterial: "nDMaterial",
};

function renderNode(n: ChainNode): HTMLElement {
  const box = el("div", "node");
  const head = el("div", "node-head");
  head.append(el("span", "role", ROLE_LABEL[n.role]), el("span", "node-name", n.name), el("span", "node-type", n.type));
  box.append(head, el("div", "node-src", `name from ${n.nameSource}`));
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
    for (const c of n.children) kids.append(renderNode(c));
    box.append(kids);
  }
  return box;
}

export function mountInspector(store: Store): () => void {
  const inspector = byId("inspector");
  let unlisten: (() => void)[] = [];
  const render = (s: State, prev: State | null) => {
    if (prev && s.selection === prev.selection && s.inspector === prev.inspector && s.decls === prev.decls) return;
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
      parts.push(renderNode(chain.root));
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
