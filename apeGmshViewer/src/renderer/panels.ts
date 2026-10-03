// DOM panels: header, warning banner, legend and the inspector. Each one
// renders from the store state alone (D6).

import type { ChainNode, Field } from "../chain/resolve.ts";
import type { State, Store } from "../state/store.ts";

const el = <K extends keyof HTMLElementTagNameMap>(
  tag: K,
  cls?: string,
  text?: string,
): HTMLElementTagNameMap[K] => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
};

const byId = (id: string): HTMLElement => {
  const e = document.getElementById(id);
  if (!e) throw new Error(`#${id} is missing from index.html`);
  return e;
};

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

export function mountPanels(store: Store): void {
  const header = byId("header");
  const banner = byId("banner");
  const legend = byId("legend");
  const inspector = byId("inspector");
  const empty = byId("empty");

  store.subscribe((s: State, prev: State) => {
    if (s.model !== prev.model || s.error !== prev.error) {
      empty.hidden = s.model !== null;
      const card = empty.querySelector(".empty-card")!;
      card.querySelector(".empty-error")?.remove();
      if (s.error) card.append(el("div", "empty-error", s.error));

      header.hidden = s.model === null;
      if (s.model && s.mesh) {
        const m = s.model;
        const cells = m.blocks.reduce((a, b) => a + b.ids.length, 0);
        byId("file-name").textContent = m.path.split(/[\\/]/).pop() ?? m.path;
        byId("file-stats").textContent =
          `${m.nodeIds.length.toLocaleString()} nodes · ${cells.toLocaleString()} cells` +
          (s.mesh.counts.opsOnly ? ` (+${s.mesh.counts.opsOnly.toLocaleString()} OpenSees-only)` : "") +
          ` · neutral ${m.neutralVersion}` +
          (m.opensees ? ` · opensees ${m.opensees.version}` : " · no /opensees");
        document.title = `${byId("file-name").textContent} - apeGmshViewer`;
      }

      const warnings = [...(s.model?.warnings ?? []), ...(s.mesh?.warnings ?? [])];
      banner.hidden = warnings.length === 0;
      banner.replaceChildren(...warnings.map((w) => el("div", undefined, w)));

      legend.hidden = !s.mesh;
      legend.replaceChildren(el("h3", undefined, "Physical groups"));
      for (const e of s.mesh?.legend ?? []) {
        const row = el("div", "legend-row");
        const chip = el("span", "chip");
        chip.style.background = `rgb(${e.color.map((c) => Math.round(c * 255)).join(",")})`;
        row.append(chip, el("span", undefined, e.name), el("span", "legend-count", e.elements.toLocaleString()));
        legend.append(row);
      }
    }
    if (s.chain !== prev.chain) {
      inspector.hidden = s.chain === null;
      if (s.chain) {
        const parts: HTMLElement[] = [el("h3", undefined, "Definition chain")];
        if (s.chain.problems.length) {
          const p = el("div", "problems");
          for (const msg of s.chain.problems) p.append(el("div", undefined, msg));
          parts.push(p);
        }
        parts.push(renderNode(s.chain.root));
        inspector.replaceChildren(...parts);
      }
    }
  });
}
