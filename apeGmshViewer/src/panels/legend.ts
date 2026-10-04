// Legend panel: the colour groups. A click on a group's row hides or shows
// it (`setHidden`); "show all" clears the hidden set. It dispatches events
// and reads selectors, nothing else.

import { isHidden, legendOf } from "../state/selectors.ts";
import type { State, Store } from "../state/store.ts";
import { byId, el, listen } from "../ui/dom.ts";

export function mountLegend(store: Store): () => void {
  const legend = byId("legend");
  let unlisten: (() => void)[] = [];
  const render = (s: State, prev: State | null) => {
    if (prev && s.mesh === prev.mesh && s.visibility.hidden === prev.visibility.hidden) return;
    for (const u of unlisten) u();
    unlisten = [];
    legend.hidden = s.mesh === null;
    const head = el("h3", undefined, "Physical groups");
    legend.replaceChildren(head);
    if (s.visibility.hidden.length) {
      const all = el("button", "legend-all", "show all");
      unlisten.push(listen(all, "click", () => store.dispatch({ type: "showAll" })));
      head.append(all);
    }
    for (const e of legendOf(s)) {
      const row = el("div", "legend-row" + (isHidden(s, e.decl) ? " hidden-group" : "") + (e.decl ? " clickable" : ""));
      const chip = el("span", "chip");
      chip.style.background = `rgb(${e.color.map((c) => Math.round(c * 255)).join(",")})`;
      row.append(chip, el("span", undefined, e.name), el("span", "legend-count", e.elements.toLocaleString()));
      if (e.decl) {
        const decl = e.decl;
        row.title = "click to hide or show";
        unlisten.push(listen(row, "click", () => store.dispatch({ type: "setHidden", decls: [decl], hidden: !isHidden(store.get(), decl) })));
      }
      legend.append(row);
    }
  };
  render(store.get(), null);
  const unsubscribe = store.subscribe(render);
  return () => {
    unsubscribe();
    for (const u of unlisten) u();
  };
}
