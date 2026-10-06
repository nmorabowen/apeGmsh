// Legend panel: the colour mode (by physical group, or by the structural
// role the file records) and its rows. A click on a group's row hides or
// shows it (`setHidden`); "show all" clears the hidden set. It dispatches
// events and reads selectors, nothing else.

import { legendOf, isHidden, roleLegendOf } from "../state/selectors.ts";
import type { State, Store } from "../state/store.ts";
import type { ColourBy, LegendEntry } from "../state/types.ts";
import { byId, el, listen } from "../ui/dom.ts";

const MODES: readonly { by: ColourBy; label: string }[] = [
  { by: "group", label: "by group" },
  { by: "role", label: "by role" },
];

export function mountLegend(store: Store): () => void {
  const legend = byId("legend");
  let unlisten: (() => void)[] = [];
  const render = (s: State, prev: State | null) => {
    if (prev && s.mesh === prev.mesh && s.visibility.hidden === prev.visibility.hidden && s.visibility.colourBy === prev.visibility.colourBy) return;
    for (const u of unlisten) u();
    unlisten = [];
    legend.hidden = s.mesh === null;
    const byRole = s.visibility.colourBy === "role";
    const head = el("h3", undefined, byRole ? "Structural roles" : "Physical groups");
    legend.replaceChildren(head);
    if (!byRole && s.visibility.hidden.length) {
      const all = el("button", "legend-all", "show all");
      unlisten.push(listen(all, "click", () => store.dispatch({ type: "showAll" })));
      head.append(all);
    }
    const modes = el("div", "legend-mode");
    for (const m of MODES) {
      const b = el("button", s.visibility.colourBy === m.by ? "on" : "", m.label);
      unlisten.push(listen(b, "click", () => store.dispatch({ type: "setColourBy", by: m.by })));
      modes.append(b);
    }
    legend.append(modes);
    let rows: LegendEntry[];
    if (byRole) {
      const r = roleLegendOf(s);
      rows = r?.legend ?? [];
      if (r && !r.fileHasRoles) {
        legend.append(el("div", "legend-note", "The file records no structural role (no role attribute on its physical groups), so nothing is coloured by role; every element is drawn as unassigned."));
      }
    } else rows = legendOf(s);
    for (const e of rows) {
      const hidden = !byRole && isHidden(s, e.decl);
      const row = el("div", "legend-row" + (hidden ? " hidden-group" : "") + (!byRole && e.decl ? " clickable" : ""));
      const chip = el("span", "chip" + (e.cue ? ` ${e.cue}` : ""));
      chip.style.backgroundColor = `rgb(${e.color.map((c) => Math.round(c * 255)).join(",")})`;
      row.append(chip, el("span", undefined, e.name), el("span", "legend-count", e.elements.toLocaleString()));
      if (!byRole && e.decl) {
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
