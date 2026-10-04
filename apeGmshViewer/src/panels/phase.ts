// Phase panel: the phase axis (ADR 0112 D7: geometry -> mesh -> ...) as a
// row of buttons, shown when there is more than one phase to choose. A click
// dispatches `setPhase`; the viewport draws the phase the state names.

import type { State, Store } from "../state/store.ts";
import type { PhaseKey } from "../state/types.ts";
import { byId, el, listen } from "../ui/dom.ts";

const label = (k: PhaseKey): string =>
  k.kind === "stage" ? `stage ${k.index + 1}` : k.kind === "results" ? `results (step ${k.step})` : k.kind;

const same = (a: PhaseKey | null, b: PhaseKey): boolean => a !== null && JSON.stringify(a) === JSON.stringify(b);

export function mountPhase(store: Store): () => void {
  const bar = byId("phase");
  let unlisten: (() => void)[] = [];
  const render = (s: State, prev: State | null) => {
    if (prev && s.phase === prev.phase) return;
    for (const u of unlisten) u();
    unlisten = [];
    bar.hidden = s.phase.axis.length < 2;
    const buttons = s.phase.axis.map((k) => {
      const b = el("button", same(s.phase.at, k) ? "phase active" : "phase", label(k));
      unlisten.push(listen(b, "click", () => store.dispatch({ type: "setPhase", at: k })));
      return b;
    });
    bar.replaceChildren(...buttons);
  };
  render(store.get(), null);
  const unsubscribe = store.subscribe(render);
  return () => {
    unsubscribe();
    for (const u of unlisten) u();
  };
}
