// Empty-state panel: the drop target, a load failure, and the zones a
// reader refused, each as a sentence (never a blank window).

import { failureOf, geometryPairing, refusalsOf } from "../state/selectors.ts";
import type { State, Store } from "../state/store.ts";
import { byId, el } from "../ui/dom.ts";

export function mountEmpty(store: Store): () => void {
  const empty = byId("empty");
  const card = empty.querySelector(".empty-card");
  if (!card) throw new Error("#empty has no .empty-card");
  const render = (s: State, prev: State | null) => {
    if (prev && s.artifacts === prev.artifacts && s.mesh === prev.mesh && s.geometry === prev.geometry && s.particles === prev.particles) return;
    // A geometry file opened alone is something to look at, not an empty view; so are particles read with no model (ADR 0134 D5).
    const particles = s.particles?.status === "ready" && s.particles.stages.length > 0;
    empty.hidden = s.mesh !== null || (s.artifacts.model === null && (geometryPairing(s).draw || particles));
    for (const old of card.querySelectorAll(".empty-error, .empty-refused")) old.remove();
    for (const r of refusalsOf(s)) card.append(el("div", "empty-refused", r));
    const failure = failureOf(s);
    if (failure) card.append(el("div", "empty-error", failure));
  };
  render(store.get(), null);
  return store.subscribe(render);
}
