// Offers panel (ADR 0128 D5): the views agents offered through `--present`,
// newest first, each with its name, caption and the time it arrived. The
// window never changes on its own: the view moves only when the human clicks
// Apply here, or Back. A refused offer says why and has no working Apply. An
// offer naming files outside the open set says it will open them. The panel
// dispatches the clicks and reads the store, nothing else; the offer runner
// (src/spec/snapshot.ts) judges and carries them out. Its box sits at the top
// of the right rail, and takes no room with nothing on offer.

import type { State, Store } from "../state/store.ts";
import type { Offer } from "../state/types.ts";
import { byId, el, listen } from "../ui/dom.ts";

const STATUS_LABEL: Readonly<Record<Offer["status"], string>> = {
  pending: "offered",
  applying: "applying",
  applied: "applied",
  refused: "refused",
};

const baseName = (path: string): string => path.split(/[\\/]/).pop() ?? path;

/** The arrival time as the clock on the wall shows it. */
export function arrivalLabel(at: number): string {
  const d = new Date(at);
  const two = (n: number) => String(n).padStart(2, "0");
  return `${two(d.getHours())}:${two(d.getMinutes())}:${two(d.getSeconds())}`;
}

/**
 * A collapsed rail hides this panel, and an offer must not go unseen (T's
 * ruling on #2654): the rail's toggle carries how many are on offer. The
 * stylesheet shows it only while the rail is collapsed. Mount it after the
 * collapsible chrome, which makes the toggle.
 */
export function mountOfferBadge(store: Store): () => void {
  const toggle = byId("rail-toggle");
  const render = (s: State) => {
    const n = s.offers.list.filter((o) => o.status === "pending").length;
    if (n > 0) toggle.dataset["offers"] = n === 1 ? "1 offer" : `${n} offers`;
    else delete toggle.dataset["offers"];
  };
  render(store.get());
  const unsubscribe = store.subscribe((s, prev) => {
    if (s.offers !== prev.offers) render(s);
  });
  return () => {
    unsubscribe();
    delete toggle.dataset["offers"];
  };
}

export function mountOffers(store: Store): () => void {
  const panel = el("aside");
  panel.id = "offers";
  panel.hidden = true;
  byId("rail").prepend(panel);
  let unlisten: (() => void)[] = [];
  const render = (s: State, prev: State | null) => {
    if (prev && s.offers === prev.offers) return;
    for (const u of unlisten) u();
    unlisten = [];
    const { list, back, request, notice } = s.offers;
    panel.hidden = list.length === 0 && back.length === 0;
    if (panel.hidden) {
      panel.replaceChildren();
      return;
    }
    const head = el("h3", undefined, "Offers");
    head.append(el("span", "legend-count", String(list.length)));
    const backBtn = el("button", "offer-back", "Back");
    backBtn.type = "button";
    backBtn.title = back.length ? "Return to the view from before the last applied offer" : "No applied offer to go back from";
    backBtn.disabled = back.length === 0 || request !== null;
    unlisten.push(listen(backBtn, "click", () => store.dispatch({ type: "backOffer" })));
    head.append(backBtn);
    const parts: HTMLElement[] = [head];
    if (notice) parts.push(el("div", "offer-reason", notice));
    for (const o of list) {
      const row = el("div", `offer-row ${o.status}`);
      row.dataset["offer"] = String(o.id);
      const top = el("div", "offer-top");
      top.append(el("span", "offer-name", o.name), el("span", "offer-time", arrivalLabel(o.arrivedAt)), el("span", `offer-status ${o.status}`, STATUS_LABEL[o.status]));
      const apply = el("button", "offer-apply", "Apply");
      apply.type = "button";
      apply.disabled = o.status === "refused" || request !== null;
      apply.title = o.status === "refused" ? "A refused offer cannot be applied" : "Show this view";
      unlisten.push(listen(apply, "click", () => store.dispatch({ type: "applyOffer", id: o.id })));
      top.append(apply);
      row.append(top);
      if (o.caption) row.append(el("div", "offer-caption", o.caption));
      if (o.reason) row.append(el("div", "offer-reason", o.reason));
      else if (o.opens.length) {
        const opens = el("div", "offer-note", `Applying opens ${o.opens.map(baseName).join(" and ")}.`);
        opens.title = o.opens.join("\n");
        row.append(opens);
      }
      if (o.ignored && !o.reason) row.append(el("div", "offer-note", o.ignored));
      parts.push(row);
    }
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
