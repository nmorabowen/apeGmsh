// The offers slice of the reducer (state/reduce.ts delegates here): VA5's
// list of view offers, the click on one, and the way back (ADR 0128 D5).
// INV-5: an offer event touches the offers and nothing else of the state.

import type { OffersEvent } from "../types/events/offers.ts";
import type { Offer, OffersState } from "../types.ts";

/** VA5 (ADR 0128 D5): the longest offer list; the next offer drops the oldest. */
export const MAX_OFFERS = 8;

export const initialOffers: OffersState = { list: [], seq: 0, request: null, requestSeq: 0, back: [], notice: null };

/** A listed offer, or a loud error: an event for an offer never listed is a bug. */
function offerOf(o: OffersState, id: number, who: string): Offer {
  const x = o.list.find((x) => x.id === id);
  if (!x) throw new RangeError(`${who}: offer ${JSON.stringify(id)} is not on the list`);
  return x;
}

/** The offers with one offer changed. A click already running can outlive its offer (the cap dropped it): nothing to change then. */
function withOffer(o: OffersState, id: number, change: Partial<Offer>): OffersState {
  if (!o.list.some((x) => x.id === id)) {
    if (id > 0 && id <= o.seq) return o;
    throw new RangeError(`no offer ${JSON.stringify(id)} was ever listed`);
  }
  return { ...o, list: o.list.map((x) => (x.id === id ? { ...x, ...change } : x)) };
}

/** The offers after `e`: the same object when nothing changed. */
export function reduceOffers(o: OffersState, e: OffersEvent): OffersState {
  switch (e.type) {
    case "offerReceived": {
      if (!Number.isFinite(e.at)) throw new RangeError(`offerReceived: ${JSON.stringify(e.at)} is not a time`);
      let seq = o.seq;
      const arrived = e.offers.map((x): Offer => ({ ...x, id: ++seq, arrivedAt: e.at, status: x.refused ? "refused" : "pending", reason: x.refused ?? null, opens: [], viewRefused: Boolean(x.refused) }));
      // Newest first; a spec's shots keep their own order within one arrival.
      return { ...o, seq, list: [...arrived, ...o.list].slice(0, MAX_OFFERS) };
    }
    case "offerAssessed": {
      const x = offerOf(o, e.id, "offerAssessed");
      if (x.status === "applying" || x.viewRefused) return o;
      const status = x.status === "applied" ? "applied" : e.refused === null ? "pending" : "refused";
      const reason = status === "refused" ? e.refused : null;
      const sameOpens = x.opens.length === e.opens.length && x.opens.every((p, i) => p === e.opens[i]);
      if (x.status === status && x.reason === reason && sameOpens) return o;
      return withOffer(o, e.id, { status, reason, opens: e.opens });
    }
    case "applyOffer": {
      const x = offerOf(o, e.id, "applyOffer");
      if (x.status === "refused") throw new Error(`applyOffer: offer ${e.id} is refused and cannot be applied: ${x.reason}`);
      // One click at a time: a click while another runs does nothing.
      if (o.request) return o;
      const seq = o.requestSeq + 1;
      const t = withOffer(o, e.id, { status: "applying" });
      return { ...t, requestSeq: seq, request: { kind: "apply", id: e.id, seq }, notice: null };
    }
    case "backOffer": {
      if (o.back.length === 0) throw new Error("backOffer: no applied offer to go back from");
      if (o.request) return o;
      const seq = o.requestSeq + 1;
      return { ...o, requestSeq: seq, request: { kind: "back", seq }, notice: null };
    }
    case "offerApplied": {
      const t = withOffer(o, e.id, { status: "applied", reason: null });
      const back = e.memo ? [{ offer: e.id, memo: e.memo }, ...t.back] : t.back;
      return { ...t, request: null, back };
    }
    case "offerRefused": {
      if (e.reason === "") throw new RangeError("offerRefused: a refusal names its reason");
      const t = withOffer(o, e.id, { status: "refused", reason: e.reason, viewRefused: true });
      return { ...t, request: null };
    }
    case "offerBackDone": {
      const [top, ...back] = o.back;
      if (!top) throw new Error("offerBackDone: no applied offer to go back from");
      // Its offer is on offer again, unless an earlier click on it is still on the way back.
      const t = back.some((b) => b.offer === top.offer) ? o : withOffer(o, top.offer, { status: "pending", reason: null });
      return { ...t, request: null, back, notice: e.notice };
    }
    default: {
      const unknown: never = e;
      throw new Error(`reduceOffers: unknown event ${JSON.stringify(unknown)}`);
    }
  }
}
