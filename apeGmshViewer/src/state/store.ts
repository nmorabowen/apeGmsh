// The one state store (ADR 0112 D6).
//
// Artifacts load into a single state; views render from it; user actions
// are events that a pure reducer applies (reduce.ts). No view keeps its own
// copy of model data, and no view calls another view: panels dispatch
// events and read selectors (selectors.ts); side effects live in
// src/effects.ts. The types are in types.ts.

import { initialState, reduce } from "./reduce.ts";
import type { Event, State } from "./types.ts";

export type { Event, State } from "./types.ts";
export { initialState, reduce } from "./reduce.ts";

export type Listener = (s: State, prev: State) => void;

export class Store {
  private state: State = initialState;
  private listeners: Listener[] = [];

  get(): State {
    return this.state;
  }

  dispatch(e: Event): void {
    const prev = this.state;
    const next = reduce(prev, e);
    if (next === prev) return;
    this.state = next;
    for (const l of this.listeners) l(next, prev);
  }

  /** Subscribe; the returned function unsubscribes. */
  subscribe(l: Listener): () => void {
    this.listeners = [...this.listeners, l];
    return () => {
      this.listeners = this.listeners.filter((x) => x !== l);
    };
  }
}
