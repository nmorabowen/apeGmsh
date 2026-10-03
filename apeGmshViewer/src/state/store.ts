// The one state store (ADR 0112 D6).
//
// Artifacts load into a single state; views render from it; user actions
// are events that a pure reducer applies. No view keeps its own copy of
// model data, and no view calls another view.

import { resolveChain, type Chain, type ElementRef } from "../chain/resolve.ts";
import type { MeshBuffers } from "../mesh/build.ts";
import type { ModelFile } from "../model/types.ts";

export interface State {
  model: ModelFile | null;
  mesh: MeshBuffers | null;
  selection: ElementRef | null;
  chain: Chain | null;
  /** a load failure, shown instead of the model */
  error: string | null;
}

export type Event =
  | { type: "model/loaded"; model: ModelFile; mesh: MeshBuffers }
  | { type: "model/failed"; error: string }
  | { type: "select"; ref: ElementRef }
  | { type: "clear-selection" };

export const initialState: State = { model: null, mesh: null, selection: null, chain: null, error: null };

export function reduce(s: State, e: Event): State {
  switch (e.type) {
    case "model/loaded":
      return { model: e.model, mesh: e.mesh, selection: null, chain: null, error: null };
    case "model/failed":
      return { ...initialState, error: e.error };
    case "select":
      if (!s.model) return s;
      return { ...s, selection: e.ref, chain: resolveChain(s.model, e.ref) };
    case "clear-selection":
      return { ...s, selection: null, chain: null };
  }
}

export type Listener = (s: State, prev: State) => void;

export class Store {
  private state: State = initialState;
  private readonly listeners: Listener[] = [];

  get(): State {
    return this.state;
  }

  dispatch(e: Event): void {
    const prev = this.state;
    this.state = reduce(prev, e);
    for (const l of this.listeners) l(this.state, prev);
  }

  subscribe(l: Listener): void {
    this.listeners.push(l);
  }
}
