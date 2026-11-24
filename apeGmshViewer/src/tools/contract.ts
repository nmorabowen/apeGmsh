// The tool contract (ADR 0137 D9-D11, H1). A tool is written once and does not know where it is
// placed: in H1 a separate sandboxed tool window; later a tile (H4), the model window's dock (H5)
// or a card. The model window owns the store and the viewport. A tool sees a small, one-way,
// serializable `ToolContext`, changes anything only by sending a `ToolEvent` back, and (from H2)
// asks main for the reduced values it draws. No large array travels in context (D10).
//
// This module holds types only, so a tool bundle and the model window's page share it without
// either importing the other.

/** D11: what a tool costs to update. Only a light tool may later mount in the model window's page. */
export type ToolCost = "light" | "heavy";

/** The model window's results step, as a tool sees it; null when no results stage is shown. */
export interface ToolStep {
  /** the results stage's id and the label every site shows */
  stage: string;
  stageName: string;
  /** the step shown (0-based), and the stage's step count */
  step: number;
  steps: number;
  /** playback is running: a tool may lower its rate (ADR 0137 D10) */
  playing: boolean;
  /** the stage's nodal components, which a tool may ask for (`ToolRequest`) */
  components: string[];
}

/** The small context the model window sends a tool: plain data, a few hundred bytes. */
export interface ToolContext {
  /** the model's file name (not its path), or null with none open */
  model: string | null;
  step: ToolStep | null;
  /** what is picked in the model window: counts, and at most `TARGETS_SHOWN` of each */
  targets: { elements: number; nodes: number; decls: string[]; nodeTags: number[] };
  /** the tool's own parameters (none in H1) */
  params: Record<string, string | number | boolean>;
  theme: "dark" | "light";
}

/** How many picked declarations and node tags a context carries; the counts give the rest. */
export const TARGETS_SHOWN = 64;

/**
 * What a tool may ask the model window's store to do. The model window checks each against the
 * state it holds before dispatching it (state/tools.ts `checkToolEvent`); one it refuses is
 * dropped. The list grows with the tools that need more.
 */
export type ToolEvent = { type: "setResultStep"; step: number };

/**
 * ADR 0137 D10 (H2b): data a tool asks main for. It names no file: main reads the model window's
 * results file. The answer is the reduced values the tool draws, never the step's arrays.
 */
export type ToolRequest = { kind: "nodeHistory"; stage: string; node: number; component: string };

export type ToolAnswer =
  | { ok: true; kind: "nodeHistory"; stage: string; node: number; component: string; values: ArrayLike<number> }
  /** `superseded`: the same tool asked again before this one was answered; the newest request wins */
  | { ok: false; error: string; superseded?: true };

/** What the host gives a tool when it mounts it. */
export interface ToolHost {
  /** the element the tool draws into; the host owns it and its size */
  readonly root: HTMLElement;
  /** ask the model window's store for a change */
  send(event: ToolEvent): void;
  /** ask main for data; a newer request from this tool supersedes an older one still in flight */
  request(q: ToolRequest): Promise<ToolAnswer>;
}

/** A mounted tool. The host calls `update` at most once per frame, with the newest context, and never while hidden. */
export interface ToolHandle {
  update(ctx: ToolContext): void;
  unmount?(): void;
}

export interface ToolDefinition {
  /** the registry key: lowercase letters and dashes */
  id: string;
  /** the window's title */
  title: string;
  cost: ToolCost;
  mount(host: ToolHost): ToolHandle;
}
