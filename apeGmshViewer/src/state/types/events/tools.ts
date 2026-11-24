// The tool windows' events (ADR 0137 H1): their part of the store's Event union (types/event.ts).
// Not re-exported by src/state/types.ts, which exports only the whole union.

import type { ToolStatus } from "../tools.ts";

export type ToolsEvent =
  /** Add a tile of `tool` to the newest tool window, opening one when none is; `newWindow`: in a new window (Spotlight). */
  | { type: "openTool"; tool: string; newWindow?: boolean }
  /** Main's report of a tool window's status. */
  | { type: "toolStatus"; status: ToolStatus }
  /** ADR 0137 D4: main paused (or resumed) tool updates for memory. */
  | { type: "toolsPaused"; paused: boolean }
  /** The step bar's playback started or stopped (a tool's context carries it, ADR 0137 D10). */
  | { type: "setPlaying"; on: boolean };
