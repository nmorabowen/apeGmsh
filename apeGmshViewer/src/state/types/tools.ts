// The tool windows' slice (ADR 0137 D9, D11; H1). The model window holds only what it shows of
// each tool window: its status and why. A tool's own state lives in its window, which holds no store.

/** A tool window's status, as main reports it (src/main/toolWindows.ts). */
export interface ToolStatus {
  /** the window's id ("w1", "w2", …): a window holds tiles of several tools (H4) */
  window: string;
  /** the window's title, for the banner */
  title: string;
  /**
   * opening: made, its page not ready yet; open: linked; unresponsive: its renderer stopped
   * answering; stopped: its renderer died; refused: not opened (memory); closed: the user closed it
   */
  status: "opening" | "open" | "unresponsive" | "stopped" | "refused" | "closed";
  reason: string | null;
}

export interface ToolsState {
  /** the last status of each tool window, by window id */
  windows: Record<string, ToolStatus>;
  /** ADR 0137 D4: under 1 GB of RAM available, tool updates are paused */
  paused: boolean;
  /**
   * the latest request, which an effect (renderer/tools.ts) carries out; `seq` grows per request. H4:
   * a tile of `tool` in the newest tool window (one opens when none is), or `newWindow`: a new window.
   */
  request: { tool: string; seq: number; newWindow: boolean } | null;
}
