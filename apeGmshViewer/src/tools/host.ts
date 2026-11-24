// The host's scheduling (ADR 0137 D11): `update` runs at most once per frame, with the newest
// context, and never for a hidden tool. Contexts that arrive between frames replace each other;
// a tool that was hidden updates with the newest one when it is shown again. The frame clock and
// the visibility are handed in, so a test drives them.

import type { ToolContext, ToolHandle } from "./contract.ts";

export interface FrameClock {
  request(fn: () => void): void;
}

export class ToolScheduler {
  private readonly tool: ToolHandle;
  private readonly frames: FrameClock;
  private readonly visible: () => boolean;
  /** the newest context not yet drawn */
  private newest: ToolContext | null = null;
  private waiting = false;
  /** how many updates ran (tests and the harness read it) */
  updates = 0;

  constructor(tool: ToolHandle, frames: FrameClock, visible: () => boolean) {
    this.tool = tool;
    this.frames = frames;
    this.visible = visible;
  }

  /** A context arrived: draw it on the next frame, unless a newer one arrives first. */
  push(ctx: ToolContext): void {
    this.newest = ctx;
    this.schedule();
  }

  /** The tool was shown again: draw what arrived while it was hidden. */
  shown(): void {
    this.schedule();
  }

  private schedule(): void {
    if (this.waiting || this.newest === null || !this.visible()) return;
    this.waiting = true;
    this.frames.request(() => {
      this.waiting = false;
      if (this.newest === null || !this.visible()) return;
      const ctx = this.newest;
      this.newest = null;
      this.updates++;
      this.tool.update(ctx);
    });
  }
}
