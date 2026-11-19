// The measure and capture harness (VH1, #2865): what it reads from the renderer's composition root
// (renderer/app.ts builds these five and hands them in). Nothing under harness/ imports app.ts.

import type { Bridge, Effects } from "../../effects.ts";
import type { BlobStore } from "../../state/blobs.ts";
import type { Store } from "../../state/store.ts";
import type { Viewport } from "../viewport.ts";

/** The five the harness drives: the page's store, BlobStore, effects, viewport and main's bridge. */
export interface HarnessContext {
  readonly store: Store;
  readonly blobs: BlobStore;
  readonly effects: Effects;
  readonly viewport: Viewport;
  readonly bridge: Bridge;
  /** VU9b: the combination authoring panel (panels/combine.ts), for its stills; absent before it is mounted */
  readonly combine?: { open(): void; receive(answer: import("../../state/types.ts").CombinationSaveIn): void };
}

/** What main's `config()` answers. */
export type HarnessConfig = Awaited<ReturnType<Bridge["config"]>>;

export const nextFrame = () => new Promise<number>((r) => requestAnimationFrame(r));
/** Resolves once the frame after the current DOM change has been painted. */
export const painted = () => new Promise<void>((r) => requestAnimationFrame(() => setTimeout(r, 0)));

/** Poll the store (every 20 ms, at most `ms`) until `ok` holds; false on timeout. */
export async function until(ok: () => boolean, ms = 15000): Promise<boolean> {
  for (let t = 0; t < ms; t += 20) {
    if (ok()) return true;
    await new Promise((r) => setTimeout(r, 20));
  }
  return ok();
}
