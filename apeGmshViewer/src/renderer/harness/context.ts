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

/**
 * Poll (every 20 ms) until `ok` holds, for as long as `progress` keeps changing: false once it has not changed for
 * `stallMs`, or after `maxMs` in all. For work whose length grows with the model (a history scan reads every step),
 * where a fixed total wait passes on a small fixture and fails on a real building that is still making progress.
 */
export async function untilSteady(ok: () => boolean, progress: () => unknown, stallMs = 30000, maxMs = 600000): Promise<boolean> {
  let last = progress(), since = 0;
  for (let t = 0; t < maxMs && since < stallMs; t += 20, since += 20) {
    if (ok()) return true;
    await new Promise((r) => setTimeout(r, 20));
    const now = progress();
    if (now !== last) {
      last = now;
      since = 0;
    }
  }
  return ok();
}
